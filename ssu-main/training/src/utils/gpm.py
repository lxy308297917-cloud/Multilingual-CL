# ssu_main/training/cl/gpm_llm.py
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn


@dataclass
class GPMConfig:
    # 阈值策略：与 LibContinual 类似，task 越往后阈值略升
    threshold_base: float = 0.97
    threshold_inc: float = 0.003

    # 采样多少条“token 向量”来做 SVD（越大越准，但越耗内存/时间）
    max_tokens_per_layer: int = 4096

    # 是否只对部分 Linear 做 GPM（推荐：先从 mlp / o_proj 开始）
    only_module_name_keywords: Tuple[str, ...] = (
        "mlp", "o_proj", "down_proj", "up_proj", "gate_proj"
    )

    # 数值稳定：SVD 用 float32 更稳
    svd_dtype: torch.dtype = torch.float32


class _ActivationCatcher:
    """
    用 forward hook 抓取某些 Linear 的输入激活 X：
    - Linear: y = x @ W^T
    - 我们需要收集 x（输入空间），构造 A = X^T 做 SVD
    """
    def __init__(self, cfg: GPMConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.handles = []
        self.buffer: Dict[str, torch.Tensor] = {}  # name -> [N, in_dim] token 向量

    def _want_module(self, name: str) -> bool:
        if not self.cfg.only_module_name_keywords:
            return True
        lname = name.lower()
        return any(k in lname for k in self.cfg.only_module_name_keywords)

    def register(self, model: nn.Module):
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and self._want_module(name):
                h = module.register_forward_hook(self._make_hook(name))
                self.handles.append(h)

    def clear(self):
        self.buffer.clear()

    def remove(self):
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles = []

    def _make_hook(self, name: str):
        def hook(module: nn.Module, inputs, outputs):
            # inputs[0] 通常是 x，形状可能是 [B,S,H] 或 [B,H]
            x = inputs[0]
            if x is None:
                return

            # 统一展平成 token 维度：[N, in_dim]
            if x.dim() == 3:
                # [B,S,H] -> [B*S, H]
                x2 = x.reshape(-1, x.shape[-1])
            elif x.dim() == 2:
                x2 = x
            else:
                return

            # 只保留一部分 token，避免爆内存
            x2 = x2.detach()
            if x2.numel() == 0:
                return

            # 采样到 CPU/或保持在 GPU 都可以；这里放 CPU 更省显存
            x2 = x2.to("cpu")

            if name not in self.buffer:
                self.buffer[name] = x2
            else:
                # 追加并截断
                cat = torch.cat([self.buffer[name], x2], dim=0)
                if cat.shape[0] > self.cfg.max_tokens_per_layer:
                    cat = cat[: self.cfg.max_tokens_per_layer]
                self.buffer[name] = cat

        return hook


class GPMState:
    """
    保存/加载的状态：
    - basis[name] = U: [in_dim, r]
    - task_idx: 当前已完成到哪个 task
    """
    def __init__(self):
        self.basis: Dict[str, torch.Tensor] = {}
        self.task_idx: int = -1

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cpu_basis = {k: v.detach().to("cpu") for k, v in self.basis.items()}
        torch.save({"task_idx": self.task_idx, "basis": cpu_basis}, path)

    @staticmethod
    def load(path: str) -> "GPMState":
        st = GPMState()
        obj = torch.load(path, map_location="cpu")
        st.task_idx = int(obj.get("task_idx", -1))
        st.basis = obj.get("basis", {})
        return st


class GPMManager:
    """
    LLM 版 GPM 管理器（对齐 LibContinual 的 before_task/after_task/observe 思路）：
    - before_task: 生成每层的投影矩阵 P = U U^T
    - training step: 在 optimizer step 前投影梯度
    - after_task: 用当前任务样本前向收集激活，更新每层子空间 U
    """
    def __init__(self, model: nn.Module, device: torch.device, cfg: Optional[GPMConfig] = None):
        self.model = model
        self.device = device
        self.cfg = cfg or GPMConfig()

        self.state = GPMState()
        self.proj: Dict[str, torch.Tensor] = {}  # name -> P: [in_dim, in_dim]（放在 GPU 方便乘）

        self.catcher = _ActivationCatcher(self.cfg, device)

    def maybe_load(self, state_path: str):
        if os.path.exists(state_path):
            self.state = GPMState.load(state_path)

    def before_task(self, task_idx: int):
        # 根据已保存的 basis 生成投影矩阵 P = U U^T
        self.proj.clear()
        for name, U in self.state.basis.items():
            # U: [in_dim, r]
            U = U.to(self.device, dtype=torch.float32)
            P = U @ U.T  # [in_dim, in_dim]
            self.proj[name] = P

    @torch.no_grad()
    def project_gradients(self):
        """
        在 optimizer.step() 之前调用：
        对每个目标 Linear，执行：grad = grad - grad @ P
        其中 grad 视作 [out_dim, in_dim]，P 是 [in_dim, in_dim]
        """
        if not self.proj:
            return

        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if name not in self.proj:
                continue
            if module.weight.grad is None:
                continue

            g = module.weight.grad.data  # [out, in]
            P = self.proj[name]

            # 确保形状匹配
            if g.dim() != 2 or P.dim() != 2:
                continue
            if g.shape[1] != P.shape[0]:
                continue

            module.weight.grad.data = g - (g @ P)

    @torch.no_grad()
    def after_task_update_basis(self, task_idx: int, train_dataloader, max_batches: int = 20):
        """
        task 结束后更新 basis（对应 LibContinual 的 after_task）：
        1) 用少量 batch 前向，hook 收集各层输入激活 X
        2) 对每层构造 A = X^T（in_dim, N）
        3) SVD 按阈值选 r，更新 U
        """
        self.model.eval()
        self.catcher.clear()
        self.catcher.register(self.model)

        # 1) 跑少量 batch 收集激活
        seen = 0
        for batch_idx, batch in enumerate(train_dataloader):
            if batch_idx >= max_batches:
                break

            # 兼容 HF 的 batch：input_ids / attention_mask / labels ...
            batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
            _ = self.model(**batch)
            seen += 1

        self.catcher.remove()

        # 2) 逐层更新 basis
        threshold = self.cfg.threshold_base + task_idx * self.cfg.threshold_inc

        for name, X in self.catcher.buffer.items():
            # X: [N, in_dim] on CPU
            if X is None or X.numel() == 0:
                continue

            X = X.to(dtype=self.cfg.svd_dtype)
            A = X.T  # [in_dim, N]

            # 为了数值稳定/速度：A 可能很大，但 N 已经被 max_tokens_per_layer 限制
            # SVD: A = U S V^T
            U, S, Vh = torch.linalg.svd(A, full_matrices=False)

            # 能量占比选择 r（对齐 LibContinual 的 Eq-5 思路）
            sval_total = torch.sum(S * S)
            sval_ratio = (S * S) / (sval_total + 1e-12)
            cumsum = torch.cumsum(sval_ratio, dim=0)

            if name not in self.state.basis:
                # 第一个 task：直接取前 r 个方向
                r = int(torch.sum(cumsum < threshold).item())
                r = max(r, 1)
                self.state.basis[name] = U[:, :r].detach().cpu()
            else:
                # 后续 task：先做残差投影 act_hat = A - U_old U_old^T A，再更新
                U_old = self.state.basis[name].to(dtype=self.cfg.svd_dtype)  # CPU
                P_old = U_old @ U_old.T
                A_hat = A - (P_old @ A)

                U2, S2, Vh2 = torch.linalg.svd(A_hat, full_matrices=False)
                sval_hat = torch.sum(S2 * S2)

                accumulated = (sval_total - sval_hat) / (sval_total + 1e-12)

                if accumulated >= threshold:
                    # 和 LibContinual 一样：已经覆盖阈值，不更新
                    continue
                else:
                    # 需要补充新方向
                    # r = 使得 (旧覆盖 + 新覆盖) 达到阈值
                    # 近似实现：找最小 r 使 accumulated + cumsum(r) >= threshold
                    r = int(torch.sum(cumsum + accumulated < threshold).item()) + 1
                    r = max(r, 1)

                    U_new = torch.cat([U_old, U2[:, :r].cpu()], dim=1)
                    # U_new 的列数不能超过行数
                    U_new = U_new[:, : min(U_new.shape[0], U_new.shape[1])]
                    self.state.basis[name] = U_new.detach().cpu()

        self.state.task_idx = task_idx

    def save(self, state_path: str):
        self.state.save(state_path)


# ssu_main/training/cl/gpm_callback.py
from transformers import TrainerCallback

class GPMCallback(TrainerCallback):
    def __init__(self, gpm_manager):
        self.gpm = gpm_manager

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        # 在 optimizer.step() 之前投影梯度（等价于 LibContinual 的 observe 里投影）
        self.gpm.project_gradients()
        return control