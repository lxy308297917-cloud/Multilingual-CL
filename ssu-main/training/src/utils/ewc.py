import os
import torch
from dataclasses import dataclass
from typing import Dict, Optional, Iterable, Any
from transformers import PreTrainedModel


@dataclass
class EWCConfig:
    """
    EWC 的超参数配置
    """
    lambda_ewc: float = 1.0
    fisher_max_batches: Optional[int] = None
    fisher_use_token_count: bool = False
    fisher_decay: float = 0.95


class EWC:
    """
    适用于 LLM 的 EWC（Elastic Weight Consolidation）

    核心思想：
    1. ref_param：保存上一个任务结束时的参数快照 θ*
    2. fisher：估计 Fisher 信息矩阵（用梯度平方的期望）
    3. penalty：在新任务训练中加入 Σ F_i (θ_i - θ*_i)^2 / 2
    """

    def __init__(
        self,
        model: PreTrainedModel,
        device: torch.device,
        cfg: Optional[EWCConfig] = None,
    ):
        self.device = device
        self.cfg = cfg or EWCConfig()

        # 只对当前可训练参数做 EWC
        self.param_names = [n for n, p in model.named_parameters() if p.requires_grad]

        # Task 1 has no previous task to protect. Allocate no duplicate model
        # state until Fisher is estimated at the task boundary. This removes an
        # unnecessary multi-GB memory penalty during the first task.
        self.ref_param: Dict[str, torch.Tensor] = {}
        self.fisher: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update_ref_param(self, model: PreTrainedModel):
        """
        在任务结束时更新参考参数 θ*
        """
        for n, p in model.named_parameters():
            if n in self.param_names:
                self.ref_param[n] = p.detach().clone().to(self.device)

    def estimate_fisher(
        self,
        model: PreTrainedModel,
        dataloader: Iterable[Dict[str, Any]],
    ):
        """
        使用当前任务的数据估计 Fisher 信息矩阵
        """
        model.eval()

        name_to_param = {
            n: p
            for n, p in model.named_parameters()
            if n in self.param_names and p.requires_grad
        }
        # Accumulate squared gradients in float32, then store the online Fisher
        # in bfloat16. BF16 retains FP32's exponent range and halves persistent
        # state memory, while FP32 accumulation avoids precision loss over many
        # calibration batches.
        fisher_new = {
            n: torch.zeros_like(p, dtype=torch.float32, device=self.device)
            for n, p in name_to_param.items()
        }

        num_batches = 0
        max_batches = self.cfg.fisher_max_batches

        for step, batch in enumerate(dataloader):
            if max_batches is not None and step >= max_batches:
                break

            model.zero_grad(set_to_none=True)
            batch = {k: v.to(self.device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            num_batches += 1

            for n, p in model.named_parameters():
                if n in fisher_new and p.grad is not None:
                    fisher_new[n].add_(p.grad.detach().float().square())

        denom = max(num_batches, 1)
        for n in fisher_new:
            fisher_new[n] /= denom

        # Online EWC: F_total = gamma * F_old + F_new. Keep persistent
        # Fisher tensors in BF16 so full-model EWC remains feasible on 32GB GPUs.
        old_fisher = self.fisher
        updated_fisher: Dict[str, torch.Tensor] = {}
        gamma = self.cfg.fisher_decay
        for n, current in fisher_new.items():
            if gamma > 0.0 and n in old_fisher:
                # Historical state may be offloaded to CPU at a task boundary.
                # Move one tensor at a time to keep peak GPU memory bounded.
                previous = old_fisher[n].to(
                    device=current.device, dtype=torch.float32
                )
                current.add_(previous, alpha=gamma)
                del previous
            updated_fisher[n] = current.to(dtype=torch.bfloat16)
        self.fisher = updated_fisher
        del fisher_new

        with torch.no_grad():
            vals = [f.float().mean().item() for f in self.fisher.values()]
            if len(vals) > 0:
                print(f"[EWC] Fisher batches = {num_batches}")
                print(f"[EWC] Fisher mean avg = {sum(vals) / len(vals):.6e}")
                print(f"[EWC] Fisher mean max = {max(vals):.6e}")
                print(f"[EWC] Fisher mean min = {min(vals):.6e}")

        model.zero_grad(set_to_none=True)
        model.train()

    def offload_state_to_cpu(self):
        """Move historical EWC state off GPU before boundary Fisher estimation."""
        self.ref_param = {
            n: tensor.detach().cpu() for n, tensor in self.ref_param.items()
        }
        self.fisher = {
            n: tensor.detach().cpu() for n, tensor in self.fisher.items()
        }

    def penalty(self, model: PreTrainedModel) -> torch.Tensor:
        """
        计算 EWC 正则项
        """
        loss = torch.tensor(0.0, device=self.device)

        for n, p in model.named_parameters():
            if n in self.fisher:
                loss = loss + (
                    self.fisher[n] * (p - self.ref_param[n]).pow(2)
                ).sum() / 2.0

        return self.cfg.lambda_ewc * loss

    def save(self, save_path: str):
        """
        保存 EWC 状态：ref_param + fisher
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        ref_cpu = {k: v.detach().cpu() for k, v in self.ref_param.items()}
        fisher_cpu = {k: v.detach().cpu() for k, v in self.fisher.items()}

        state = {
            "param_names": list(self.param_names),
            "ref_param": ref_cpu,
            "fisher": fisher_cpu,
            "cfg": {
                "lambda_ewc": float(self.cfg.lambda_ewc),
                "fisher_max_batches": self.cfg.fisher_max_batches,
                "fisher_use_token_count": bool(self.cfg.fisher_use_token_count),
                "fisher_decay": float(self.cfg.fisher_decay),
            },
        }

        torch.save(state, save_path)

    def load(self, load_path: str, model: PreTrainedModel):
        """
        加载 EWC 状态，并对齐到当前模型的可训练参数
        """
        state = torch.load(load_path, map_location="cpu")

        cur_param_names = [n for n, p in model.named_parameters() if p.requires_grad]
        self.param_names = cur_param_names

        loaded_ref = state.get("ref_param", {})
        loaded_fisher = state.get("fisher", {})

        new_ref: Dict[str, torch.Tensor] = {}
        new_fisher: Dict[str, torch.Tensor] = {}

        name_to_param = {
            n: p for n, p in model.named_parameters()
            if n in cur_param_names
        }

        for n, p in name_to_param.items():
            if (n in loaded_ref) and (n in loaded_fisher):
                ref_t = loaded_ref[n]
                fish_t = loaded_fisher[n]

                if tuple(ref_t.shape) == tuple(p.shape) and tuple(fish_t.shape) == tuple(p.shape):
                    new_ref[n] = ref_t.to(self.device, dtype=p.dtype)
                    new_fisher[n] = fish_t.to(self.device, dtype=torch.bfloat16)
                else:
                    new_ref[n] = p.detach().clone().to(self.device)
                    new_fisher[n] = torch.zeros_like(p, device=self.device)
            else:
                new_ref[n] = p.detach().clone().to(self.device)
                new_fisher[n] = torch.zeros_like(p, device=self.device)

        self.ref_param = new_ref
        self.fisher = new_fisher