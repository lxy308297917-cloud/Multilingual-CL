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
    fisher_use_token_count: bool = True
    fisher_decay: float = 0.0


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

        self.ref_param: Dict[str, torch.Tensor] = {
            n: p.detach().clone().to(device)
            for n, p in model.named_parameters()
            if n in self.param_names
        }

        self.fisher: Dict[str, torch.Tensor] = {
            n: torch.zeros_like(self.ref_param[n], device=device)
            for n in self.param_names
        }

    @torch.no_grad()
    def update_ref_param(self, model: PreTrainedModel):
        """
        在任务结束时更新参考参数 θ*
        """
        for n, p in model.named_parameters():
            if n in self.ref_param:
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

        fisher_new = {
            n: torch.zeros_like(v, device=self.device)
            for n, v in self.ref_param.items()
        }

        total_units = 0.0
        max_batches = self.cfg.fisher_max_batches

        for step, batch in enumerate(dataloader):
            if max_batches is not None and step >= max_batches:
                break

            model.zero_grad(set_to_none=True)
            batch = {k: v.to(self.device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            if self.cfg.fisher_use_token_count and "attention_mask" in batch:
                units = float(batch["attention_mask"].sum().item())
                units = max(units, 1.0)
            else:
                units = 1.0

            total_units += units

            for n, p in model.named_parameters():
                if n in fisher_new and p.grad is not None:
                    fisher_new[n] += (p.grad.detach() ** 2) * units

        denom = max(total_units, 1.0)
        for n in fisher_new:
            fisher_new[n] /= denom

        if self.cfg.fisher_decay > 0.0:
            alpha = self.cfg.fisher_decay
            for n in self.fisher:
                self.fisher[n] = alpha * self.fisher[n] + (1.0 - alpha) * fisher_new[n]
        else:
            self.fisher = fisher_new

        model.train()

    def penalty(self, model: PreTrainedModel) -> torch.Tensor:
        """
        计算 EWC 正则项
        """
        loss = torch.tensor(0.0, device=self.device)
        for n, p in model.named_parameters():
            if n in self.fisher:
                loss = loss + (self.fisher[n] * (p - self.ref_param[n]).pow(2)).sum() / 2.0

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

        name_to_param = {n: p for n, p in model.named_parameters() if n in cur_param_names}

        for n, p in name_to_param.items():
            if (n in loaded_ref) and (n in loaded_fisher):
                ref_t = loaded_ref[n]
                fish_t = loaded_fisher[n]

                if tuple(ref_t.shape) != tuple(p.shape) or tuple(fish_t.shape) != tuple(p.shape):
                    continue

                new_ref[n] = ref_t.to(self.device)
                new_fisher[n] = fish_t.to(self.device)
            else:
                new_ref[n] = p.detach().clone().to(self.device)
                new_fisher[n] = torch.zeros_like(p, device=self.device)

        self.ref_param = new_ref
        self.fisher = new_fisher