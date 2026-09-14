import copy
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel


@dataclass
class LwFConfig:
    temperature: float = 2.0
    lambda_lwf: float = 1.0


class LwF:
    """
    LwF for LLM continual pretraining (CPT):
    - teacher: frozen old model (same tokenizer/vocab)
    - student: current model
    - loss: lambda * KL(softmax(teacher/T) || softmax(student/T)) * T^2
    """

    def __init__(
        self,
        teacher_model: PreTrainedModel,
        cfg: Optional[LwFConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.cfg = cfg or LwFConfig()

        # Deepcopy is safer: avoid weight sharing / accidental updates
        self.teacher = copy.deepcopy(teacher_model)
        self.teacher.eval()

        for p in self.teacher.parameters():
            p.requires_grad = False

        # 关键修改：
        # main_bf16.py 使用 flash_attention_2 时，teacher 不能是 fp32
        # FlashAttention 只支持 fp16 / bf16，所以这里强制 teacher 为 bf16
        if device is not None:
            self.teacher.to(device=device, dtype=torch.bfloat16)
        else:
            self.teacher.to(dtype=torch.bfloat16)

        # teacher 不需要 cache，避免额外显存和 checkpointing 相关问题
        if hasattr(self.teacher, "config"):
            self.teacher.config.use_cache = False

    @torch.no_grad()
    def _teacher_logits(self, inputs: Dict[str, Any]) -> torch.Tensor:
        """
        Run teacher forward. Make sure labels do not affect teacher forward.
        For main_bf16.py + flash_attention_2, teacher forward must stay in bf16.
        """
        device = next(self.teacher.parameters()).device

        teacher_inputs = {}
        for k, v in inputs.items():
            if k == "labels":
                continue
            if torch.is_tensor(v):
                teacher_inputs[k] = v.to(device)
            else:
                teacher_inputs[k] = v

        self.teacher.eval()

        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = self.teacher(**teacher_inputs)
        else:
            out = self.teacher(**teacher_inputs)

        return out.logits.detach()

    def kd_loss(
        self,
        student_logits: torch.Tensor,
        inputs: Dict[str, Any],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute KD loss. If attention_mask is provided, we compute token-level KL
        and mask out padding tokens.
        """
        T = float(self.cfg.temperature)
        teacher_logits = self._teacher_logits(inputs)

        # forward 可以是 bf16；KL 计算前转 fp32，数值更稳定
        student_logits = student_logits.float()
        teacher_logits = teacher_logits.float()

        # [B, L, V] for causal LM
        if student_logits.dim() == 3:
            s = F.log_softmax(student_logits / T, dim=-1)
            t = F.softmax(teacher_logits / T, dim=-1)

            # KL per token: [B, L]
            kl = F.kl_div(s, t, reduction="none").sum(dim=-1)

            if attention_mask is not None:
                mask = attention_mask.to(kl.dtype)
                kl = (kl * mask).sum() / mask.sum().clamp_min(1.0)
            else:
                kl = kl.mean()

            return self.cfg.lambda_lwf * kl * (T * T)

        # [B, V] for classification-style logits
        return self.cfg.lambda_lwf * F.kl_div(
            F.log_softmax(student_logits / T, dim=-1),
            F.softmax(teacher_logits / T, dim=-1),
            reduction="batchmean",
        ) * (T * T)