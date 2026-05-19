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

        # Deepcopy is safer (avoid weight sharing / accidental updates)
        self.teacher = copy.deepcopy(teacher_model)
        self.teacher.eval()

        for p in self.teacher.parameters():
            p.requires_grad = False

        # 注意：这里不要放在 for 循环里，也先不要用 float16
        if device is not None:
            self.teacher.to(device=device, dtype=torch.float32)

    @torch.no_grad()
    def _teacher_logits(self, inputs: Dict[str, Any]) -> torch.Tensor:
        """
        Run teacher forward. Make sure labels do not affect teacher forward.
        """
        teacher_inputs = dict(inputs)
        teacher_inputs.pop("labels", None)
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

        # 为了数值稳定，KL 前统一转 float32
        student_logits = student_logits.float()
        teacher_logits = teacher_logits.float()

        # [B, L, V] for causal LM, or [B, V] for classification
        if student_logits.dim() == 3:
            # token-level KL
            s = F.log_softmax(student_logits / T, dim=-1)
            t = F.softmax(teacher_logits / T, dim=-1)

            # KL per token: sum over vocab -> [B, L]
            kl = F.kl_div(s, t, reduction="none").sum(dim=-1)

            if attention_mask is not None:
                mask = attention_mask.to(kl.dtype)
                kl = (kl * mask).sum() / mask.sum().clamp_min(1.0)
            else:
                kl = kl.mean()

            return self.cfg.lambda_lwf * kl * (T * T)

        else:
            # batch-level KL for [B, V]
            return self.cfg.lambda_lwf * F.kl_div(
                F.log_softmax(student_logits / T, dim=-1),
                F.softmax(teacher_logits / T, dim=-1),
                reduction="batchmean",
            ) * (T * T)