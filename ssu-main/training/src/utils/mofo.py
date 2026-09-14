import math

import torch
from torch.optim import Optimizer
from transformers import Trainer


class MoFOAdamW(Optimizer):
    """AdamW whose per-tensor updates keep only top momentum magnitudes."""

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        update_fraction=0.15,
    ):
        if not 0.0 < update_fraction <= 1.0:
            raise ValueError("update_fraction must be in (0, 1]")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            update_fraction=update_fraction,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            fraction = group["update_fraction"]

            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("MoFOAdamW does not support sparse gradients")

                state = self.state[param]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                numel = exp_avg.numel()
                keep = min(numel, max(1, int(math.ceil(numel * fraction))))
                if keep == numel:
                    mask = torch.ones_like(exp_avg, dtype=torch.bool)
                else:
                    flat_abs = exp_avg.abs().reshape(-1)
                    threshold = torch.kthvalue(flat_abs, numel - keep + 1).values
                    mask = exp_avg.abs().ge(threshold)

                if weight_decay:
                    param.add_(param * mask, alpha=-lr * weight_decay)

                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
                filtered_momentum = exp_avg * mask
                param.addcdiv_(
                    filtered_momentum,
                    denom,
                    value=-lr / bias_correction1,
                )
        return loss


class MoFOTrainer(Trainer):
    mofo_update_fraction = 0.15

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        decay_names = self.get_decay_parameter_names(self.model)
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if p.requires_grad and n in decay_names
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if p.requires_grad and n not in decay_names
                ],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = MoFOAdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
            update_fraction=self.mofo_update_fraction,
        )
        return self.optimizer
