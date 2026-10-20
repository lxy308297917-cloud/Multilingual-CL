"""Research prototype: fresh student trajectories -> risk-aware sparse AdamW.

No production trainer, GPU launcher, or claims of exact sequence-KL gradients.
All losses are local, stop-gradient-state distillation surrogates.
"""
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Rollout:
    tokens: torch.Tensor
    valid: torch.Tensor
    policy_version: int
    temperature: float = 1.0
    sampler: str = 'full_categorical'

    def require_fresh(self, version):
        if self.policy_version != version:
            raise ValueError('Stale rollout: regenerate after each optimizer update')
        if self.temperature != 1.0 or self.sampler != 'full_categorical':
            raise ValueError('Prototype requires untruncated temperature-1 sampling')
        if self.tokens.ndim != 2 or self.valid.shape != self.tokens.shape:
            raise ValueError('Invalid rollout shape')
        if self.valid.dtype != torch.bool or not self.valid.any(dim=1).all():
            raise ValueError('Each trajectory needs at least one valid answer token')


def logits_of(model, tokens):
    output = model(tokens)
    return output.logits if hasattr(output, 'logits') else output


@torch.no_grad()
def collect_rollout(model, prompt, *, version, max_new_tokens, generator=None, eos_id=None):
    """Uniform-length unpadded prompts. Fresh student; no hint or reference input.

    EOS is included as a valid token; later filler tokens are masked. The caller
    must increment version for every weight-changing update. Metadata alone is
    not cryptographic proof of sampling provenance.
    """
    if prompt.ndim != 2 or prompt.shape[0] == 0 or prompt.shape[1] == 0 or max_new_tokens < 1:
        raise ValueError('Nonempty rank-2 prompts and positive horizon required')
    modes = {m: m.training for m in model.modules()}
    try:
        model.eval()
        context = prompt
        alive = torch.ones(prompt.shape[0], device=prompt.device, dtype=torch.bool)
        tokens, valid = [], []
        for _ in range(max_new_tokens):
            probs = logits_of(model, context)[:, -1].float().softmax(-1)
            token = torch.multinomial(probs, 1, generator=generator)
            tokens.append(token)
            valid.append(alive.clone()[:, None])
            context = torch.cat([context, token], dim=1)
            if eos_id is not None:
                alive = alive & (token[:, 0] != eos_id)
                if not alive.any():
                    break
        return Rollout(torch.cat(tokens, 1), torch.cat(valid, 1), version)
    finally:
        for module, training in modes.items():
            module.training = training


def answer_logits(model, prompt, rollout):
    """Teacher prompt may contain a demonstration; answer suffix stays identical."""
    if prompt.ndim != 2 or prompt.shape[1] < 1 or prompt.shape[0] != rollout.tokens.shape[0]:
        raise ValueError('Prompt/rollout batch mismatch')
    sequence = torch.cat([prompt, rollout.tokens], dim=1)
    start = prompt.shape[1] - 1
    modes = {m: m.training for m in model.modules()}
    try:
        model.eval()  # Match the rollout policy; gradients still enabled for student.
        return logits_of(model, sequence)[:, start:start + rollout.tokens.shape[1]]
    finally:
        for module, training in modes.items():
            module.training = training


def state_kl(student_logits, teacher_logits, valid, *, direction='forward'):
    """Full-vocabulary local KL, per-answer mean then per-example mean.

    Forward: KL(teacher || student); reverse: KL(student || teacher).
    Sampled prefix occupancy and teacher are detached. Neither option includes
    the score-function term for differentiating the prefix sampling distribution.
    """
    if student_logits.shape != teacher_logits.shape or student_logits.ndim != 3:
        raise ValueError('Teacher/student logit alignment mismatch')
    if valid.shape != student_logits.shape[:-1] or valid.dtype != torch.bool:
        raise ValueError('Invalid answer-position mask')
    if not valid.any(dim=1).all():
        raise ValueError('Empty valid answer')
    if not torch.isfinite(student_logits).all() or not torch.isfinite(teacher_logits).all():
        raise FloatingPointError('Nonfinite logits')
    logp = F.log_softmax(student_logits.float(), -1)
    logq = F.log_softmax(teacher_logits.detach().float(), -1)
    if direction == 'forward':
        token_loss = (logq.exp() * (logq - logp)).sum(-1)
    elif direction == 'reverse':
        token_loss = (logp.exp() * (logp - logq)).sum(-1)
    else:
        raise ValueError('Unknown KL direction')
    return ((token_loss * valid).sum(1) / valid.sum(1)).mean()


def gradients(loss, named):
    """Independent FP32 gradients; do not overwrite parameter .grad buffers."""
    values = torch.autograd.grad(loss, tuple(named.values()), allow_unused=False)
    return {n: g.detach().float() for n, g in zip(named, values)}


def exact_mask(score, fraction):
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('Invalid fraction')
    if not torch.isfinite(score).all():
        raise FloatingPointError('Nonfinite selection score')
    count = math.floor(score.numel() * fraction)
    order = torch.argsort(score.flatten(), descending=True, stable=True)
    mask = torch.zeros(score.numel(), dtype=torch.bool, device=score.device)
    mask[order[:count]] = True
    return mask.reshape(score.shape)


class RiskAwareSparseAdamW:
    """Select exact per-matrix quotas using actual proposed AdamW deltas.

    utility_i = -g_new_i * delta_i
    risk_i = max(g_old_i * delta_i, 0) + 0.5*sensitivity_i*delta_i^2
    score_i = utility_i - risk_weight*risk_i

    sensitivity is caller-supplied nonnegative diagonal score-square proxy;
    this is not a certified Hessian/Fisher bound. Fixed per-matrix quotas avoid
    the old LP's module-level coverage collapse. Negative scores are still
    selected if required by the exact budget. No hard retention constraint.
    """
    def __init__(self, named, *, fraction=.15, lr=5e-5, wd=.01,
                 anchor_weight=.1, risk_weight=1., mode='risk'):
        self.named = dict(named)
        if not self.named or any(p.ndim != 2 for p in self.named.values()):
            raise ValueError('Nonempty 2D candidate matrices required')
        for value in (lr, wd, anchor_weight, risk_weight):
            if not math.isfinite(value) or value < 0:
                raise ValueError('Invalid scalar setting')
        if not math.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError('Invalid fraction')
        if mode not in ('risk', 'magnitude'):
            raise ValueError('Unknown selection mode')
        self.settings = dict(fraction=fraction, lr=lr, wd=wd,
                             anchor_weight=anchor_weight, risk_weight=risk_weight, mode=mode)
        self.version = 0
        self.m = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in self.named.items()}
        self.v = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in self.named.items()}

    @torch.no_grad()
    def step(self, new_grad, old_grad, sensitivity, *, new_rollout, old_rollout):
        new_rollout.require_fresh(self.version)
        old_rollout.require_fresh(self.version)
        # Validate every input before mutating any weight or moment.
        for mapping in (new_grad, old_grad, sensitivity):
            if set(mapping) != set(self.named):
                raise ValueError('Gradient/sensitivity identity mismatch')
            for n, p in self.named.items():
                x = mapping[n]
                if x.shape != p.shape or x.device != p.device:
                    raise ValueError('Gradient shape/device mismatch: ' + n)
                if not torch.isfinite(x).all():
                    raise FloatingPointError('Nonfinite signal: ' + n)
        if any((x < 0).any() for x in sensitivity.values()):
            raise ValueError('Negative sensitivity proxy')
        step, s = self.version + 1, self.settings
        proposed = {}
        for n, p in self.named.items():
            combined = new_grad[n].float() + s['anchor_weight'] * old_grad[n].float()
            m = .9 * self.m[n] + .1 * combined
            v = .999 * self.v[n] + .001 * combined.square()
            delta = -s['lr'] * (m / (1 - .9 ** step)) / ((v / (1 - .999 ** step)).sqrt() + 1e-8)
            delta = delta - s['lr'] * s['wd'] * p.float()
            utility = -new_grad[n].float() * delta
            risk = (old_grad[n].float() * delta).clamp_min(0) + .5 * sensitivity[n].float() * delta.square()
            score = utility - s['risk_weight'] * risk if s['mode'] == 'risk' else m.abs()
            mask = exact_mask(score, s['fraction'])
            updated = (p[mask].float() + delta[mask]).to(p.dtype)
            if not torch.isfinite(updated).all():
                raise FloatingPointError('Nonfinite candidate update: ' + n)
            proposed[n] = (m, v, mask, updated, utility, risk)
        audit = {'version': step, 'matrices': {}}
        for n, p in self.named.items():
            m, v, mask, updated, utility, risk = proposed[n]
            previous = p[mask].clone()
            p[mask] = updated
            self.m[n], self.v[n] = m, v
            audit['matrices'][n] = dict(selected=int(mask.sum()), changed=int((previous != updated).sum()),
                                        predicted_utility=float(utility[mask].sum()),
                                        predicted_risk=float(risk[mask].sum()))
        self.version = step
        return audit

    def state_dict(self):
        return dict(version=self.version, settings=self.settings.copy(), names=list(self.named),
                    m={n: x.clone() for n, x in self.m.items()},
                    v={n: x.clone() for n, x in self.v.items()})

    def load_state_dict(self, state):
        if state['settings'] != self.settings or state['names'] != list(self.named):
            raise ValueError('Optimizer identity mismatch')
        if not isinstance(state['version'], int) or state['version'] < 0:
            raise ValueError('Invalid saved version')
        for key in ('m', 'v'):
            if set(state[key]) != set(self.named):
                raise ValueError('Saved moment identity mismatch')
            for n, p in self.named.items():
                x = state[key][n]
                if x.shape != p.shape or x.dtype != torch.float32 or not torch.isfinite(x).all():
                    raise ValueError('Invalid saved moment')
                if key == 'v' and (x < 0).any():
                    raise ValueError('Negative saved second moment')
        self.m = {n: x.to(self.named[n].device).clone() for n, x in state['m'].items()}
        self.v = {n: x.to(self.named[n].device).clone() for n, x in state['v'].items()}
        self.version = state['version']
