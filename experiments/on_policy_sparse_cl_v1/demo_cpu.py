"""Synthetic two-stage causal-model mechanics demo. No task-quality evidence."""
import argparse
import copy
import json
from pathlib import Path

import torch
from torch import nn

from core import (RiskAwareSparseAdamW, answer_logits, collect_rollout,
                  gradients, state_kl)


class TinyCausalLM(nn.Module):
    def __init__(self, vocab=8, hidden=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab, hidden)
        self.embedding.weight.requires_grad_(False)
        self.output = nn.Linear(hidden, vocab, bias=False)

    def forward(self, tokens):
        features = self.embedding(tokens).cumsum(1)
        lengths = torch.arange(1, tokens.shape[1] + 1, device=tokens.device)[None, :, None]
        return self.output(features / lengths)


def score_square_proxy(model, prompt, rollout, named):
    """Mean squared per-sequence score gradient; uncalibrated diagonal proxy."""
    logits = answer_logits(model, prompt, rollout)
    sampled_logp = logits.float().log_softmax(-1).gather(-1, rollout.tokens[..., None]).squeeze(-1)
    scores = (sampled_logp * rollout.valid).sum(1)
    result = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in named.items()}
    for row, score in enumerate(scores):
        gs = torch.autograd.grad(score, tuple(named.values()), retain_graph=row + 1 < len(scores))
        for n, g in zip(named, gs):
            result[n].add_(g.detach().float().square(), alpha=1 / len(scores))
    return result


def run(steps=8, mode='risk'):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    student = TinyCausalLM()
    base = copy.deepcopy(student).eval().requires_grad_(False)
    teacher = copy.deepcopy(base)
    named = {'output.weight': student.output.weight}
    optimizer = RiskAwareSparseAdamW(named, fraction=.15, lr=.01, mode=mode)
    sampler = torch.Generator(device='cpu').manual_seed(43)
    trace = []
    for stage in range(2):
        # Preserve the previous completed stage, not just the original Base.
        previous = copy.deepcopy(student).eval().requires_grad_(False)
        new_prompt = torch.tensor([[stage + 1, 2], [stage + 1, 3]])
        old_prompt = torch.tensor([[0, 2], [0, 3]])
        if stage:
            old_prompt = torch.cat([old_prompt, torch.tensor([[1, 2], [1, 3]])], dim=0)
        # This token is only a synthetic privileged context, not a real demonstration.
        hinted_prompt = torch.cat([torch.full((2, 1), 6 - stage), new_prompt], dim=1)
        for _ in range(steps):
            nr = collect_rollout(student, new_prompt, version=optimizer.version,
                                 max_new_tokens=3, generator=sampler)
            er = collect_rollout(student, old_prompt, version=optimizer.version,
                                 max_new_tokens=3, generator=sampler)
            new_loss = state_kl(answer_logits(student, new_prompt, nr),
                                answer_logits(teacher, hinted_prompt, nr), nr.valid)
            ng = gradients(new_loss, named)
            # Base and previous-stage reference remain distinct, task-balanced anchors.
            old_student = answer_logits(student, old_prompt, er)
            old_loss = .5 * state_kl(old_student, answer_logits(base, old_prompt, er), er.valid)
            old_loss += .5 * state_kl(old_student, answer_logits(previous, old_prompt, er), er.valid)
            og = gradients(old_loss, named)
            sensitivity = score_square_proxy(student, old_prompt, er, named)
            audit = optimizer.step(ng, og, sensitivity, new_rollout=nr, old_rollout=er)
            trace.append(dict(stage=stage + 1, new_kl_before=float(new_loss.detach()),
                              old_kl_before=float(old_loss.detach()), **audit))
    return dict(status='synthetic_cpu_mechanics_only', mode=mode, steps=optimizer.version,
                gpu_used=False, trace=trace)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps-per-stage', type=int, default=8)
    parser.add_argument('--mode', choices=['risk', 'magnitude'], default='risk')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.steps_per_stage < 1:
        parser.error('Positive steps required')
    result = run(args.steps_per_stage, args.mode)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'trace'}))
    print('No downstream scores: this checks sampling, alignment, gradients and sparse writes only.')
