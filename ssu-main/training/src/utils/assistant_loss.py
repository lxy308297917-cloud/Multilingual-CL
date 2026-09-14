"""Assistant-only causal-LM loss shared by v3 task-mixture training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def per_sequence_causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Average active-token CE inside each sequence, then average sequences.

    ``labels == -100`` marks prompt and padding tokens.  A sequence without an
    assistant target is rejected rather than silently changing the batch weight.
    """
    shift_logits = logits[..., :-1, :].contiguous().float()
    shift_labels = labels[..., 1:].contiguous()
    token_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(shift_labels)
    active = shift_labels.ne(-100)
    counts = active.sum(dim=-1)
    if torch.any(counts == 0):
        indices = torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()
        raise ValueError(f"Sequences without assistant labels: {indices}")
    sequence_loss = (token_loss * active).sum(dim=-1) / counts
    return sequence_loss.mean()
