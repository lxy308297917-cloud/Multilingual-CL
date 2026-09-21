"""Exact global or fixed-layer block selection; prototype, not training release."""
import math
import torch
from block_ops import allocate_budget, select_equal_blocks


def select_grouped(scores, groups, fraction, policy='global', weights=None, positive_only=False):
    if not scores or len(scores) != len(groups) or policy not in ('global', 'fixed'):
        raise ValueError('Invalid grouped selection')
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('Invalid fraction')
    if any(not torch.isfinite(s).all() for s in scores):
        raise FloatingPointError('Nonfinite selection scores')
    keys=list(dict.fromkeys(groups))
    capacities=[sum(s.numel() for s,g in zip(scores,groups) if g==key) for key in keys]
    total=math.floor(sum(capacities)*fraction)
    if policy=='global':
        if weights is not None:raise ValueError('Global adaptive selection does not use layer weights')
        masks=select_equal_blocks(scores,fraction,positive_only)
        quotas=None
    else:
        if weights is not None and set(weights)!=set(keys):raise ValueError('Layer weight identity mismatch')
        quotas=allocate_budget(capacities,total,None if weights is None else [weights[k] for k in keys])
        masks=[torch.zeros_like(s,dtype=torch.bool) for s in scores]
        for key,quota in zip(keys,quotas):
            indices=[i for i,g in enumerate(groups) if g==key]
            flat=torch.cat([scores[i].flatten() for i in indices])
            chosen=torch.argsort(flat,descending=True,stable=True)[:quota]
            if positive_only:chosen=chosen[flat[chosen]>0]
            selected=torch.zeros_like(flat,dtype=torch.bool);selected[chosen]=True
            # No redistribution of unused positive-only quota: preserves fixed-layer control.
            for i,part in zip(indices,selected.split([scores[i].numel() for i in indices])):
                masks[i]=part.reshape(scores[i].shape)
    actual=[sum(int(m.sum()) for m,g in zip(masks,groups) if g==key) for key in keys]
    return masks,dict(groups=keys,capacities=capacities,budget_blocks=total,quotas=quotas,
                      selected_per_group=actual,selected_blocks=sum(actual),unused_blocks=total-sum(actual))
