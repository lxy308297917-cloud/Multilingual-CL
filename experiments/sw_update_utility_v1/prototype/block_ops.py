"""Numerical primitives only; not yet a released training optimizer."""
import math
import torch

def block_sum(x, size=64):
    if x.ndim != 2 or size <= 0 or any(n % size for n in x.shape):
        raise ValueError('Require a matrix tiled exactly by equal-size blocks')
    return x.float().reshape(x.shape[0]//size,size,x.shape[1]//size,size).sum((1,3))

def utility_scores(target_grad, source_grad, delta, risk_lambda=1., size=64):
    if target_grad.shape != source_grad.shape or delta.shape != target_grad.shape:
        raise ValueError('Gradient/delta shape mismatch')
    if not math.isfinite(risk_lambda) or risk_lambda < 0:
        raise ValueError('Invalid risk penalty')
    utility = -block_sum(target_grad.float()*delta.float(),size)
    risk = block_sum(source_grad.float()*delta.float(),size).clamp_min(0)
    score = utility-risk_lambda*risk
    if not torch.isfinite(score).all():
        raise FloatingPointError('Nonfinite block scores')
    return score,utility,risk

def select_equal_blocks(scores, fraction, positive_only=False):
    """Global exact upper budget; ties broken by input tensor order then block index."""
    if not scores or not 0 <= fraction <= 1:
        raise ValueError('Invalid scores or fraction')
    flat=torch.cat([s.flatten() for s in scores])
    if not torch.isfinite(flat).all():
        raise FloatingPointError('Nonfinite selection scores')
    k=math.floor(flat.numel()*fraction)
    order=torch.argsort(flat,descending=True,stable=True)
    chosen=order[:k]
    if positive_only:chosen=chosen[flat[chosen]>0]
    selected=torch.zeros_like(flat,dtype=torch.bool);selected[chosen]=True
    return [v.reshape(s.shape) for s,v in zip(scores,selected.split([s.numel() for s in scores]))]

@torch.no_grad()
def apply_delta(param, delta, blocks, size=64):
    if param.shape != delta.shape or tuple(param.shape) != (blocks.shape[0]*size,blocks.shape[1]*size):
        raise ValueError('Delta/mask shape mismatch')
    mask=blocks.repeat_interleave(size,0).repeat_interleave(size,1)
    if not torch.isfinite(delta).all():raise FloatingPointError('Nonfinite delta')
    # Index assignment guarantees that inactive coordinates are not even rounded again.
    param[mask]=(param[mask].float()+delta[mask].float()).to(param.dtype)

def adamw_candidate(param, grad, first, second, step, lr, betas=(.9,.999), eps=1e-8, weight_decay=.01):
    """Pure FP32 candidate; returns next states, never mutates parameters or inputs."""
    if step<1 or lr<0 or eps<=0:raise ValueError('Invalid optimizer arguments')
    if any(t.shape!=param.shape for t in (grad,first,second)):raise ValueError('State shape mismatch')
    b1,b2=betas
    if not (0<=b1<1 and 0<=b2<1):raise ValueError('Invalid betas')
    m=first.float()*b1+grad.float()*(1-b1)
    v=second.float()*b2+grad.float().square()*(1-b2)
    delta=-lr*(m/(1-b1**step))/((v/(1-b2**step)).sqrt()+eps)-lr*weight_decay*param.float()
    if not torch.isfinite(delta).all():raise FloatingPointError('Nonfinite AdamW candidate')
    return delta,m,v

def allocate_budget(capacities, total, weights=None):
    """Capped largest-remainder integer allocation; deterministic ties by group order."""
    if any(type(c)!=int or c<0 for c in capacities) or type(total)!=int or not 0<=total<=sum(capacities):
        raise ValueError('Invalid capacity/budget')
    weights=list(capacities if weights is None else weights)
    if len(weights)!=len(capacities) or any(not math.isfinite(w) or w<0 for w in weights):
        raise ValueError('Invalid weights')
    allocation=[0]*len(capacities);remaining=total
    while remaining:
        active=[i for i,c in enumerate(capacities) if allocation[i]<c]
        w=[weights[i] for i in active]
        if sum(w)==0:w=[1.]*len(active)
        quota={i:remaining*wi/sum(w) for i,wi in zip(active,w)}
        capped=[i for i in active if quota[i]>=capacities[i]-allocation[i]]
        if capped:
            for i in capped:
                take=capacities[i]-allocation[i];allocation[i]+=take;remaining-=take
            continue
        floors={i:math.floor(quota[i]) for i in active}
        for i,n in floors.items():allocation[i]+=n;remaining-=n
        order=sorted(active,key=lambda i:(-(quota[i]-floors[i]),i))
        for i in order[:remaining]:allocation[i]+=1
        remaining=0
    assert sum(allocation)==total and all(a<=c for a,c in zip(allocation,capacities))
    return allocation
