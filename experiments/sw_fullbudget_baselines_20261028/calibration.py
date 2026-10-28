"""Sequential calibration/replay, restoring RNG and every module's mode."""
from contextlib import contextmanager
import random
import time
import numpy as np
import torch

@contextmanager
def calibration_mode(model):
    states = [(module, module.training) for module in model.modules()]
    state = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
             torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)
    try:
        model.eval()
        yield
    finally:
        for module, training in states:
            module.training = training
        random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])
        if state[3] is not None:
            torch.cuda.set_rng_state_all(state[3])

def collect(model, named, rows, loss_function, replay=False):
    """Call before main backward, when grad buffers are empty.

    Non-replay: return CPU FP32 estimates, clear all calibration grads.
    Replay: retain0.1x auxiliary grads, then add ordinary main gradients and clip
    once in the caller. This is the intended replay training objective.
    """
    assert rows and all(p.grad is None for p in named.values())
    started = time.monotonic()
    value = 0.
    with calibration_mode(model):
        for row in rows:
            loss = loss_function(row)/len(rows)
            value += float(loss.detach())
            loss.backward()
            del loss
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in named.values())
    raw_sq = sum(float(p.grad.float().square().sum()) for p in named.values())
    if replay:
        for p in named.values():
            p.grad.mul_(.1)
        gradients = None
    else:
        gradients = {n: p.grad.detach().to(device='cpu', dtype=torch.float32).clone()
                     for n, p in named.items()}
        model.zero_grad(set_to_none=True)
    return gradients, dict(CE=value, raw_gradient_norm=raw_sq**.5,
                           auxiliary_multiplier=.1 if replay else None,
                           input_tokens=sum(len(r['input_ids']) for r in rows),
                           supervised_tokens=sum(sum(y != -100 for y in r['labels'][1:]) for r in rows),
                           seconds=time.monotonic()-started, n=len(rows),
                           ids=[r['id'] for r in rows], RNG_and_modes_restored=True)
