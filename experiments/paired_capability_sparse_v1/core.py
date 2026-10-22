"""Sparse-support projection of actual AdamW displacements (research prototype).

All candidate moments advance once from target gradients; calibration gradients
never enter moments. Small joint QPs use FP64 sufficient statistics. No claim of
non-forgetting: checks concern the current linear surrogate only.
"""
import contextlib
import itertools
import math
import random
import numpy as np
import torch


@contextlib.contextmanager
def calibration_context(model):
    modes = [(m, m.training) for m in model.modules()]
    py, npstate = random.getstate(), np.random.get_state()
    devices = sorted({p.device.index for p in model.parameters() if p.is_cuda})
    try:
        with torch.random.fork_rng(devices=devices):
            model.eval()
            yield
    finally:
        random.setstate(py)
        np.random.set_state(npstate)
        for module, training in modes:
            module.training = training


def _all_finite(tensor):
    # aminmax propagates NaN and detects either infinity without constructing a
    # full-size temporary mask. This is important for CPU-staged Qwen guards.
    if tensor.numel() == 0:
        return True
    low, high = torch.aminmax(tensor)
    return math.isfinite(low.item()) and math.isfinite(high.item())


def pair_weights(known_en, known_sw, mode='independent', permutation=None):
    """Joint-mastery weighting; shuffling partner identities changes F, not E.

    Own mastery is never changed by the permutation. A joint-mastered partner
    doubles protection weight before normalization *within each language*.
    No wrong response is treated as a target. Degenerate strata may equal E.
    """
    en = torch.as_tensor(known_en, dtype=torch.float64)
    sw = torch.as_tensor(known_sw, dtype=torch.float64)
    if en.ndim != 1 or en.shape != sw.shape or not bool(((en == 0) | (en == 1)).all()) or not bool(((sw == 0) | (sw == 1)).all()):
        raise ValueError('Binary aligned mastery vectors required')
    if mode == 'independent':
        return en, sw
    if mode != 'paired':
        raise ValueError(mode)
    perm = torch.arange(len(en)) if permutation is None else torch.as_tensor(permutation)
    if perm.dtype != torch.long or not torch.equal(perm.sort().values, torch.arange(len(en))):
        raise ValueError('Not a partner bijection')
    inverse = torch.argsort(perm)
    return en * (1 + sw[perm]), sw * (1 + en[inverse])


def solve_dual(gram, dots, epsilon, tol=1e-9):
    """Exact active-set enumeration for <=8 halfspaces, including dependent rows.

    Return nonnegative lambda for d=z-A.T@lambda. Normals are internally
    normalized for scale invariance. epsilon>=0 ensures zero is feasible.
    """
    q, b, eps = [torch.as_tensor(x, dtype=torch.float64, device='cpu') for x in (gram, dots, epsilon)]
    k = b.numel()
    if b.ndim != 1 or k > 8 or q.shape != (k, k) or eps.shape != b.shape:
        raise ValueError('Invalid small QP dimensions')
    if not all(bool(torch.isfinite(x).all()) for x in (q, b, eps)) or bool((eps < 0).any()):
        raise ValueError('Finite QP and nonnegative tolerances required')
    if not torch.allclose(q, q.T, atol=1e-12, rtol=1e-8) or (k and torch.linalg.eigvalsh(q).min() < -1e-8 * max(1., q.abs().max().item())):
        raise ValueError('Gram is not symmetric positive semidefinite')
    scales = q.diag().clamp_min(0).sqrt()
    zero = scales == 0
    if bool((b[zero].abs() > 1e-12).any()):
        raise ValueError('Zero normal with nonzero dot')
    scales[zero] = 1
    Q = q / scales[:, None] / scales[None, :]
    c = (b - eps) / scales
    if bool((c <= 0).all()):
        return torch.zeros(k, dtype=torch.float64)
    for size in range(1, k + 1):
        for active in itertools.combinations(range(k), size):
            idx = list(active)
            block = Q[idx][:, idx]
            val = torch.linalg.lstsq(block, c[idx], rcond=1e-12, driver='gelsd').solution
            lam = torch.zeros(k, dtype=torch.float64)
            lam[idx] = val
            slack = Q @ lam - c
            if bool((val >= -tol).all()) and bool((slack >= -tol).all()) and float((lam * slack).abs().max()) <= tol:
                return lam.clamp_min(0) / scales
    raise ArithmeticError('No numerically valid joint projection; do not write weights')


def project_vector(z, normals, epsilon=None):
    """Small/reference API; normal rows already restricted to mask coordinates."""
    z = z.double()
    a = normals.double()
    eps = torch.zeros(a.shape[0], dtype=torch.float64) if epsilon is None else torch.as_tensor(epsilon).double()
    lam = solve_dual(a @ a.T, a @ z, eps)
    return z - a.T @ lam, lam


class SparseProjectedAdamW:
    """FP32 moments on parameter device; constraints/proposals staged on CPU.

    Full gradients for candidate matrices; mask rank = normalized |first moment|.
    Optional original-English sensitivity affects mask ranking only. Update
    projection is global across all selected matrices, not one QP per matrix.
    BF16 policy: if rounded displacement violates a guard, reject whole weight
    write, retain advanced moments, and report rejection. No silent precision
    change, no retries with adjusted epsilon. Checkpoints contain masks/unions.
    """
    def __init__(self, named, fraction=.15, mask_interval=20, wd=.01, bf16_policy='reject'):
        self.named = dict(named)
        if not self.named or not 0 < fraction <= 1 or mask_interval < 1 or wd < 0 or bf16_policy not in ('reject', 'record'):
            raise ValueError('Invalid optimizer configuration')
        if any(not p.is_contiguous() for p in self.named.values()):
            raise ValueError('Contiguous candidate tensors required')
        self.fraction, self.mask_interval, self.wd = fraction, mask_interval, wd
        self.bf16_policy = bf16_policy
        self.steps, self.state, self.masks, self.unions = 0, {}, {}, {}
        self.indices, self.cpu_indices = {}, {}
        self.selected_count=sum(math.floor(p.numel()*fraction) for p in self.named.values())
        self.union_counts={}

    @torch.no_grad()
    def step(self, lr, guards=(), epsilon=None, sensitivity=None):
        if not math.isfinite(lr) or lr < 0:
            raise ValueError('Invalid learning rate')
        names = set(self.named)
        for n, p in self.named.items():
            if p.grad is None or p.grad.shape != p.shape or not _all_finite(p.grad):
                raise FloatingPointError('Invalid target gradient: ' + n)
        for guard in guards:
            if set(guard) != names:
                raise ValueError('Constraint identity mismatch')
            for n, p in self.named.items():
                if guard[n].shape != p.shape or not _all_finite(guard[n]):
                    raise FloatingPointError('Invalid guard: ' + n)
        k = len(guards)
        eps = torch.zeros(k, dtype=torch.float64) if epsilon is None else torch.as_tensor(epsilon, dtype=torch.float64)
        if k > 8 or eps.shape != (k,) or not bool(torch.isfinite(eps).all()) or bool((eps < 0).any()):
            raise ValueError('Invalid guard budget')
        if sensitivity is not None:
            if set(sensitivity) != names:
                raise ValueError('Sensitivity identity mismatch')
            for n, p in self.named.items():
                if sensitivity[n].shape != p.shape or not _all_finite(sensitivity[n]) or bool((sensitivity[n] < 0).any()):
                    raise ValueError('Invalid sensitivity')
        t = self.steps + 1
        proposed, restricted = {}, {}
        gram, dots = torch.zeros((k, k), dtype=torch.float64), torch.zeros(k, dtype=torch.float64)
        z2 = 0.
        for n, p in self.named.items():
            if n not in self.state:
                self.state[n] = dict(m=torch.zeros_like(p, dtype=torch.float32), v=torch.zeros_like(p, dtype=torch.float32))
            m, v = self.state[n]['m'], self.state[n]['v']
            g = p.grad.float()
            m.mul_(.9).add_(g, alpha=.1)
            v.mul_(.999).addcmul_(g, g, value=.001)
            if self.steps % self.mask_interval == 0:
                score = m.abs()
                if sensitivity is not None:
                    s = sensitivity[n].to(p.device, dtype=torch.float32)
                    score = score / (score.mean() + 1e-12) / (1 + s / (s.mean() + 1e-12))
                count = math.floor(p.numel() * self.fraction)
                order = torch.argsort(score.flatten(), descending=True, stable=True)
                mask = torch.zeros(p.numel(), dtype=torch.bool, device=p.device)
                mask[order[:count]] = True
                self.masks[n] = mask.view_as(p).cpu()
                self.cpu_indices[n] = self.masks[n].flatten().nonzero().flatten()
                self.indices[n] = self.cpu_indices[n].to(p.device)
                self.unions[n] = self.unions.get(n, torch.zeros_like(p, dtype=torch.bool, device='cpu')) | self.masks[n]
                self.union_counts[n]=int(torch.count_nonzero(self.unions[n]))
            # Cached ascending indices preserve boolean-mask coordinate order,
            # avoiding repeated CUDA nonzero/synchronization and mask transfers.
            idx, cpu_idx = self.indices[n], self.cpu_indices[n]
            z = (-lr * (m.flatten().index_select(0, idx) / (1 - .9**t)) / ((v.flatten().index_select(0, idx) / (1 - .999**t)).sqrt() + 1e-8) - lr * self.wd * p.flatten().index_select(0, idx).float()).cpu()
            proposed[n] = z.double()
            aa = torch.stack([h[n].flatten().index_select(0, cpu_idx if h[n].device.type == 'cpu' else idx).float().cpu() for h in guards]) if k else torch.empty((0, z.numel()))
            restricted[n] = aa.double()
            zz, a = proposed[n], restricted[n]
            gram += a @ a.T
            dots += a @ zz
            z2 += zz.square().sum().item()
        lam = solve_dual(gram, dots, eps)
        staged, actual_dots, intended_dots = {}, torch.zeros(k, dtype=torch.float64), torch.zeros(k, dtype=torch.float64)
        d2 = correction2 = actual2 = nonzero = 0
        cosine_dot = 0.
        for n, p in self.named.items():
            a, z, idx = restricted[n], proposed[n], self.indices[n]
            d = (z.double() - a.double().T @ lam).float()
            old = p.flatten().index_select(0, idx).cpu()
            new = (old.float() + d).to(p.dtype)
            actual = new.float() - old.float()
            staged[n] = new
            actual_dots += a.double() @ actual.double()
            intended_dots += a.double() @ d.double()
            d2 += d.double().square().sum().item()
            correction2 += (d.double() - z.double()).square().sum().item()
            actual2 += actual.double().square().sum().item()
            cosine_dot += (z.double() * d.double()).sum().item()
            nonzero += int((new != old).sum())
        rejected = self.bf16_policy == 'reject' and bool((actual_dots > eps + 1e-12).any())
        if not rejected:
            for n, p in self.named.items():
                p.view(-1).index_copy_(0, self.indices[n], staged[n].to(p.device))
        self.steps = t
        return dict(step=t, lr=lr, selected=self.selected_count,
                    union=sum(self.union_counts.values()),
                    nonzero_write=0 if rejected else nonzero, candidate_norm=math.sqrt(z2),
                    projected_norm=math.sqrt(d2), correction_norm=math.sqrt(correction2),
                    cosine=cosine_dot / max(math.sqrt(z2*d2), 1e-300),
                    intended_residual=(intended_dots-eps).tolist(),
                    rounded_proposal_residual=(actual_dots-eps).tolist(),
                    actual_residual=(-eps if rejected else actual_dots-eps).tolist(),
                    actual_norm=0. if rejected else math.sqrt(actual2),
                    multipliers=lam.tolist(), active=int((lam>0).sum()), rejected_rounding=rejected)

    def _metadata(self):
        return dict(names=list(self.named), shapes={n:tuple(p.shape) for n,p in self.named.items()},
                    fraction=self.fraction, mask_interval=self.mask_interval, wd=self.wd,
                    bf16_policy=self.bf16_policy)

    def state_dict(self):
        return dict(**self._metadata(), steps=self.steps,
                    state={n:{k:v.cpu() for k,v in s.items()} for n,s in self.state.items()},
                    masks={n:v.cpu() for n,v in self.masks.items()}, unions=self.unions)

    def load_state_dict(self, data):
        metadata=self._metadata()
        for key in ('names','shapes','fraction','mask_interval','wd','bf16_policy'):
            expected = metadata[key]
            if data[key] != expected:
                raise ValueError('Resume identity mismatch: '+key)
        if data['steps'] and (set(data['state']) != set(self.named) or set(data['masks']) != set(self.named) or set(data['unions']) != set(self.named)):
            raise ValueError('Incomplete optimizer checkpoint')
        for n,s in data['state'].items():
            if set(s) != {'m','v'} or any(v.dtype != torch.float32 or v.shape != self.named[n].shape or not _all_finite(v) for v in s.values()):
                raise ValueError('Invalid FP32 moment state')
        for key in ('masks','unions'):
            for n,v in data[key].items():
                if v.dtype != torch.bool or v.shape != self.named[n].shape:
                    raise ValueError('Invalid support state')
        if not isinstance(data['steps'], int) or data['steps'] < 0:
            raise ValueError('Invalid optimizer step')
        if data['steps'] == 0:
            if data['state'] or data['masks'] or data['unions']:
                raise ValueError('Nonempty state at step zero')
            self.steps=0;self.state.clear();self.masks.clear();self.unions.clear();self.indices.clear();self.cpu_indices.clear();self.union_counts.clear()
            return
        self.steps = data['steps']
        # Reuse validated storage. Building a second full GPU moment dictionary
        # before releasing the old one exceeds 24GB on real Qwen candidates.
        for n,s in data['state'].items():
            if n not in self.state:self.state[n]={}
            for k,v in s.items():
                if k in self.state[n]:self.state[n][k].copy_(v)
                else:self.state[n][k]=v.to(device=self.named[n].device, copy=True)
        for n,v in data['masks'].items():
            if n in self.masks:self.masks[n].copy_(v)
            else:self.masks[n]=v.cpu().clone()
            cpu_idx=self.masks[n].flatten().nonzero().flatten()
            if n not in self.cpu_indices or not torch.equal(cpu_idx,self.cpu_indices[n]):
                self.cpu_indices[n]=cpu_idx
                self.indices[n]=cpu_idx.to(self.named[n].device)
        self.unions = {n:v.cpu().clone() for n,v in data['unions'].items()}
        self.union_counts={n:int(torch.count_nonzero(v)) for n,v in self.unions.items()}
