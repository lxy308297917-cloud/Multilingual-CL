"""Ordinary sparse SFT: direction-dependent selection, without update projection.

Calibration estimates live on CPU; only one matrix is materialized on the device.
Adam moments use the clipped training gradient exactly once per update. English
estimates affect the mask, never the moments (except replay's explicit CE loss).
"""
import math
import numpy as np
import torch


class DirectionSparseAdamW:
    def __init__(self, named, quota, method, mask_interval=20, wd=.01,
                 calibration_beta=.9):
        if method not in {'U', 'P', 'R', 'N', 'N_abs'}:
            raise ValueError(method)
        self.named = dict(named)
        self.quota = dict(quota)
        assert self.named.keys() == self.quota.keys()
        assert all(p.is_contiguous() and 0 < quota[n] <= p.numel()
                   for n, p in self.named.items())
        self.method, self.interval, self.wd = method, mask_interval, wd
        self.beta = calibration_beta
        self.steps = self.calibrations = self.calibration_cursor = 0
        self.state = {n: {k: torch.zeros_like(p, dtype=torch.float32)
                          for k in ('m', 'v')} for n, p in self.named.items()}
        self.english = {}
        self.masks, self.unions, self.indices = {}, {}, {}
        self.union_counts = {}

    def metadata(self):
        return dict(method=self.method, quota=self.quota,
                    shapes={n: tuple(p.shape) for n, p in self.named.items()},
                    mask_interval=self.interval, wd=self.wd,
                    calibration_beta=self.beta,
                    score_version='BF16-counterfactual-risk-v1')

    @torch.no_grad()
    def calibrate(self, gradients, next_cursor):
        assert self.method in {'P', 'N', 'N_abs'}
        assert gradients.keys() == self.named.keys()
        assert self.steps % self.interval == 0
        assert next_cursor > self.calibration_cursor
        for n, p in self.named.items():
            g = gradients[n].detach().to(device='cpu', dtype=torch.float32)
            assert g.shape == p.shape and bool(torch.isfinite(g).all())
            # P uses the square of the shared batch-mean CE gradient.
            value = g.square() if self.method == 'P' else g
            assert bool(torch.isfinite(value).all()), 'Nonfinite calibration statistic: '+n
            if n not in self.english:
                self.english[n] = value.clone()
            else:
                self.english[n].mul_(self.beta).add_(value, alpha=1-self.beta)
            assert bool(torch.isfinite(self.english[n]).all())
        self.calibrations += 1
        self.calibration_cursor = next_cursor

    @torch.no_grad()
    def step(self, lr):
        assert math.isfinite(lr) and lr >= 0
        refresh = self.steps % self.interval == 0
        if self.method in {'P', 'N', 'N_abs'}:
            assert self.calibrations == self.steps // self.interval + 1
            assert self.english.keys() == self.named.keys()
        t = self.steps + 1
        stats = dict(step=t, lr=lr, mask_refresh=refresh, risk_measured=refresh and self.method in {'N', 'N_abs'}, selected=0,
                     nonzero_write=0, mask_union=0, actual_delta_sq=0.,
                     candidate_delta_sq=0., predicted_risk_sum=0.,
                     selected_positive_risk=0, per_module={})
        for n, p in self.named.items():
            assert p.grad is not None
            m, v = self.state[n]['m'], self.state[n]['v']
            g = p.grad.float()
            assert bool(torch.isfinite(g).all())
            m.mul_(.9).add_(g, alpha=.1)
            v.mul_(.999).addcmul_(g, g, value=.001)
            delta = -lr*(m/(1-.9**t))/((v/(1-.999**t)).sqrt()+1e-8)-lr*self.wd*p.float()
            assert bool(torch.isfinite(delta).all())
            if refresh:
                target = m.abs()
                score = target/(target.mean()+1e-12)
                if self.method in {'P', 'N', 'N_abs'}:
                    e = self.english[n].to(p.device)
                    if self.method == 'P':
                        score = score/(1+e/(e.mean()+1e-12))
                    else:
                        q = (p.float()+delta).to(p.dtype).float()-p.float()
                        risk = e*q
                        assert bool(torch.isfinite(risk).all()), 'Nonfinite candidate risk: '+n
                        penalty = risk.clamp_min(0) if self.method == 'N' else risk.abs()
                        score = score/(1+penalty/(risk.abs().mean()+torch.finfo(torch.float32).tiny))
                        del q, risk, penalty
                    del e
                assert bool(torch.isfinite(score).all()), 'Nonfinite mask score: '+n
                order = torch.argsort(score.flatten(), descending=True, stable=True)
                idx = order[:self.quota[n]].sort().values
                mask = torch.zeros(p.numel(), dtype=torch.bool)
                mask[idx.cpu()] = True
                self.masks[n] = mask.view_as(p)
                self.unions[n] = self.unions.get(n, torch.zeros_like(self.masks[n])) | self.masks[n]
                self.indices[n] = idx
                self.union_counts[n] = int(torch.count_nonzero(self.unions[n]))
                del target, score, order
            idx = self.indices[n]
            old = p.flatten().index_select(0, idx)
            d = delta.flatten().index_select(0, idx)
            new = (old.float()+d).to(p.dtype)
            actual = new.float()-old.float()
            assert bool(torch.isfinite(new).all())
            a2, d2 = float(actual.square().sum()), float(d.square().sum())
            nz = int(torch.count_nonzero(actual))
            bucket = 'Attention' if '.self_attn.' in n else 'MLP'
            module = stats['per_module'].setdefault(bucket, dict(selected=0, nonzero_write=0,
                                                              actual_delta_sq=0., candidate_delta_sq=0.))
            for dst in (stats, module):
                dst['selected'] += self.quota[n]
                dst['nonzero_write'] += nz
                dst['actual_delta_sq'] += a2
                dst['candidate_delta_sq'] += d2
            stats['mask_union'] += self.union_counts[n]
            if refresh and self.method in {'N', 'N_abs'}:
                # Risk diagnostic is sampled only at mask refresh; stale EMA otherwise.
                # CPU indexed EMA avoids a full second matrix transfer per step.
                selected_e = self.english[n].flatten().index_select(0, idx.cpu()).to(p.device)
                r = selected_e*actual
                stats['predicted_risk_sum'] += float(r.sum())
                stats['selected_positive_risk'] += int(torch.count_nonzero(r > 0))
                del selected_e, r
            p.view(-1).index_copy_(0, idx, new)
            del g, delta, old, d, new, actual
        self.steps = t
        assert stats['selected'] == sum(self.quota.values())
        stats['actual_norm'] = math.sqrt(stats['actual_delta_sq'])
        stats['candidate_norm'] = math.sqrt(stats['candidate_delta_sq'])
        return stats

    def state_dict(self):
        def packed(values):
            return {n: torch.from_numpy(np.packbits(v.numpy().reshape(-1), bitorder='little'))
                    for n, v in values.items()}
        return dict(metadata=self.metadata(), steps=self.steps,
                    calibrations=self.calibrations, calibration_cursor=self.calibration_cursor,
                    state={n: {k: v.detach().cpu().clone() for k, v in s.items()}
                           for n, s in self.state.items()},
                    english={n: v.clone() for n, v in self.english.items()},
                    masks=packed(self.masks), unions=packed(self.unions))

    @torch.no_grad()
    def load_state_dict(self, data):
        assert data['metadata'] == self.metadata()
        assert data['state'].keys() == self.named.keys()
        self.steps, self.calibrations = data['steps'], data['calibrations']
        self.calibration_cursor = data['calibration_cursor']
        for n in self.named:
            for k in ('m', 'v'):
                value = data['state'][n][k]
                assert value.shape == self.named[n].shape and value.dtype == torch.float32
                self.state[n][k].copy_(value)
        self.english = {n: v.clone().cpu() for n, v in data['english'].items()}
        for key in ('masks', 'unions'):
            dest = getattr(self, key)
            dest.clear()
            for n, value in data[key].items():
                unpacked = np.unpackbits(value.numpy(), bitorder='little', count=self.named[n].numel()).copy()
                dest[n] = torch.from_numpy(unpacked).bool().view_as(self.named[n])
        self.indices = {n: mask.flatten().nonzero().flatten().to(self.named[n].device)
                        for n, mask in self.masks.items()}
        if self.steps:
            assert self.masks.keys() == self.named.keys()
            assert all(int(self.masks[n].sum()) == self.quota[n] for n in self.named)

        self.union_counts = {n: int(torch.count_nonzero(v)) for n, v in self.unions.items()}
