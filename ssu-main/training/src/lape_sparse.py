"""Deterministic LAPE selection and absolute FFN-slice training/checkpoints."""
from __future__ import annotations
import hashlib
import json
import os
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def atomic_torch(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def select_lape(counts, tokens, target, ratios, quantile, random_seed):
    """counts: [language, layer, neuron]; stable ties follow layer/neuron ID."""
    counts = torch.as_tensor(counts, dtype=torch.float64)
    tokens = torch.as_tensor(tokens, dtype=torch.float64)
    if counts.ndim != 3 or len(tokens) != len(counts) or (tokens <= 0).any():
        raise ValueError('Invalid activation shape or token counts')
    if not torch.isfinite(counts).all() or (counts < 0).any() or (counts > tokens[:, None, None]).any():
        raise ValueError('Invalid activation counts')
    p = counts / tokens[:, None, None]
    q = p / p.sum(0).clamp_min(1e-300)
    entropy = -(q * q.clamp_min(1e-300).log()).sum(0)
    threshold = torch.quantile(p.flatten(), quantile)
    eligible = (p.max(0).values >= threshold) & (p.sum(0) > 0)
    order = torch.argsort(entropy.flatten(), stable=True)
    order = order[eligible.flatten()[order]]
    layers, width = entropy.shape
    result = {}
    for name, ratio in ratios.items():
        if not 0 < ratio < 1:
            raise ValueError('LAPE fraction must be in (0,1)')
        selected = order[:min(int(layers * width * ratio), len(order))]
        selected = selected[p[target].flatten()[selected] >= threshold]
        ids = {str(l): sorted(int(i % width) for i in selected.tolist() if i // width == l) for l in range(layers)}
        if not any(ids.values()):
            raise ValueError(f'Empty target mask: {name}; do not pad inactive neurons')
        random_ids = {}
        for l in range(layers):
            chosen = set(ids[str(l)])
            candidates = [i for i in range(width) if i not in chosen]
            if len(candidates) < len(chosen):
                raise ValueError('Insufficient disjoint random neurons')
            rng = random.Random(random_seed + l)
            random_ids[str(l)] = sorted(rng.sample(candidates, len(chosen)))
        result[name] = {'selected': ids, 'random': random_ids, 'candidate_fraction': ratio,
                        'selected_neurons': sum(map(len, ids.values()))}
    return result, {'activation_threshold': float(threshold), 'eligible_neurons': int(eligible.sum()),
                    'entropy_min': float(entropy.min()), 'entropy_max': float(entropy.max())}


class SliceMLP(nn.Module):
    """Frozen original plus absolute selected slices; scatter keeps input gradients."""
    def __init__(self, original, indices):
        super().__init__()
        if not indices or len(set(indices)) != len(indices):
            raise ValueError('Nonempty unique neuron IDs required')
        self.original = original
        for p in self.original.parameters():
            p.requires_grad_(False)
        ix = torch.tensor(indices, dtype=torch.long, device=original.gate_proj.weight.device)
        if ix.min() < 0 or ix.max() >= original.gate_proj.weight.shape[0]:
            raise ValueError('Neuron ID outside FFN')
        self.register_buffer('indices', ix)
        self.gate = nn.Parameter(original.gate_proj.weight.index_select(0, ix).detach().clone())
        self.up = nn.Parameter(original.up_proj.weight.index_select(0, ix).detach().clone())
        self.down = nn.Parameter(original.down_proj.weight.index_select(1, ix).detach().clone())

    def forward(self, x):
        # Use the same full GEMM shapes as the original forward to avoid changing its numerics.
        gate = self.original.gate_proj.weight.index_copy(0, self.indices, self.gate)
        up = self.original.up_proj.weight.index_copy(0, self.indices, self.up)
        down = self.original.down_proj.weight.index_copy(1, self.indices, self.down)
        z = self.original.act_fn(F.linear(x, gate, self.original.gate_proj.bias)) * F.linear(x, up, self.original.up_proj.bias)
        return F.linear(z, down, self.original.down_proj.bias)


def install_slices(model, ids):
    for p in model.parameters():
        p.requires_grad_(False)
    for layer, values in ids.items():
        if values:
            model.model.layers[int(layer)].mlp = SliceMLP(model.model.layers[int(layer)].mlp, values)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError('No trainable slices')
    return params


def slice_values(model):
    return {str(i): {k: getattr(layer.mlp, k).detach().cpu().clone() for k in ('gate', 'up', 'down')}
            for i, layer in enumerate(model.model.layers) if isinstance(layer.mlp, SliceMLP)}


def load_slices(model, values):
    expected = {str(i) for i, l in enumerate(model.model.layers) if isinstance(l.mlp, SliceMLP)}
    if set(values) != expected:
        raise ValueError('Checkpoint layers mismatch')
    with torch.no_grad():
        for i, tensors in values.items():
            for k in ('gate', 'up', 'down'):
                p = getattr(model.model.layers[int(i)].mlp, k)
                if p.shape != tensors[k].shape or p.dtype != tensors[k].dtype:
                    raise ValueError('Slice shape/dtype mismatch')
                p.copy_(tensors[k])


def merge_slices(model, values, ids):
    with torch.no_grad():
        for i, tensors in values.items():
            mlp = model.model.layers[int(i)].mlp
            for name, key, axis in [('gate_proj', 'gate', 0), ('up_proj', 'up', 0), ('down_proj', 'down', 1)]:
                weight = getattr(mlp, name).weight
                ix = torch.tensor(ids[i], device=weight.device)
                val = tensors[key].to(device=weight.device)
                if val.dtype != weight.dtype:
                    raise ValueError('No lossy checkpoint casts allowed')
                weight.index_copy_(axis, ix, val)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def checkpoint(model, optimizer, scheduler, step, identity):
    return {'format': 'lape_absolute_slices_v1', 'identity': identity, 'step': step,
            'slices': slice_values(model), 'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(), 'rng': rng_state()}


def resume(model, optimizer, scheduler, payload, identity):
    if payload['format'] != 'lape_absolute_slices_v1' or payload['identity'] != identity:
        raise ValueError('Checkpoint provenance mismatch')
    load_slices(model, payload['slices'])
    optimizer.load_state_dict(payload['optimizer']); scheduler.load_state_dict(payload['scheduler'])
    restore_rng(payload['rng'])
    return int(payload['step'])
