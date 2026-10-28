"""Reproducible SSU and HFT selection for a common 15% candidate budget."""
import hashlib
import json
import math
import random
from pathlib import Path

import torch


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _save_selection(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return file_sha(path)


def ssu_masks(model, candidates, calibration_rows, collate, fraction, output, column_quota=None):
    """Train the lowest Wanda/SSU columns; high English-importance columns stay frozen."""
    sums = {name: torch.zeros(param.shape[1], dtype=torch.float64) for name, param in candidates.items()}
    counts = {name: 0 for name in candidates}
    modules = dict(model.named_modules())
    hooks = []
    for name in candidates:
        module_name = name.rsplit(".weight", 1)[0]
        if module_name not in modules:
            raise KeyError(module_name)

        def hook(_module, inputs, _output, parameter_name=name):
            value = inputs[0].detach().float()
            flat = value.reshape(-1, value.shape[-1])
            sums[parameter_name].add_(flat.square().sum(0).cpu().double())
            counts[parameter_name] += flat.shape[0]

        hooks.append(modules[module_name].register_forward_hook(hook))
    model.eval()
    try:
        with torch.no_grad():
            for row in calibration_rows:
                batch = {k: v.to(next(model.parameters()).device) for k, v in collate([row]).items() if k != "labels"}
                model(**batch, use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()
    masks = {}
    selected_columns = {}
    per_parameter = []
    for name, param in candidates.items():
        if counts[name] <= 0:
            raise RuntimeError("missing activation statistics: " + name)
        activation = (sums[name] / counts[name]).to(torch.float32)
        score = param.detach().float().cpu().norm(dim=0) * activation
        keep_columns = math.floor(param.shape[1] * fraction) if column_quota is None else column_quota[name]
        assert 0 < keep_columns <= param.shape[1]
        order = torch.argsort(score, descending=False, stable=True)
        mask = torch.zeros_like(param, dtype=torch.bool, device="cpu")
        mask[:, order[:keep_columns]] = True
        masks[name] = mask
        selected_columns[name] = order[:keep_columns].cpu()
        per_parameter.append({"name": name, "columns": param.shape[1], "selected_columns": keep_columns,
                              "selected_elements": int(mask.sum()), "activation_positions": counts[name]})
    selected = sum(int(m.sum()) for m in masks.values())
    total = sum(m.numel() for m in masks.values())
    metadata = {"method": "ssu15", "score": "column_L2_weight_times_mean_squared_english_input_activation",
                "direction": "train lowest; freeze highest", "requested_fraction": fraction,
                "selected_elements": selected, "candidate_elements": total,
                "actual_fraction": selected / total, "calibration_ids": [x["id"] for x in calibration_rows],
                "parameters": per_parameter, "common_budget_column_allocation": column_quota is not None}
    metadata["selection_sha256"] = _save_selection(output, {"format": "columns_v1", "indices": selected_columns,
                                                              "metadata": metadata})
    return masks, metadata


def hft_masks(candidates, fraction, seed, output):
    """Choose whole Attention/MLP matrices with a seeded closest-subset budget."""
    names = list(candidates)
    random.Random(seed).shuffle(names)
    sizes = {n: candidates[n].numel() for n in names}
    unit = 0
    for size in sizes.values():
        unit = math.gcd(unit, size)
    target_elements = round(sum(sizes.values()) * fraction)
    target = round(target_elements / unit)
    scaled = {n: sizes[n] // unit for n in names}
    limit = target + max(scaled.values())
    reachable = {0: ()}
    for name in names:
        value = scaled[name]
        additions = {}
        for subtotal, chosen in list(reachable.items()):
            new = subtotal + value
            if new <= limit and new not in reachable and new not in additions:
                additions[new] = chosen + (name,)
        reachable.update(additions)
    best = min(reachable, key=lambda x: (abs(x - target), x > target, x))
    selected_names = set(reachable[best])
    masks = {n: n in selected_names for n in candidates}
    selected = sum(candidates[n].numel() for n in selected_names)
    total = sum(p.numel() for p in candidates.values())
    metadata = {"method": "hft15_matched", "selection": "seeded whole Attention/MLP matrices",
                "seed": seed, "requested_fraction": fraction, "target_elements": target_elements,
                "selected_elements": selected, "candidate_elements": total,
                "actual_fraction": selected / total, "absolute_budget_error": selected - target_elements,
                "selected_parameters": sorted(selected_names), "gcd_unit": unit}
    metadata["selection_sha256"] = _save_selection(output, {"format": "whole_parameters_v1",
                                                              "selected_names": sorted(selected_names),
                                                              "metadata": metadata})
    return masks, metadata


def load_masks(path, candidates):
    data = torch.load(path, map_location="cpu", weights_only=False)
    if data["format"] == "columns_v1":
        masks = {}
        for name, parameter in candidates.items():
            mask = torch.zeros_like(parameter, dtype=torch.bool, device="cpu")
            mask[:, data["indices"][name].long()] = True
            masks[name] = mask
    elif data["format"] == "whole_parameters_v1":
        selected = set(data["selected_names"])
        masks = {name: name in selected for name in candidates}
    else:
        raise ValueError("unknown selection format")
    return masks, data["metadata"]
