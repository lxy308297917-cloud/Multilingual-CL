"""Exact PLND scores, selection, validation, and sparse-gradient masks."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch

PLND_METHODS = {
    "plnd_paper",
    "plnd_paper_random",
    "plnd_edge_coupled",
    "plnd_edge_random",
    "plnd_edge_protect",
    "plnd_edge_protect_random",
}
GROUP_CAPS = {"q": 38, "k": 6, "v": 6, "ffn": 63}


def ffn_removal_scores(z: torch.Tensor, down_weight: torch.Tensor) -> torch.Tensor:
    """Exact Frobenius output change when each intermediate channel is removed."""
    if z.shape[-1] != down_weight.shape[1]:
        raise ValueError(f"FFN shape mismatch: z={tuple(z.shape)}, down={tuple(down_weight.shape)}")
    axes = tuple(range(z.ndim - 1))
    return torch.linalg.vector_norm(z.float(), dim=axes) * torch.linalg.vector_norm(
        down_weight.float(), dim=0
    )


def _softmax(logits: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is not None:
        logits = logits + mask.to(device=logits.device, dtype=logits.dtype)
    return torch.softmax(logits.float(), dim=-1)


def qk_probability_removal_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    channel_chunk_size: int = 32,
    compute_q: bool = True,
    compute_k: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact native-Q/native-K effects on GQA attention probabilities."""
    if q.ndim != 4 or k.ndim != 4 or q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1]:
        raise ValueError(f"Invalid Q/K shapes: q={tuple(q.shape)}, k={tuple(k.shape)}")
    if q.shape[1] % k.shape[1]:
        raise ValueError("Q head count must be divisible by KV head count")
    q_heads, kv_heads, head_dim = q.shape[1], k.shape[1], q.shape[-1]
    groups = q_heads // kv_heads
    k_repeated = k.repeat_interleave(groups, dim=1)
    logits = torch.matmul(q.float(), k_repeated.float().transpose(-1, -2)) / math.sqrt(head_dim)
    base = _softmax(logits, attention_mask)
    q_scores = torch.zeros(q_heads, head_dim, device=q.device)
    k_scores = torch.zeros(kv_heads, head_dim, device=q.device)
    chunk = max(1, int(channel_chunk_size))
    if compute_q:
        for head in range(q_heads):
            kv_head = head // groups
            for start in range(0, head_dim, chunk):
                end = min(head_dim, start + chunk)
                delta = torch.einsum(
                    "bqd,bkd->dbqk", q[:, head, :, start:end].float(), k[:, kv_head, :, start:end].float()
                ) / math.sqrt(head_dim)
                mask = attention_mask
                if mask is not None and mask.shape[1] > 1:
                    mask = mask[:, head : head + 1]
                changed = _softmax(logits[:, head].unsqueeze(0) - delta, mask)
                q_scores[head, start:end] = torch.linalg.vector_norm(
                    changed - base[:, head].unsqueeze(0), dim=(1, 2, 3)
                )
    if compute_k:
        for kv_head in range(kv_heads):
            heads = slice(kv_head * groups, (kv_head + 1) * groups)
            for start in range(0, head_dim, chunk):
                end = min(head_dim, start + chunk)
                delta = torch.einsum(
                    "bhqd,bkd->dbhqk", q[:, heads, :, start:end].float(), k[:, kv_head, :, start:end].float()
                ) / math.sqrt(head_dim)
                mask = attention_mask
                if mask is not None and mask.shape[1] > 1:
                    mask = mask[:, heads]
                changed = _softmax(logits[:, heads].unsqueeze(0) - delta, mask)
                k_scores[kv_head, start:end] = torch.linalg.vector_norm(
                    changed - base[:, heads].unsqueeze(0), dim=(1, 2, 3, 4)
                )
    return q_scores.flatten(), k_scores.flatten(), base


def v_output_removal_scores(attention_probability: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Exact pre-o_proj output change for each native GQA V channel."""
    q_heads, kv_heads, head_dim = attention_probability.shape[1], v.shape[1], v.shape[-1]
    if q_heads % kv_heads:
        raise ValueError("Q head count must be divisible by KV head count")
    groups = q_heads // kv_heads
    scores = torch.zeros(kv_heads, head_dim, device=v.device)
    for kv_head in range(kv_heads):
        probability = attention_probability[:, kv_head * groups : (kv_head + 1) * groups].float()
        contribution = torch.einsum("bhqk,bkd->bhqd", probability, v[:, kv_head].float())
        scores[kv_head] = torch.linalg.vector_norm(contribution, dim=(0, 1, 2))
    return scores.flatten()


def strict_intersection_rank(
    document_scores: Sequence[torch.Tensor], candidate_ratio: float, cap: int
) -> list[int]:
    """Intersect per-document top candidates and retain mean normalized-impact order."""
    if not document_scores:
        return []
    width = int(document_scores[0].numel())
    if any(int(score.numel()) != width for score in document_scores):
        raise ValueError("All document scores must have equal width")
    keep = max(1, min(width, int(math.ceil(width * candidate_ratio))))
    intersection = torch.ones(width, dtype=torch.bool)
    normalized_sum = torch.zeros(width, dtype=torch.float64)
    for raw in document_scores:
        score = raw.detach().float().cpu()
        if not torch.isfinite(score).all():
            raise ValueError("Non-finite PLND score")
        candidate = torch.zeros(width, dtype=torch.bool)
        candidate[torch.topk(score, keep, sorted=False).indices] = True
        intersection &= candidate
        maximum = float(score.max().item())
        normalized_sum += score.double() / maximum if maximum > 0 else score.double()
    ids = intersection.nonzero(as_tuple=False).flatten()
    if not ids.numel():
        return []
    order = torch.argsort(normalized_sum[ids], descending=True, stable=True)
    return ids[order[:cap]].tolist()


def matched_random_ids(width: int, count: int, forbidden: Iterable[int], seed: int) -> list[int]:
    forbidden = {int(index) for index in forbidden}
    pool = torch.tensor([index for index in range(width) if index not in forbidden])
    if count > pool.numel():
        raise ValueError(f"Cannot draw {count} controls from complement of size {pool.numel()}")
    generator = torch.Generator().manual_seed(int(seed))
    return pool[torch.randperm(pool.numel(), generator=generator)[:count]].tolist()


def canonical_json_hash(value: Mapping) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(raw).hexdigest()


def load_and_validate_mask(path, model, method: str) -> dict:
    if method not in PLND_METHODS:
        raise ValueError(f"Unknown PLND method: {method}")
    mask_path = Path(path).resolve()
    if not mask_path.is_file():
        raise FileNotFoundError(mask_path)
    artifact = torch.load(mask_path, map_location="cpu", weights_only=False)
    if artifact.get("schema_version") != 1:
        raise ValueError("Unsupported PLND mask schema")
    unsigned = {key: value for key, value in artifact.items() if key != "artifact_sha256"}
    if artifact.get("artifact_sha256") != canonical_json_hash(unsigned):
        raise ValueError("PLND mask artifact hash mismatch")
    source = "random" if method.endswith("_random") else "selected"
    selected = artifact.get(source, {})
    if not selected or sum(len(ids) for groups in selected.values() for ids in groups.values()) == 0:
        raise ValueError(f"PLND {source} mask is empty")
    for layer_text, groups in selected.items():
        layer_id = int(layer_text)
        if not 0 <= layer_id < len(model.model.layers):
            raise IndexError(f"Layer {layer_id} out of range")
        layer = model.model.layers[layer_id]
        dimensions = {
            "q": layer.self_attn.q_proj.weight.shape[0],
            "k": layer.self_attn.k_proj.weight.shape[0],
            "v": layer.self_attn.v_proj.weight.shape[0],
            "ffn": layer.mlp.up_proj.weight.shape[0],
        }
        for group, ids in groups.items():
            if group not in dimensions or len(ids) != len(set(ids)):
                raise ValueError(f"Invalid group or duplicate IDs: layer={layer_id} group={group}")
            if any(not 0 <= int(index) < dimensions[group] for index in ids):
                raise IndexError(f"Out-of-range ID: layer={layer_id} group={group}")
            if source == "random":
                wanted = len(artifact["selected"].get(str(layer_id), {}).get(group, []))
                if len(ids) != wanted:
                    raise ValueError(f"Random count mismatch: layer={layer_id} group={group}")
    return artifact


def _row_hook(weight: torch.nn.Parameter, ids: Sequence[int]):
    mask = torch.zeros((weight.shape[0], 1), dtype=weight.dtype, device=weight.device)
    mask[list(ids), 0] = 1
    weight.requires_grad = True
    return weight.register_hook(lambda gradient, m=mask: gradient * m.to(gradient))


def _column_hook(weight: torch.nn.Parameter, ids: Sequence[int]):
    mask = torch.zeros((1, weight.shape[1]), dtype=weight.dtype, device=weight.device)
    mask[0, list(ids)] = 1
    weight.requires_grad = True
    return weight.register_hook(lambda gradient, m=mask: gradient * m.to(gradient))


def _inverse_row_hook(weight: torch.nn.Parameter, ids: Sequence[int]):
    mask = torch.ones((weight.shape[0], 1), dtype=weight.dtype, device=weight.device)
    mask[list(ids), 0] = 0
    weight.requires_grad = True
    return weight.register_hook(lambda gradient, m=mask: gradient * m.to(gradient))


def _inverse_column_hook(weight: torch.nn.Parameter, ids: Sequence[int]):
    mask = torch.ones((1, weight.shape[1]), dtype=weight.dtype, device=weight.device)
    mask[0, list(ids)] = 0
    weight.requires_grad = True
    return weight.register_hook(lambda gradient, m=mask: gradient * m.to(gradient))


def apply_plnd_mask_from_env(model) -> bool:
    method = os.environ.get("PLND_METHOD", "").strip()
    if not method:
        return False
    artifact = load_and_validate_mask(os.environ.get("PLND_MASK_PATH", ""), model, method)
    source = "random" if method.endswith("_random") else "selected"
    edge = method.startswith("plnd_edge_")
    protect = method.startswith("plnd_edge_protect")
    for parameter in model.parameters():
        parameter.requires_grad = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    handles, effective = [], 0
    edge_layers = set(artifact.get("edge_layers", [1, 2, 3, 4, 5, 23, 24, 25, 26, 27]))
    if protect:
        for layer_id in edge_layers:
            for parameter in model.model.layers[layer_id].parameters():
                parameter.requires_grad = True
    for layer_text, groups in artifact[source].items():
        if edge and int(layer_text) not in edge_layers:
            continue
        layer = model.model.layers[int(layer_text)]
        if not edge:
            for group, projection in (
                ("q", layer.self_attn.q_proj), ("k", layer.self_attn.k_proj), ("v", layer.self_attn.v_proj)
            ):
                ids = groups.get(group, [])
                if ids:
                    handles.append(_row_hook(projection.weight, ids))
                    effective += len(ids) * projection.weight.shape[1]
        ids = groups.get("ffn", [])
        if ids:
            if protect:
                handles += [
                    _inverse_row_hook(layer.mlp.gate_proj.weight, ids),
                    _inverse_row_hook(layer.mlp.up_proj.weight, ids),
                    _inverse_column_hook(layer.mlp.down_proj.weight, ids),
                ]
            else:
                handles += [_row_hook(layer.mlp.up_proj.weight, ids), _column_hook(layer.mlp.down_proj.weight, ids)]
                effective += len(ids) * (layer.mlp.up_proj.weight.shape[1] + layer.mlp.down_proj.weight.shape[0])
                if edge:
                    handles.append(_row_hook(layer.mlp.gate_proj.weight, ids))
                    effective += len(ids) * layer.mlp.gate_proj.weight.shape[1]
    model._plnd_hook_handles = handles
    model._plnd_artifact = artifact
    total = sum(parameter.numel() for parameter in model.parameters())
    visible = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if protect:
        protected = sum(
            len(artifact[source][str(layer_id)].get("ffn", []))
            * (
                model.model.layers[layer_id].mlp.gate_proj.weight.shape[1]
                + model.model.layers[layer_id].mlp.up_proj.weight.shape[1]
                + model.model.layers[layer_id].mlp.down_proj.weight.shape[0]
            )
            for layer_id in edge_layers
        )
        effective = visible - protected
    print("=" * 88)
    print(f"[PLND] method={method} mask={Path(os.environ['PLND_MASK_PATH']).resolve()}")
    print(f"[PLND] optimizer_visible={visible:,} effective={effective:,} ratio={effective / total:.8%}")
    print("[PLND] weight_decay must be 0 for exact masked-entry protection")
    print("=" * 88)
    return True
