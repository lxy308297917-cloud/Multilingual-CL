"""Apply an auditable Base->adapted hybrid swap when a HF model is loaded.

Activation is opt-in via HYBRID_SWAP_MANIFEST and HYBRID_SWAP_CONDITION.
Ordinary Python processes are completely unaffected.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


MANIFEST_PATH = os.environ.get("HYBRID_SWAP_MANIFEST")
CONDITION = os.environ.get("HYBRID_SWAP_CONDITION")
REPORT_PATH = os.environ.get("HYBRID_SWAP_REPORT")


if bool(MANIFEST_PATH) != bool(CONDITION):
    raise RuntimeError("HYBRID_SWAP_MANIFEST and HYBRID_SWAP_CONDITION must be set together")


if MANIFEST_PATH and CONDITION:
    import torch
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM

    manifest = json.loads(Path(MANIFEST_PATH).read_text(encoding="utf-8"))
    if manifest.get("format") != "qwen_hybrid_swap_v1":
        raise RuntimeError(f"Unsupported hybrid manifest: {MANIFEST_PATH}")
    if CONDITION not in manifest["conditions"]:
        raise KeyError(f"Unknown hybrid condition {CONDITION!r}")

    class TensorStore:
        def __init__(self, model_dir):
            model_dir = Path(model_dir)
            index_path = model_dir / "model.safetensors.index.json"
            if index_path.exists():
                weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
                self.key_to_file = {key: model_dir / name for key, name in weight_map.items()}
            else:
                self.key_to_file = {}
                for path in sorted(model_dir.glob("*.safetensors")):
                    for key in safe_open(path, framework="pt", device="cpu").keys():
                        self.key_to_file[key] = path
            self.handles = {}

        def get(self, key):
            path = self.key_to_file[key]
            if path not in self.handles:
                self.handles[path] = safe_open(path, framework="pt", device="cpu")
            return self.handles[path].get_tensor(key)

    def apply_hybrid_swap(model):
        expected_adapted = str(Path(manifest["adapted_model"]).resolve())
        loaded = getattr(model.config, "_name_or_path", "")
        if loaded and Path(loaded).exists() and str(Path(loaded).resolve()) != expected_adapted:
            loaded_weight = Path(loaded) / "model.safetensors"
            expected_weight = Path(expected_adapted) / "model.safetensors"
            is_view = loaded_weight.exists() and loaded_weight.resolve() == expected_weight.resolve()
            if not is_view:
                raise RuntimeError(f"Hybrid manifest expects {expected_adapted} or a symlink view, loaded {Path(loaded).resolve()}")
        store = TensorStore(manifest["base_model"])
        parameters = dict(model.named_parameters())
        buffers = dict(model.named_buffers())
        targets = {**parameters, **buffers}
        condition = manifest["conditions"][CONDITION]
        keys = condition["tensor_keys"]
        missing = [key for key in keys if key not in targets]
        if missing:
            raise KeyError(f"Hybrid target keys missing from loaded model: {missing[:20]}")

        replaced = []
        with torch.no_grad():
            for index, key in enumerate(keys, 1):
                target = targets[key]
                source_cpu = store.get(key)
                if tuple(target.shape) != tuple(source_cpu.shape):
                    raise RuntimeError(f"Shape mismatch for {key}: {target.shape} vs {source_cpu.shape}")
                source = source_cpu.to(device=target.device, dtype=target.dtype)
                before_delta = torch.max(torch.abs(target.detach() - source)).item()
                target.copy_(source)
                after_delta = torch.max(torch.abs(target.detach() - source)).item()
                if after_delta != 0.0:
                    raise RuntimeError(f"Post-swap verification failed for {key}: {after_delta}")
                replaced.append({"key": key, "numel": target.numel(),
                                 "max_abs_delta_before": before_delta,
                                 "max_abs_delta_after": after_delta})
                del source
                if index % 50 == 0 or index == len(keys):
                    print(f"[hybrid-swap:{CONDITION}] verified {index}/{len(keys)} tensors", flush=True)

        actual_numel = sum(row["numel"] for row in replaced)
        if actual_numel != condition["numel"]:
            raise RuntimeError(f"Manifest numel mismatch: expected {condition['numel']}, got {actual_numel}")
        report = {
            "format": "qwen_hybrid_swap_report_v1",
            "condition": CONDITION,
            "manifest": str(Path(MANIFEST_PATH).resolve()),
            "base_model": manifest["base_model"],
            "adapted_model": manifest["adapted_model"],
            "tensor_count": len(replaced),
            "numel": actual_numel,
            "all_after_deltas_zero": all(row["max_abs_delta_after"] == 0 for row in replaced),
            "changed_tensor_count": sum(row["max_abs_delta_before"] > 0 for row in replaced),
            "tensors": replaced,
        }
        if REPORT_PATH:
            report_path = Path(REPORT_PATH)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        model._hybrid_swap_report = report
        return model

    original_from_pretrained = AutoModelForCausalLM.from_pretrained.__func__

    def hybrid_from_pretrained(cls, *args, **kwargs):
        return apply_hybrid_swap(original_from_pretrained(cls, *args, **kwargs))

    AutoModelForCausalLM.from_pretrained = classmethod(hybrid_from_pretrained)
