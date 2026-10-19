"""Check PEFT merge/unmerge on actual saved BF16 base and trained LoRA weights.

CPU only. This diagnoses a weight-sync invariant, not GPU throughput or scores.
No source model/checkpoint is changed. The output must be a new local file.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from peft.tuners.lora.layer import Linear
from safetensors import safe_open
from lora_sync_weights import merged_lora_weight


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not torch.cuda.is_initialized(), "CPU diagnostic must not initialize CUDA"
    assert not args.output.exists()
    torch.set_num_threads(4)
    index = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
    config = json.loads((args.adapter / "adapter_config.json").read_text())
    rows = []
    with safe_open(args.adapter / "adapter_model.safetensors", framework="pt", device="cpu") as adapter:
        for module in ["model.layers.0.self_attn.q_proj", "model.layers.0.mlp.down_proj"]:
            key = module + ".weight"
            with safe_open(args.model / index[key], framework="pt", device="cpu") as model:
                weight = model.get_tensor(key)
            a = adapter.get_tensor("base_model.model." + module + ".lora_A.weight")
            b = adapter.get_tensor("base_model.model." + module + ".lora_B.weight")
            base = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False, dtype=weight.dtype)
            with torch.no_grad():
                base.weight.copy_(weight)
            layer = Linear(base, "default", r=config["r"], lora_alpha=config["lora_alpha"], lora_dropout=0.0)
            # Actual trainable LoRA tensors are FP32; retain their saved dtype.
            layer.lora_A["default"].to(dtype=a.dtype)
            layer.lora_B["default"].to(dtype=b.dtype)
            with torch.no_grad():
                layer.lora_A["default"].weight.copy_(a)
                layer.lora_B["default"].weight.copy_(b)
            original = base.weight.detach().clone()
            exported = merged_lora_weight(layer)
            assert torch.equal(original, base.weight), "Non-mutating export changed the base"
            rounds = []
            for repeat in range(1, 4):
                layer.merge()
                if repeat == 1:
                    assert torch.equal(exported, base.weight), "Export differs from original PEFT merge"
                layer.unmerge()
                restored = base.weight.detach()
                rounds.append({"round": repeat, "changed_elements": int((original != restored).sum()),
                               "max_abs_change": float((original.float() - restored.float()).abs().max()),
                               "restored_sha256": digest(restored)})
            rows.append({"module": module, "elements": weight.numel(), "shape": list(weight.shape),
                         "base_dtype": str(weight.dtype), "adapter_dtype": str(a.dtype),
                         "nonmutating_export_matches_peft_merge_bitwise": True,
                         "nonmutating_export_preserves_base_bitwise": True,
                         "base_sha256": digest(original), "roundtrips": rounds})
    report = {"device": "cpu", "torch": torch.__version__,
              "scope": "Two actual checkpoint matrices only; no full-model distribution or GPU equivalence claim.",
              "base_lock_sha256": hashlib.sha256((args.model / "MODEL_LOCK.json").read_bytes()).hexdigest(),
              "adapter_sha256": hashlib.sha256((args.adapter / "adapter_model.safetensors").read_bytes()).hexdigest(),
              "matrices": rows,
              "bitwise_invariant_for_checked_matrices": all(r["changed_elements"] == 0 for row in rows for r in row["roundtrips"])}
    assert not torch.cuda.is_initialized()
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
