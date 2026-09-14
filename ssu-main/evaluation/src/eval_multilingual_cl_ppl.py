#!/usr/bin/env python3
"""Token-weighted PPL evaluation with auditable per-block NLL output."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from datasets import load_from_disk
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


def collate(examples):
    ids = torch.tensor([row["input_ids"] for row in examples], dtype=torch.long)
    mask = torch.tensor([row.get("attention_mask", [1] * len(row["input_ids"])) for row in examples], dtype=torch.long)
    labels = torch.tensor([row.get("labels", row["input_ids"]) for row in examples], dtype=torch.long)
    return ids, mask, labels


@torch.inference_mode()
def evaluate(model, dataset, batch_size: int) -> tuple[list[dict], float, int]:
    details, total_nll, total_tokens = [], 0.0, 0
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    offset = 0
    for ids, mask, labels in loader:
        ids, mask, labels = ids.to(model.device), mask.to(model.device), labels.to(model.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        shifted_logits = logits[:, :-1].float()
        shifted_labels = labels[:, 1:]
        losses = F.cross_entropy(
            shifted_logits.reshape(-1, shifted_logits.shape[-1]),
            shifted_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view(shifted_labels.shape)
        valid = (shifted_labels != -100) & mask[:, 1:].bool()
        per_sum = (losses * valid).sum(dim=1).cpu().numpy()
        per_count = valid.sum(dim=1).cpu().numpy()
        for local, (loss_sum, count) in enumerate(zip(per_sum, per_count)):
            count = int(count)
            mean = float(loss_sum / max(count, 1))
            details.append({"id": offset + local, "nll": mean, "tokens": count})
            total_nll += float(loss_sum)
            total_tokens += count
        offset += ids.shape[0]
        print(f"blocks={offset}/{len(dataset)} tokens={total_tokens}", flush=True)
    return details, total_nll / max(total_tokens, 1), total_tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-blocks", type=int)
    args = parser.parse_args()

    dataset = load_from_disk(str(args.dataset))
    if args.max_blocks:
        dataset = dataset.select(range(min(args.max_blocks, len(dataset))))
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True,
    ).eval()
    details, avg_nll, token_count = evaluate(model, dataset, args.batch_size)
    args.output.mkdir(parents=True, exist_ok=True)
    task_hash = hashlib.sha256(
        f"{dataset._fingerprint}|{len(dataset)}|causal_shift_v1".encode("utf-8")
    ).hexdigest()
    metrics = {
        "benchmark": "ppl", "metric": "ppl", "value": math.exp(min(50.0, avg_nll)),
        "sample_count": len(details), "token_count": token_count, "avg_nll": avg_nll,
        "task_hash": task_hash,
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        pq.write_table(pa.Table.from_pylist(details), args.output / "details_ppl.parquet")
    except ImportError:
        with (args.output / "details_ppl.jsonl").open("w", encoding="utf-8") as handle:
            for row in details:
                handle.write(json.dumps(row) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
