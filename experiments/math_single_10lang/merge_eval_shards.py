"""Merge disjoint per-item-seeded math evaluation shards in the original order."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    manifest = json.loads(args.manifest.read_text())
    source = Path(manifest["source"])
    if digest(source) != manifest["source_sha256"]:
        raise ValueError("Original evaluation items changed")
    items = read_jsonl(source)
    ids = [item["id"] for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate original item IDs")
    prefix = read_jsonl(Path(manifest["original_partial"]))
    if len(prefix) != manifest["completed_prefix_count"] or [row["id"] for row in prefix] != ids[:len(prefix)]:
        raise ValueError("Partial evaluation prefix changed")
    original_config = json.loads((Path(manifest["original_partial"]).parent / "EVAL_CONFIG.json").read_text())
    if original_config["items_sha256"] != digest(source) or original_config["seed"] != manifest["seed"]:
        raise ValueError("Original evaluation config mismatch")
    all_rows = list(prefix)
    for number, expected_count in enumerate(manifest["shard_counts"]):
        shard_items = args.manifest.parent / f"shard{number}_items.jsonl"
        result_dir = args.manifest.parent / f"shard{number}" / "result"
        config = json.loads((result_dir / "EVAL_CONFIG.json").read_text())
        done = json.loads((result_dir / "DONE.json").read_text())
        rows = read_jsonl(result_dir / "predictions.jsonl")
        if len(rows) != expected_count or done["count"] != expected_count:
            raise ValueError(f"Shard {number} is incomplete")
        if config["items_sha256"] != digest(shard_items) or config["count"] != expected_count:
            raise ValueError(f"Shard {number} item identity mismatch")
        for key in ("model", "tokenizer_sha256", "generation", "seed", "polymath_scorer_commit"):
            if config[key] != original_config[key]:
                raise ValueError(f"Shard {number} changed {key}")
        expected_ids = [item["id"] for item in read_jsonl(shard_items)]
        if [row["id"] for row in rows] != expected_ids:
            raise ValueError(f"Shard {number} row order changed")
        all_rows.extend(rows)
    rows_by_id = {row["id"]: row for row in all_rows}
    if len(rows_by_id) != len(all_rows) or set(rows_by_id) != set(ids):
        raise ValueError("Duplicate or missing result ID")
    ordered = [rows_by_id[item_id] for item_id in ids]
    for item, row in zip(items, ordered):
        for key in ("id", "dataset", "language", "gold_answer", "question_target"):
            if item[key] != row[key]:
                raise ValueError(f"Result row changed {key}: {item['id']}")
        if row["generation_seed"] != manifest["seed"]:
            raise ValueError(f"Result row changed seed: {item['id']}")
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "EVAL_CONFIG.json").write_text(json.dumps(original_config, ensure_ascii=False, indent=2) + "\n")
    (args.output_dir / "predictions.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered))
    by_dataset = {}
    for dataset in sorted(set(row["dataset"] for row in ordered)):
        subset = [row for row in ordered if row["dataset"] == dataset]
        by_dataset[dataset] = {"count": len(subset), "correct": sum(bool(row["math_correct"]) for row in subset), "accuracy": sum(bool(row["math_correct"]) for row in subset) / len(subset)}
    summary = {"count": len(ordered), "scored_count": sum(row["scoring_status"] in ("scored", "official_math_equal") for row in ordered), "correct": sum(bool(row["math_correct"]) for row in ordered), "by_dataset": by_dataset, "scoring_statuses": dict(Counter(row["scoring_status"] for row in ordered)), "prompt_tokens_recorded": sum(row["prompt_tokens"] for row in ordered), "generated_tokens": sum(row["generated_tokens"] for row in ordered), "generation_seconds": sum(row["generation_seconds"] for row in ordered), "truncated_count": sum(bool(row["truncated"]) for row in ordered), "valid_final_count": sum(bool(row["final_section_present"] and row["boxed_count_final"] == 1) for row in ordered), "source_items_sha256": digest(source), "manifest_sha256": digest(args.manifest), "prediction_file": "predictions.jsonl"}
    (args.output_dir / "DONE.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
