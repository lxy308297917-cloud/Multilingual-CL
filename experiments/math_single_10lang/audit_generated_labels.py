"""Summarize generated labels and export stratified independent-review packets."""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

from common import CONFIG, dump_json


def quantiles(xs):
    xs = sorted(xs)
    return {k: xs[min(len(xs)-1, round((len(xs)-1)*p))] for k,p in {"p50":.5,"p90":.9,"p95":.95,"p99":.99,"max":1}.items()} if xs else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels-dir", type=Path, required=True)
    p.add_argument("--splits", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    splits = json.loads(a.splits.read_text())
    report = {}
    a.output_dir.mkdir(parents=True, exist_ok=True)
    for lang in CONFIG["languages"]:
        records = {}
        for name in ("train", "dev_tune", "dev_select"):
            for key in splits[name]:
                file = a.labels_dir / lang / f"{key}.json"
                if file.exists():
                    row = json.loads(file.read_text())
                    if row.get("problem_id") != key or row.get("language") != lang:
                        raise ValueError(f"Identity mismatch: {file}")
                    records[key] = row
        statuses = Counter(r.get("quality_status", "unknown") for r in records.values())
        lengths = [r["full_tokens"] for r in records.values() if isinstance(r.get("full_tokens"), int)]
        required = splits["train"] + splits["dev_tune"]
        formal_ready = all(k in records and records[k].get("quality_status") == "accepted" for k in required)
        report[lang] = {"expected": {k: len(splits[k]) for k in ("train", "dev_tune", "dev_select")}, "present": len(records), "statuses": dict(statuses), "full_tokens": quantiles(lengths), "over_2048": sum(n>CONFIG["max_length"] for n in lengths), "generator_input_tokens_recorded": sum(r.get("generator_input_tokens") or 0 for r in records.values()), "generator_output_tokens_recorded": sum(r.get("generator_output_tokens") or 0 for r in records.values()), "generator_token_cost_missing_rows": sum(r.get("generator_input_tokens") is None or r.get("generator_output_tokens") is None for r in records.values()), "formal_ready": formal_ready}
        rng = random.Random(42)
        chosen = []
        for name, n in (("train", 30), ("dev_tune", 10), ("dev_select", 10)):
            pool = [records[k] for k in splits[name] if k in records and records[k].get("assistant_completion")]
            pool.sort(key=lambda r: r.get("full_tokens", 0))
            buckets = [pool[i::5] for i in range(5)]
            for bucket in buckets:
                rng.shuffle(bucket)
                chosen.extend(bucket[:max(1, n//5)])
        with (a.output_dir / f"review_{lang}.jsonl").open("w") as f:
            for row in chosen:
                f.write(json.dumps({k: row.get(k) for k in ("problem_id", "split", "language", "question_en", "question_target", "reference_en_original", "reference_en_short", "gold_answer", "reasoning_target", "final_target", "assistant_completion", "full_tokens", "quality_status", "screening_flags", "generator_input_tokens", "generator_output_tokens")}, ensure_ascii=False) + "\n")
    english = [json.loads(f.read_text()) for f in (a.labels_dir / "en_short").glob("*.json")] if (a.labels_dir / "en_short").exists() else []
    report["_english_short"] = {"present": len(english), "statuses": dict(Counter(r.get("status", "unknown") for r in english)), "generator_input_tokens_recorded": sum(r.get("input_tokens") or 0 for r in english), "generator_output_tokens_recorded": sum(r.get("output_tokens") or 0 for r in english), "generator_token_cost_missing_rows": sum(r.get("input_tokens") is None or r.get("output_tokens") is None for r in english)}
    dump_json(a.output_dir / "label_audit.json", report)
    print(json.dumps({l: {"present": x["present"], "statuses": x["statuses"], "formal_ready": x.get("formal_ready")} for l,x in report.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
