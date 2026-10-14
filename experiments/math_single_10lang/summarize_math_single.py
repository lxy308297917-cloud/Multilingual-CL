"""Paired fixed-protocol Base versus single checkpoint summary."""
import argparse
import json
import random
from pathlib import Path

from common import dump_json


def load_run(path):
    path = Path(path)
    cfg = json.loads((path / "EVAL_CONFIG.json").read_text())
    done = json.loads((path / "DONE.json").read_text())
    rows = [json.loads(x) for x in (path / "predictions.jsonl").read_text().splitlines() if x]
    if len(rows) != done["count"] or len(rows) != done["scored_count"]:
        raise ValueError(f"Incomplete or unscored evaluation: {path}")
    keyed = {r["id"]: r for r in rows}
    if len(keyed) != len(rows):
        raise ValueError(f"Duplicate evaluation IDs: {path}")
    if any(r["math_correct"] is None for r in rows):
        raise ValueError(f"Unscored predictions: {path}")
    return cfg, keyed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--single", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=2000)
    a = p.parse_args()
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise FileExistsError(a.output_dir)
    base_cfg, base = load_run(a.base)
    single_cfg, single = load_run(a.single)
    for key in ("items_sha256", "seed", "generation"):
        if base_cfg[key] != single_cfg[key]:
            raise ValueError(f"Protocol mismatch in {key}")
    if set(base) != set(single):
        raise ValueError("Question IDs differ")
    ids = sorted(base)
    pairs = []
    for key in ids:
        b, s = base[key], single[key]
        if b["gold_answer"] != s["gold_answer"] or b["question_target"] != s["question_target"]:
            raise ValueError(f"Question/gold mismatch: {key}")
        pairs.append({"id": key, "dataset": b["dataset"], "language": b["language"], "base_correct": bool(b["math_correct"]), "single_correct": bool(s["math_correct"]), "gain": int(bool(s["math_correct"])) - int(bool(b["math_correct"])), "base_output": b["raw_output"], "single_output": s["raw_output"]})
    rng = random.Random(42)
    gains = [r["gain"] for r in pairs]
    samples = []
    for _ in range(a.bootstrap):
        samples.append(sum(gains[rng.randrange(len(gains))] for _ in gains)/len(gains))
    samples.sort()
    a.output_dir.mkdir(parents=True)
    (a.output_dir / "paired_questions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in pairs))
    dump_json(a.output_dir / "summary.json", {"count": len(pairs), "base_accuracy": sum(r["base_correct"] for r in pairs)/len(pairs), "single_accuracy": sum(r["single_correct"] for r in pairs)/len(pairs), "gain": sum(gains)/len(gains), "paired_bootstrap_95_percentile_interval": [samples[int(.025*len(samples))], samples[int(.975*len(samples))-1]], "bootstrap_seed": 42, "bootstrap_replicates": a.bootstrap, "language_status": "candidate detector uncalibrated; not aggregated as language pass"})
    print((a.output_dir / "summary.json").read_text())


if __name__ == "__main__":
    main()
