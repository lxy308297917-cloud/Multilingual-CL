"""Audit all ten COPSD splits and create question-ID inventory; no labels are fabricated."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from common import CONFIG, dump_json, normalize_question, sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--benchmark-questions", type=Path, help="JSONL with English question field; omit only for preliminary audit")
    args = ap.parse_args()
    by_language = {}
    summary = {}
    for lang, split in CONFIG["languages"].items():
        files = sorted((args.data_dir / "data").glob(f"{split}-*.parquet"))
        if len(files) != 1:
            raise ValueError(f"Expected one parquet for {split}, found {len(files)}")
        rows = pq.read_table(files[0]).to_pylist()
        required = {"source", "problem", "solution", "Answer", "problem_translated", "language", "language_short"}
        if not rows or not required <= rows[0].keys():
            raise ValueError(f"Missing columns in {split}")
        keyed = {}
        issues = []
        for i, row in enumerate(rows):
            if row["language"] != split or row["language_short"] != lang:
                issues.append({"row": i, "issue": "language_metadata"})
            if any(not str(row[k] or "").strip() for k in required):
                issues.append({"row": i, "issue": "empty_required_field"})
            key = sha(normalize_question(row["problem"]))
            if key in keyed:
                issues.append({"row": i, "issue": "duplicate_english_problem", "other": keyed[key]["row"]})
            keyed[key] = {**row, "row": i}
        by_language[lang] = keyed
        summary[lang] = {"rows": len(rows), "unique_english_questions": len(keyed), "sources": dict(collections.Counter(r["source"] for r in rows)), "issues": issues[:100], "issue_count": len(issues), "parquet_sha256": hashlib.sha256(files[0].read_bytes()).hexdigest()}
    common = set.intersection(*(set(v) for v in by_language.values()))
    union = set.union(*(set(v) for v in by_language.values()))
    inconsistent = []
    for key in sorted(common):
        answers = {normalize_question(by_language[l][key]["Answer"]) for l in by_language}
        refs = {sha(by_language[l][key]["solution"]) for l in by_language}
        if len(answers) != 1 or len(refs) != 1:
            inconsistent.append({"problem_id": key, "answers": sorted(answers), "reference_variants": len(refs)})
    benchmark = set()
    benchmark_complete = args.benchmark_questions is not None
    if args.benchmark_questions:
        for line in args.benchmark_questions.open():
            row = json.loads(line)
            q = row.get("question_en") or row.get("question")
            if q:
                benchmark.add(sha(normalize_question(q)))
    overlap = common & benchmark
    inventory = []
    for key in sorted(common - overlap):
        base = by_language["sw"][key]
        inventory.append({"problem_id": key, "source": base["source"], "question_en": base["problem"], "reference_en_original": base["solution"], "gold_answer": base["Answer"], "languages": {l: {"source_row": by_language[l][key]["row"], "question_target": by_language[l][key]["problem_translated"]} for l in CONFIG["languages"]}})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "inventory.jsonl").open("w") as f:
        for row in inventory:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {"dataset_id": CONFIG["train_dataset_id"], "dataset_revision": CONFIG["train_dataset_revision"], "languages": summary, "common_ids": len(common), "union_ids": len(union), "inconsistent_reference_count": len(inconsistent), "inconsistent_reference_examples": inconsistent[:30], "benchmark_exact_overlap": len(overlap), "benchmark_exact_overlap_ids": sorted(overlap), "benchmark_audit_complete": benchmark_complete, "split_frozen": False, "warning": "Exact normalized English string matching cannot exclude translated or paraphrased benchmark overlap. Train/dev split must wait for complete benchmark and near-duplicate audit."}
    dump_json(args.output_dir / "audit.json", report)
    print(json.dumps({k: report[k] for k in ("common_ids", "union_ids", "inconsistent_reference_count", "benchmark_exact_overlap", "benchmark_audit_complete", "split_frozen")}, indent=2))


if __name__ == "__main__":
    main()
