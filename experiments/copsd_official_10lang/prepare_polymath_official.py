"""Adapt the pinned COPSD PolyMath release to the official trainer's JSON fields.

Keep both the exact 3000-question official exposure and an explicitly separate
2900/100 shared-question project split. This does not edit COPSD training code.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


LANGS = {"sw": "swa", "te": "tel", "th": "tha", "bn": "ben", "zh": "zho",
         "ja": "jpn", "ru": "rus", "de": "deu", "fr": "fra", "es": "spa"}
RELEASE = "519ea2be68fec00c8a033cb2040059ff595b06a0"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(question):
    return hashlib.sha256(question.strip().encode("utf-8")).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--benchmark-en", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise FileExistsError(a.output_dir)
    a.output_dir.mkdir(parents=True)

    source = {}
    official_order = None
    inputs = {}
    for lang, hf_code in LANGS.items():
        path = a.source_dir / f"{hf_code}-00000-of-00001.parquet"
        rows = pq.read_table(path).to_pylist()
        if len(rows) != 3000:
            raise ValueError(f"Unexpected official row count {lang}: {len(rows)}")
        by_key = {}
        for row in rows:
            if row["language_short"] != lang or row["language"] != hf_code:
                raise ValueError(f"Wrong language metadata: {lang}")
            if any(not isinstance(row.get(f), str) or not row[f].strip()
                   for f in ("problem", "problem_translated", "solution", "Answer", "source")):
                raise ValueError(f"Missing official field: {lang}")
            rid = key(row["problem"])
            if rid in by_key:
                raise ValueError(f"Duplicate English question in {lang}: {rid}")
            by_key[rid] = row
        source[lang] = by_key
        order = [key(row["problem"]) for row in rows]
        if official_order is None:
            official_order = order
        elif order != official_order:
            raise ValueError(f"Official source row order differs for {lang}")
        inputs[lang] = {"path": str(path), "sha256": sha256(path), "rows": len(rows)}

    common = set.intersection(*(set(x) for x in source.values()))
    if len(common) != 3000 or any(set(rows) != common for rows in source.values()):
        raise ValueError("Official languages do not share exactly the same English problem IDs")
    canonical = source["sw"]
    for lang, rows in source.items():
        for rid, row in rows.items():
            if any(row[f] != canonical[rid][f] for f in ("problem", "solution", "Answer", "source")):
                raise ValueError(f"Cross-language English source mismatch: {lang}/{rid}")

    benchmark = collections.defaultdict(set)
    for line in a.benchmark_en.read_text().splitlines():
        item = json.loads(line)
        benchmark[item["dataset"]].add(key(item["question_en"]))
    overlaps = {name: sorted(common & ids) for name, ids in benchmark.items()}
    # Shared deterministic split. All languages have identical English IDs.
    dev_ids = set(sorted(common)[:100])
    train_ids = common - dev_ids
    summary = {"official_repo": "yihongLiu/COPSD-PolyMath-TrainDataset", "revision": RELEASE,
               "input_files": inputs, "common_english_questions": len(common),
               "output_row_order": "preserved official parquet order in all ten languages",
               "project_split": {"train": len(train_ids), "dev": len(dev_ids),
                                 "rule": "first 100 lexicographically sorted SHA256(English question.strip()) IDs are dev"},
               "exact_english_benchmark_overlap": {name: {"count": len(ids), "ids": ids} for name, ids in overlaps.items()},
               "benchmark_manifest_sha256": sha256(a.benchmark_en),
               "note": "Exact-string overlap only; translation/semantic near-duplicates need separate review."}
    for lang, rows in source.items():
        adapted = []
        for rid in official_order:
            r = rows[rid]
            adapted.append({"problem_id": rid, "source": r["source"], "problem": r["problem"],
                            "solution": r["solution"], "Answer": r["Answer"],
                            f"problem_{lang}": r["problem_translated"],
                            f"problem_{lang}_ok": True})
        write_json(a.output_dir / f"official_3000_{lang}.json", adapted)
        write_json(a.output_dir / f"project_train_2900_{lang}.json", [r for r in adapted if r["problem_id"] in train_ids])
        write_json(a.output_dir / f"project_dev_100_{lang}.json", [r for r in adapted if r["problem_id"] in dev_ids])
    write_json(a.output_dir / "PROJECT_TRAIN_IDS.json", sorted(train_ids))
    write_json(a.output_dir / "PROJECT_DEV_IDS.json", sorted(dev_ids))
    write_json(a.output_dir / "DATA_AUDIT.json", summary)
    print(json.dumps({"common": len(common), "project_train": len(train_ids), "project_dev": len(dev_ids),
                      "exact_benchmark_overlap": {k: len(v) for k, v in overlaps.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
