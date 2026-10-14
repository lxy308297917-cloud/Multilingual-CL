"""Translate pinned benchmark files into the shared per-question evaluation schema."""
import argparse
import csv
import json
from pathlib import Path

import pyarrow.parquet as pq


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--language", required=True, choices=["sw", "te", "th", "bn", "zh", "ja", "ru", "de", "fr", "es", "en"])
    p.add_argument("--mgsm-dir", type=Path, required=True)
    p.add_argument("--polymath-dir", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    with (a.mgsm_dir / f"mgsm_{a.language}.tsv").open() as f:
        mgsm = list(csv.reader(f, delimiter="\t"))
    if len(mgsm) != 250 or any(len(r) != 2 for r in mgsm):
        raise ValueError("MGSM-Rev2 must have 250 question/answer rows")
    items = [{"id": f"MGSM-Rev2:{a.language}:{i}", "dataset": "MGSM-Rev2", "language": a.language, "question_target": q, "gold_answer": gold} for i, (q, gold) in enumerate(mgsm)]
    if a.polymath_dir:
        for level in ("low", "medium", "high", "top"):
            file = a.polymath_dir / a.language / f"{level}.parquet"
            rows = pq.read_table(file).to_pylist()
            if len(rows) != 125:
                raise ValueError(f"PolyMath {a.language}/{level} expected 125 questions")
            items.extend({"id": f"PolyMath:{a.language}:{level}:{i}", "dataset": "PolyMath", "difficulty": level, "language": a.language, "question_target": row["question"], "gold_answer": row["answer"], "source_id": row["id"]} for i, row in enumerate(rows))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in items))
    print(json.dumps({"language": a.language, "count": len(items)}))


if __name__ == "__main__":
    main()
