"""Build pinned English test inventory for contamination review."""
import argparse
import csv
import json
from pathlib import Path

import pyarrow.ipc as ipc
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mgsm", type=Path, required=True)
    ap.add_argument("--mgsm-original", type=Path)
    ap.add_argument("--polymath", type=Path, required=True)
    ap.add_argument("--gsm8k-arrow", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    rows = []
    with a.mgsm.open() as f:
        for i, parts in enumerate(csv.reader(f, delimiter="\t")):
            if len(parts) != 2:
                raise ValueError(f"Invalid MGSM row {i}: {len(parts)} fields")
            rows.append({"dataset": "MGSM-Rev2", "id": str(i), "question_en": parts[0], "gold_answer": parts[1]})
    if len(rows) != 250:
        raise ValueError(f"MGSM-Rev2 expected 250, got {len(rows)}")
    if a.mgsm_original:
        with a.mgsm_original.open() as f:
            original = list(csv.reader(f, delimiter="\t"))
        if len(original) != 250 or any(len(parts) != 2 for parts in original):
            raise ValueError("Original MGSM expected 250 question/answer rows")
        rows.extend({"dataset": "MGSM-original", "id": str(i), "question_en": parts[0], "gold_answer": parts[1]} for i, parts in enumerate(original))
    for file in sorted(a.polymath.glob("*.parquet")):
        part = file.stem
        for row in pq.read_table(file).to_pylist():
            rows.append({"dataset": "PolyMath", "difficulty": part, "id": row["id"], "question_en": row["question"], "gold_answer": row["answer"]})
    if sum(r["dataset"] == "PolyMath" for r in rows) != 500:
        raise ValueError("Expected 500 English PolyMath questions")
    gsm = ipc.open_stream(a.gsm8k_arrow).read_all().to_pylist()
    if len(gsm) != 1319:
        raise ValueError(f"Expected 1319 GSM8K test questions, got {len(gsm)}")
    for i, row in enumerate(gsm):
        rows.append({"dataset": "GSM8K-test", "id": str(i), "question_en": row["question"], "gold_answer": row["answer"].split("####")[-1].strip()})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"total": len(rows), "by_dataset": {k: sum(r["dataset"] == k for r in rows) for k in ("MGSM-Rev2", "MGSM-original", "PolyMath", "GSM8K-test")}}))


if __name__ == "__main__":
    main()
