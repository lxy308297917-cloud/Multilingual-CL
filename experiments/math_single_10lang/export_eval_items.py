"""Translate pinned benchmark files into the shared per-question evaluation schema."""
import argparse
import csv
import json
from pathlib import Path

import pyarrow.ipc as ipc
import pyarrow.parquet as pq


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--language", required=True, choices=["sw", "te", "th", "bn", "zh", "ja", "ru", "de", "fr", "es", "en"])
    p.add_argument("--mgsm-dir", type=Path, required=True)
    p.add_argument("--polymath-dir", type=Path)
    p.add_argument("--gsm8k-arrow", type=Path, help="English-only pinned GSM8K test Arrow file")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    with (a.mgsm_dir / f"mgsm_{a.language}.tsv").open() as f:
        mgsm = list(csv.reader(f, delimiter="\t"))
    if len(mgsm) != 250 or any(len(r) != 2 for r in mgsm):
        raise ValueError("MGSM-Rev2 must have 250 question/answer rows")
    if a.language != "en":
        with (a.mgsm_dir / "mgsm_en.tsv").open() as f:
            en_mgsm = list(csv.reader(f, delimiter="\t"))
        if len(en_mgsm) != len(mgsm) or any(x[1] != y[1] for x, y in zip(en_mgsm, mgsm)):
            raise ValueError("MGSM translated rows do not align with English gold answers")
    items = [{"id": f"MGSM-Rev2:{a.language}:{i}", "dataset": "MGSM-Rev2", "language": a.language, "question_target": q, "gold_answer": gold} for i, (q, gold) in enumerate(mgsm)]
    if a.polymath_dir:
        for level in ("low", "medium", "high", "top"):
            file = a.polymath_dir / a.language / f"{level}.parquet"
            rows = pq.read_table(file).to_pylist()
            if len(rows) != 125:
                raise ValueError(f"PolyMath {a.language}/{level} expected 125 questions")
            if a.language != "en":
                en_rows = pq.read_table(a.polymath_dir / "en" / f"{level}.parquet").to_pylist()
                if any(x["id"].split("-")[-1] != y["id"].split("-")[-1] or x["answer"] != y["answer"] for x, y in zip(en_rows, rows)):
                    raise ValueError(f"PolyMath {a.language}/{level} does not align with English file")
            items.extend({"id": f"PolyMath:{a.language}:{level}:{i}", "dataset": "PolyMath", "difficulty": level, "language": a.language, "question_target": row["question"], "gold_answer": row["answer"], "source_id": row["id"]} for i, row in enumerate(rows))
    if a.gsm8k_arrow:
        if a.language != "en":
            raise ValueError("GSM8K test is English-only")
        rows = ipc.open_stream(a.gsm8k_arrow).read_all().to_pylist()
        if len(rows) != 1319:
            raise ValueError(f"GSM8K test expected 1319 questions, got {len(rows)}")
        items.extend({"id": f"GSM8K-test:en:{i}", "dataset": "GSM8K-test", "language": "en", "question_target": row["question"], "gold_answer": row["answer"].split("####")[-1].strip()} for i, row in enumerate(rows))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in items))
    print(json.dumps({"language": a.language, "count": len(items)}))


if __name__ == "__main__":
    main()
