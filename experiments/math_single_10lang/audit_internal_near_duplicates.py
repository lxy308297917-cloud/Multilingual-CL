"""Conservatively quarantine likely duplicate math problems within the source pool."""
import argparse
import json
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

from common import dump_json


def norm_answer(text):
    compact = re.sub(r"\s+", "", text).lower()
    numeric_without_units = re.sub(r"\\(?:text|mathrm)\{[^{}]*\}", "", compact).replace("\\,", "")
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", numeric_without_units):
        return numeric_without_units
    return compact


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--threshold", type=float, default=0.8)
    a = p.parse_args()
    if not 0.8 <= a.threshold <= 1:
        raise ValueError("Internal lexical threshold must be at least 0.8")
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise FileExistsError(a.output_dir)
    rows = [json.loads(line) for line in a.inventory.read_text().splitlines() if line]
    features = TfidfVectorizer(analyzer="char", ngram_range=(3, 5)).fit_transform([r["question_en"] for r in rows])
    similarities = (features @ features.T).tocoo()
    pairs = []
    excluded = set()
    for i, j, score in zip(similarities.row, similarities.col, similarities.data):
        if i >= j or score < a.threshold:
            continue
        first, second = rows[i], rows[j]
        if norm_answer(first["gold_answer"]) != norm_answer(second["gold_answer"]):
            continue
        excluded.update((first["problem_id"], second["problem_id"]))
        pairs.append({"problem_id_a": first["problem_id"], "problem_id_b": second["problem_id"], "similarity": round(float(score), 6), "gold_answer": first["gold_answer"], "question_a": first["question_en"], "question_b": second["question_en"]})
    pairs.sort(key=lambda row: -row["similarity"])
    a.output_dir.mkdir(parents=True)
    (a.output_dir / "candidate_pairs.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in pairs))
    dump_json(a.output_dir / "excluded_problem_ids.json", sorted(excluded))
    dump_json(a.output_dir / "summary.json", {"inventory_count": len(rows), "threshold": a.threshold, "same_gold_pair_count": len(pairs), "quarantined_unique_ids": len(excluded), "policy": "Exclude both endpoints; high lexical similarity plus identical gold is a conservative candidate, not proof of semantic identity. Avoid train/dev paraphrase leakage."})
    print((a.output_dir / "summary.json").read_text())


if __name__ == "__main__":
    main()
