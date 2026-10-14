"""Quarantine source records whose published single gold cannot be trusted blindly."""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

from common import dump_json
from eval_math import balanced_boxes


CHOICE_MARKERS = [re.compile(rf"(?m)(?:^|\n)\s*{letter}[.、):]") for letter in "ABCD"]
PART_A = re.compile(r"(?i)(?:^|[\n ])\(?a\)")
PART_B = re.compile(r"(?i)(?:^|[\n ])\(?b\)")
PROOF_TASK = re.compile(r"(?i)\b(?:prove|show\s+that|demonstrate|verify\s+that|establish\s+that|justify\s+that)\b")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--prior-splits", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise FileExistsError(a.output_dir)
    prior = json.loads(a.prior_splits.read_text())
    membership = {key: name for name, ids in prior.items() for key in ids}
    excluded = []
    counts = Counter()
    rows = [json.loads(s) for s in a.inventory.read_text().splitlines() if s]
    for row in rows:
        question = row["question_en"]
        boxes = balanced_boxes(row["reference_en_original"])
        reasons = []
        if len(boxes) != 1:
            reasons.append(f"reference_boxed_count_{len(boxes)}")
        if sum(bool(marker.search(question)) for marker in CHOICE_MARKERS) >= 3:
            reasons.append("multiple_choice_question")
        if PART_A.search(question) and PART_B.search(question):
            reasons.append("multipart_question")
        if PROOF_TASK.search(question):
            reasons.append("proof_task_not_answer_scorable")
        if reasons:
            excluded.append({"problem_id": row["problem_id"], "prior_split": membership.get(row["problem_id"], "already_excluded"), "reasons": reasons, "gold_answer": row["gold_answer"], "boxed_answers": boxes, "question_en": question})
            counts.update(reasons)
    a.output_dir.mkdir(parents=True)
    (a.output_dir / "excluded.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in excluded))
    dump_json(a.output_dir / "excluded_problem_ids.json", sorted(row["problem_id"] for row in excluded))
    dump_json(a.output_dir / "summary.json", {"inventory_count": len(rows), "excluded_unique": len(excluded), "reason_counts": dict(counts), "prior_split_counts": dict(Counter(row["prior_split"] for row in excluded)), "policy": "Conservative quarantine, not proof that every remaining answer is correct; independent math review still required"})
    print((a.output_dir / "summary.json").read_text())


if __name__ == "__main__":
    main()
