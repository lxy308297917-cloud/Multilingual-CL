"""Screen published English source solutions against provided gold answers."""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

from common import dump_json
from eval_math import balanced_boxes, exact_number


def normalize(s):
    return re.sub(r"\s+", "", s).strip(".$ ")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    counts = Counter()
    examples = []
    for line in a.inventory.open():
        row = json.loads(line)
        boxes = balanced_boxes(row["reference_en_original"])
        gold = row["gold_answer"]
        pred = boxes[-1] if boxes else None
        if not boxes:
            status = "no_boxed_reference"
        elif len(boxes) != 1:
            status = "multiple_boxed_reference"
        elif exact_number(gold) is not None and exact_number(pred) is not None:
            status = "numeric_match" if exact_number(gold) == exact_number(pred) else "numeric_mismatch"
        elif normalize(pred) == normalize(gold):
            status = "symbolic_string_match"
        else:
            status = "symbolic_unresolved"
        counts[status] += 1
        if status in ("numeric_mismatch", "symbolic_unresolved", "multiple_boxed_reference", "no_boxed_reference") and len(examples) < 100:
            examples.append({"problem_id": row["problem_id"], "status": status, "gold_answer": gold, "boxed_answers": boxes[:3], "question_en": row["question_en"]})
    dump_json(a.output, {"count": sum(counts.values()), "statuses": dict(counts), "examples": examples, "warning": "Source answer matching is screening only; mathematical reasoning and symbolic equivalence require independent review."})
    print(json.dumps(dict(counts), indent=2))


if __name__ == "__main__":
    main()
