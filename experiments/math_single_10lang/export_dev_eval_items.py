"""Export frozen COPSD development questions for generation-based selection."""
import argparse
import json
from pathlib import Path

from common import CONFIG


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--splits", type=Path, required=True)
    p.add_argument("--language", choices=[*CONFIG["languages"], "en"], required=True)
    p.add_argument("--split", choices=["dev_tune", "dev_select"], required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    split_ids = json.loads(a.splits.read_text())[a.split]
    if len(split_ids) != 200 or len(split_ids) != len(set(split_ids)):
        raise ValueError(f"Expected 200 unique {a.split} IDs")
    inventory = {r["problem_id"]: r for r in (json.loads(s) for s in a.inventory.read_text().splitlines() if s)}
    items = []
    for key in split_ids:
        row = inventory[key]
        question = row["question_en"] if a.language == "en" else row["languages"][a.language]["question_target"]
        items.append({"id": f"{a.split}:{a.language}:{key}", "dataset": f"{a.split}_math", "language": a.language,
                      "problem_id": key, "question_target": question, "gold_answer": row["gold_answer"]})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items))
    print(json.dumps({"split": a.split, "language": a.language, "count": len(items)}))


if __name__ == "__main__":
    main()
