"""Freeze shared train-side teacher-check items after single selection and label review."""
import argparse
import json
import random
from pathlib import Path

from common import CONFIG, dump_json, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--splits", type=Path, required=True)
    p.add_argument("--selection-record", type=Path, required=True)
    p.add_argument("--labels-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    if not 100 <= a.count <= 200:
        raise ValueError("Teacher-check count must be 100-200 shared train IDs")
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise FileExistsError(a.output_dir)
    selection = json.loads(a.selection_record.read_text())
    langs = selection.get("selected_languages")
    if selection.get("status") != "frozen" or not isinstance(langs, list) or len(langs) != 3 or len(set(langs)) != 3 or any(x not in CONFIG["languages"] for x in langs):
        raise ValueError("A frozen selection of three distinct target languages is required")
    split_text = a.splits.read_text()
    if selection.get("split_sha256") != sha(split_text):
        raise ValueError("Selection record does not match the frozen train/dev split")
    splits = json.loads(split_text)
    required = splits["train"] + splits["dev_tune"]
    for lang in langs:
        for key in required:
            file = a.labels_dir / lang / f"{key}.json"
            if not file.exists() or json.loads(file.read_text()).get("quality_status") != "accepted":
                raise ValueError(f"Single-stage label gate not passed: {lang}/{key}")
    rng = random.Random(a.seed)
    chosen = rng.sample(splits["train"], a.count)
    inventory = {x["problem_id"]: x for x in (json.loads(line) for line in a.inventory.read_text().splitlines() if line)}
    rows = []
    for key in chosen:
        source = inventory[key]
        english = json.loads((a.labels_dir / "en_short" / f"{key}.json").read_text())
        if english.get("status") != "accepted" or english.get("problem_id") != key:
            raise ValueError(f"English short solution not accepted: {key}")
        for lang in langs:
            label = json.loads((a.labels_dir / lang / f"{key}.json").read_text())
            if label.get("quality_status") != "accepted" or label.get("problem_id") != key or label.get("language") != lang:
                raise ValueError(f"Target label not accepted: {lang}/{key}")
            rows.append({"problem_id": key, "language": lang, "split": "train_teacher_check", "student_visible": {"question_target": source["languages"][lang]["question_target"]}, "teacher_only": {"question_en": source["question_en"], "reference_en_short": english["reference_en_short"]}, "gold_answer": source["gold_answer"], "target_reference_completion": label["assistant_completion"]})
    a.output_dir.mkdir(parents=True)
    (a.output_dir / "items.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    dump_json(a.output_dir / "MANIFEST.json", {"status": "frozen_input_only", "languages": langs, "shared_train_ids": chosen, "items": len(rows), "seed": a.seed, "split_sha256": sha(split_text), "selection_sha256": sha(a.selection_record.read_text()), "teacher_only_fields_must_not_enter_student_prompt": ["question_en", "reference_en_short"]})
    print(json.dumps({"languages": langs, "shared_ids": len(chosen), "items": len(rows)}))


if __name__ == "__main__":
    main()
