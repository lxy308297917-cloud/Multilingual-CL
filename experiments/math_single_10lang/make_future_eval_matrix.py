"""Materialize the planned three-language stage evaluation matrix after selection."""
import argparse
import csv
import json
from pathlib import Path

from common import CONFIG


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--languages", nargs=3, required=True, help="Frozen L1 L2 L3 language codes")
    p.add_argument("--selection-record", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if len(set(a.languages)) != 3 or any(lang not in CONFIG["languages"] for lang in a.languages):
        raise ValueError("Three distinct selected target languages are required")
    selected = json.loads(a.selection_record.read_text())
    if selected.get("selected_languages") != a.languages or selected.get("status") != "frozen":
        raise ValueError("Selection record must freeze these exact ordered languages")
    if a.output.exists():
        raise FileExistsError(a.output)
    methods = ("sft", "fixed_prefix_distill", "on_policy_distill")
    orders = (a.languages, list(reversed(a.languages)))
    rows = []
    for method in methods:
        for order in orders:
            for stage in range(4):
                checkpoint = "base" if stage == 0 else f"after_{order[stage-1]}"
                for language in [*a.languages, "en"]:
                    datasets = ("dev_tune", "dev_select", "MGSM-Rev2", "PolyMath") if language != "en" else ("dev_select", "GSM8K-test", "IFEval")
                    for dataset in datasets:
                        rows.append({"method": method, "order": ">".join(order), "train_seed": "", "stage": stage, "checkpoint": checkpoint, "eval_language": language, "eval_dataset": dataset, "generation_seed": "", "status": "planned", "trained_nonpadding_input_tokens": "", "trained_valid_loss_tokens": "", "student_generated_tokens": "", "teacher_scored_tokens": "", "eval_generated_tokens": "", "eval_wall_seconds": "", "score_file": ""})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"selected_languages": a.languages, "rows": len(rows), "status": "planned_only"}))


if __name__ == "__main__":
    main()
