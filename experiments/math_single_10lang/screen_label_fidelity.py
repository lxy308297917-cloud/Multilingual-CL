"""Flag numeric, unit, answer, and language risks without accepting labels."""
import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from build_math_sft_data import norm_answer
from common import CONFIG, dump_json
from eval_math import balanced_boxes, language_status


def numbers(text):
    text = unicodedata.normalize("NFKC", text)
    text = "".join(str(unicodedata.digit(c)) if c.isdigit() else c for c in text)
    text = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", text)
    return set(re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])", text))


def units(text):
    return {x.strip().lower() for x in re.findall(r"\\text\s*\{([^{}]+)\}", text) if re.search(r"[A-Za-z]", x)}


def screen(row):
    lang = row["language"]
    source = row["reference_en_short"]["reasoning"] + " " + row["reference_en_short"]["final"]
    target = row["reasoning_target"] + " " + row["final_target"]
    flags = list(row.get("screening_flags") or [])
    missing_question_numbers = sorted(numbers(row["question_en"]) - numbers(row["question_target"]))
    missing_solution_numbers = sorted(numbers(source) - numbers(target))
    missing_units = sorted(units(source) - units(target))
    if missing_question_numbers:
        flags.append("question_numbers_missing_candidate")
    if missing_solution_numbers:
        flags.append("solution_numbers_missing_candidate")
    if missing_units:
        flags.append("solution_units_missing_candidate")
    boxes = balanced_boxes(row["assistant_completion"].split("</think>", 1)[-1])
    if len(boxes) != 1:
        flags.append("final_boxed_count_not_one")
    elif norm_answer(boxes[0]) != norm_answer(row["gold_answer"]):
        flags.append("boxed_text_differs_from_gold_manual_check")
    reasoning_lang = language_status(row["reasoning_target"], lang)
    final_lang = language_status(row["final_target"], lang)
    return {"problem_id": row["problem_id"], "language": lang, "split": row["split"], "quality_status": row["quality_status"], "flags": sorted(set(flags)), "question_numbers_missing": missing_question_numbers, "solution_numbers_missing": missing_solution_numbers, "solution_units_missing": missing_units, "reasoning_language_candidate": reasoning_lang, "final_language_candidate": final_lang, "full_tokens": row["full_tokens"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise FileExistsError(a.output_dir)
    a.output_dir.mkdir(parents=True)
    report = {}
    with (a.output_dir / "screened.jsonl").open("w") as f:
        for lang in CONFIG["languages"]:
            rows = []
            for file in sorted((a.labels_dir / lang).glob("*.json")):
                row = json.loads(file.read_text())
                if row.get("assistant_completion"):
                    result = screen(row)
                    rows.append(result)
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
            report[lang] = {"screened": len(rows), "flags": dict(Counter(flag for row in rows for flag in row["flags"])), "reasoning_language_candidate": dict(Counter(row["reasoning_language_candidate"]["status"] for row in rows)), "interpretation": "screening only; false positives expected; no labels accepted"}
    dump_json(a.output_dir / "summary.json", report)
    print(json.dumps({lang: row["screened"] for lang, row in report.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
