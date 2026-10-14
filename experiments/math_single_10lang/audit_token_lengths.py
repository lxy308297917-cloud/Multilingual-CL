"""Measure immutable source/prompt lengths before any generated target labels exist."""
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer
from common import CONFIG, dump_json, messages


def percentile(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs)-1, round((len(xs)-1)*p))]


def stats(xs):
    return {"count": len(xs), "p50": percentile(xs,.50), "p90": percentile(xs,.90), "p95": percentile(xs,.95), "p99": percentile(xs,.99), "max": max(xs)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--splits", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    split = json.loads(a.splits.read_text())
    ids = set(split["train"] + split["dev_tune"] + split["dev_select"])
    rows = [json.loads(x) for x in a.inventory.read_text().splitlines() if x]
    rows = [r for r in rows if r["problem_id"] in ids]
    tok = AutoTokenizer.from_pretrained(str(a.tokenizer), local_files_only=True)
    ref = [len(tok(r["reference_en_original"], add_special_tokens=False).input_ids) for r in rows]
    prompt = {l: [len(tok.apply_chat_template(messages(r["languages"][l]["question_target"], l), tokenize=True, add_generation_prompt=True, enable_thinking=True)) for r in rows] for l in CONFIG["languages"]}
    result = {"count": len(rows), "reference_en_original": stats(ref), "source_reference_over_7000": sum(n>7000 for n in ref), "student_prompt_by_language": {l: stats(v) for l,v in prompt.items()}, "note": "No target assistant completion exists yet; 2048 full-sequence acceptance cannot be calculated."}
    dump_json(a.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
