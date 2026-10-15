"""Fixed Qwen3 generation and auditable per-question math scoring."""
import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
from fractions import Fraction
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import CONFIG, dump_json, messages, sha


def balanced_boxes(text):
    values = []
    for m in re.finditer(r"\\boxed\s*\{", text):
        depth, i = 1, m.end()
        while i < len(text) and depth:
            depth += (text[i] == "{") - (text[i] == "}")
            i += 1
        if depth == 0:
            values.append(text[m.end():i-1])
    return values


def final_section(raw):
    if "</think>" not in raw:
        return None
    final = raw.split("</think>", 1)[1]
    return final if final.strip() else None


def exact_number(text):
    text = unicodedata.normalize("NFKC", text).strip()
    text = "".join(str(unicodedata.digit(c)) if c.isdigit() else c for c in text)
    text = text.replace("−", "-").replace(" ", "").replace(",", "")
    percent = text.endswith("%")
    if percent:
        text = text[:-1]
    try:
        if re.fullmatch(r"[-+]?\d+/\d+", text):
            value = Fraction(text)
        elif re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", text):
            value = Fraction(text)
        else:
            return None
    except (ValueError, ZeroDivisionError):
        return None
    return value / 100 if percent else value


def score_mgsm(pred, gold):
    p, g = exact_number(pred), exact_number(gold)
    return None if g is None else (p == g if p is not None else False)


def language_status(text, lang):
    from langdetect import DetectorFactory, detect_langs
    DetectorFactory.seed = 0
    prose = re.sub(r"\\\[.*?\\\]|\\\(.*?\\\)|\$.*?\$|\\boxed\{.*?\}", " ", text, flags=re.S)
    prose = re.sub(r"[\W\d_]+", " ", prose, flags=re.UNICODE).strip()
    if len(prose) < 20:
        return {"status": "unknown", "reason": "too_short"}
    try:
        guesses = detect_langs(prose)
        target = {"zh": "zh-cn"}.get(lang, lang)
        best = guesses[0]
        return {"status": "candidate_pass" if best.lang == target and best.prob >= .8 else "candidate_fail", "best": best.lang, "prob": float(best.prob), "calibrated": False}
    except Exception as exc:
        return {"status": "unknown", "reason": str(exc)}


def evaluate_one(model, tokenizer, item, seed, polymath_equal=None, max_new_tokens=None):
    max_new_tokens = max_new_tokens or CONFIG["generation"]["max_new_tokens"]
    lang = item["language"]
    prompt = tokenizer.apply_chat_template(messages(item["question_target"], lang), tokenize=False, add_generation_prompt=True, enable_thinking=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    item_seed = seed + int(sha(item["id"])[:8], 16)
    torch.manual_seed(item_seed)
    if model.device.type == "cuda":
        torch.cuda.manual_seed_all(item_seed)
    if model.device.type == "cuda":
        torch.cuda.synchronize()
    started = time.monotonic()
    with torch.inference_mode():
        output = model.generate(**inputs, do_sample=True, temperature=.6, top_p=.95, top_k=20, min_p=0.0, max_new_tokens=max_new_tokens, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    if model.device.type == "cuda":
        torch.cuda.synchronize()
    generation_seconds = time.monotonic() - started
    generated = output[0, inputs["input_ids"].shape[-1]:]
    raw = tokenizer.decode(generated, skip_special_tokens=False)
    final = final_section(raw)
    boxes = balanced_boxes(final) if final else []
    prediction = boxes[-1] if len(boxes) == 1 else None
    gold = item["gold_answer"]
    if item["dataset"] in ("MGSM-Rev2", "GSM8K-test", "dev_tune_numeric", "dev_select_numeric"):
        correct = score_mgsm(prediction, gold) if prediction is not None else False
        scoring_status = "gold_parse_error" if exact_number(gold) is None else "scored"
    elif item["dataset"] in ("PolyMath", "dev_tune_math", "dev_select_math"):
        if polymath_equal is None:
            raise ValueError("Official PolyMath scorer is required")
        correct = bool(polymath_equal(prediction, gold)) if prediction is not None else False
        scoring_status = "official_math_equal"
    else:
        raise ValueError(f"Unknown dataset: {item['dataset']}")
    thinking = raw.split("</think>", 1)[0].removeprefix("<think>") if final else ""
    return {**item, "prompt": prompt, "prompt_tokens": int(inputs["input_ids"].shape[-1]), "raw_output": raw, "generated_tokens": int(len(generated)), "generation_seconds": generation_seconds, "finished": bool(len(generated) < max_new_tokens and int(generated[-1]) == tokenizer.eos_token_id), "truncated": bool(len(generated) >= max_new_tokens), "final_section_present": final is not None, "boxed_count_final": len(boxes), "extracted_answer": prediction, "math_correct": correct, "scoring_status": scoring_status, "thinking_language": language_status(thinking, lang), "final_language": language_status(final or "", lang), "generation_seed": seed}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--items", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, help="Pinned Base tokenizer; defaults to model directory")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--polymath-code", type=Path)
    p.add_argument("--scorer-deps", type=Path)
    p.add_argument("--limit", type=int)
    p.add_argument("--max-new-tokens", type=int, choices=(4096, 8192), default=CONFIG["generation"]["max_new_tokens"])
    a = p.parse_args()
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise FileExistsError(a.output_dir)
    items = [json.loads(x) for x in a.items.read_text().splitlines() if x]
    if a.limit:
        items = items[:a.limit]
    ids = [x["id"] for x in items]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate item IDs")
    polymath_equal = None
    if any(x["dataset"] in ("PolyMath", "dev_tune_math", "dev_select_math") for x in items):
        if not a.polymath_code or not a.scorer_deps:
            raise ValueError("PolyMath items require --polymath-code and --scorer-deps")
        sys.path[:0] = [str(a.scorer_deps), str(a.polymath_code / "eval")]
        from scripts import math_equal
        polymath_equal = math_equal
        if not polymath_equal("2", "2") or polymath_equal("2", "3"):
            raise ValueError("Official scorer self-check failed")
    a.output_dir.mkdir(parents=True)
    tokenizer_path = a.tokenizer or a.model
    tokenizer_sha256 = hashlib.sha256((tokenizer_path / "tokenizer.json").read_bytes()).hexdigest()
    dump_json(a.output_dir / "EVAL_CONFIG.json", {"model": str(a.model), "tokenizer": str(tokenizer_path), "tokenizer_sha256": tokenizer_sha256, "items": str(a.items), "items_sha256": sha(a.items.read_text()), "generation": {**CONFIG["generation"], "max_new_tokens": a.max_new_tokens}, "seed": a.seed, "count": len(items), "language_detector": "langdetect uncalibrated candidate only", "polymath_scorer_commit": "fbf4e41cae78687d6be7447dbea897357c06aaa7" if polymath_equal else None, "polymath_local_import_patch": "bundled/symeval/__init__.py exports EvaluatorMath" if polymath_equal else None})
    tokenizer = AutoTokenizer.from_pretrained(str(a.tokenizer or a.model), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(a.model), local_files_only=True, torch_dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa").eval()
    with (a.output_dir / "predictions.jsonl").open("w") as f:
        for n, item in enumerate(items, 1):
            result = evaluate_one(model, tokenizer, item, a.seed, polymath_equal, a.max_new_tokens)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{n}/{len(items)} {item['id']} {result['scoring_status']}", flush=True)
    results = [json.loads(s) for s in (a.output_dir / "predictions.jsonl").read_text().splitlines()]
    dump_json(a.output_dir / "DONE.json", {"count": len(items), "prediction_file": "predictions.jsonl", "scored_count": sum(r["scoring_status"] in ("scored", "official_math_equal") for r in results), "prompt_tokens_recorded": sum(r.get("prompt_tokens", 0) for r in results), "generated_tokens": sum(r["generated_tokens"] for r in results), "generation_seconds": sum(r.get("generation_seconds", 0) for r in results), "truncated_count": sum(r["truncated"] for r in results), "valid_final_count": sum(r["final_section_present"] and r["boxed_count_final"] == 1 for r in results)})


if __name__ == "__main__":
    main()
