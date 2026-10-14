"""Generate auditable short English solutions and target-language labels.

Every generation is persisted independently. Records remain pending_review until
an independent human/qualified review accepts reasoning and language quality.
"""
import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import CONFIG, LANGUAGE_NAMES, messages, sha
from eval_math import balanced_boxes


def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s]


def parse_sections(raw):
    raw = raw.split("</think>")[-1].strip()
    reasoning = re.search(r"<reasoning>(.*?)</reasoning>", raw, re.S | re.I)
    final = re.search(r"<final>(.*?)</final>", raw, re.S | re.I)
    if reasoning and final:
        return {"reasoning": reasoning.group(1).strip(), "final": final.group(1).strip()}
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        raise ValueError("No reasoning/final sections")
    obj = json.loads(match.group())
    if not isinstance(obj.get("reasoning"), str) or not isinstance(obj.get("final"), str):
        raise ValueError("Missing reasoning/final")
    return obj


def norm_answer(s):
    return re.sub(r"\s+", "", s.replace("\\boxed{", "").replace("}", "").replace("$", "")).lower()


def generate(model, tokenizer, prompt, max_new_tokens=640, system="You prepare accurate, concise math teaching solutions. Return only <reasoning>...</reasoning><final>...</final> sections. Never invent missing information."):
    turns = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(turns, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    if inputs["input_ids"].shape[-1] > 7000:
        raise ValueError("Generator input exceeds 7000 tokens; requires shorter verified source")
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=False), int(inputs["input_ids"].shape[-1]), int(output.shape[-1] - inputs["input_ids"].shape[-1])


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--splits", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--generator-id", default="Qwen/Qwen3-8B")
    p.add_argument("--generator-revision", default="b968826d9c46dd6066d109eabc6255188de91218")
    p.add_argument("--device-map", default="cuda:0")
    p.add_argument("--max-new-tokens", type=int, default=640)
    p.add_argument("--student-tokenizer", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--languages", nargs="+", choices=list(CONFIG["languages"]), default=list(CONFIG["languages"]))
    p.add_argument("--limit", type=int)
    p.add_argument("--smoke-train", type=int)
    p.add_argument("--smoke-dev", type=int)
    p.add_argument("--english-only", action="store_true", help="Qualify short English math reasoning before spending target-language generation tokens")
    a = p.parse_args()
    splits = json.loads(a.splits.read_text())
    desired = {key: name for name in ("train", "dev_tune", "dev_select") for key in splits[name]}
    examples = [x for x in read_jsonl(a.inventory) if x["problem_id"] in desired]
    order = {"train": 0, "dev_tune": 1, "dev_select": 2}
    examples.sort(key=lambda x: (order[desired[x["problem_id"]]], x["problem_id"]))
    if a.smoke_train is not None or a.smoke_dev is not None:
        if a.limit is not None:
            raise ValueError("--limit and --smoke-train/--smoke-dev are mutually exclusive")
        train_examples = [x for x in examples if desired[x["problem_id"]] == "train"][:a.smoke_train or 0]
        dev_examples = [x for x in examples if desired[x["problem_id"]] == "dev_tune"][:a.smoke_dev or 0]
        examples = train_examples + dev_examples
    if a.limit:
        examples = examples[:a.limit]
    tokenizer = AutoTokenizer.from_pretrained(str(a.model), local_files_only=True)
    student_tokenizer = AutoTokenizer.from_pretrained(str(a.student_tokenizer), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(a.model), local_files_only=True, torch_dtype=torch.bfloat16, device_map=a.device_map, attn_implementation="sdpa").eval()
    print(json.dumps({"generator_id": a.generator_id, "generator_revision": a.generator_revision, "model_device_map_counts": dict(Counter(str(device) for device in model.hf_device_map.values()))}, ensure_ascii=False), flush=True)
    generator_id = a.generator_id
    generator_revision = a.generator_revision
    for n, row in enumerate(examples, 1):
        key = row["problem_id"]
        en_path = a.output_dir / "en_short" / f"{key}.json"
        prompt = f"Extract a SHORT, COMPLETE, mathematically valid solution from the supplied worked solution. Preserve its correct equations and numerical values; omit unnecessary algebra. Before writing, substitute any parameter values you mention into all given conditions. If a proposed value fails a condition, correct it. Explain the mathematics directly, without mentioning any checking process. Return only <reasoning>...</reasoning><final>...</final>, with exactly one boxed final answer.\nQuestion: {row['question_en']}\nOriginal worked solution: {row['reference_en_original']}\nGold answer: {row['gold_answer']}"
        if not en_path.exists():
            raw, in_tok, out_tok = None, None, None
            try:
                raw, in_tok, out_tok = generate(model, tokenizer, prompt, a.max_new_tokens, "You write accurate, concise math solutions from verified source reasoning.")
                result = parse_sections(raw)
                flags = [] if norm_answer(row["gold_answer"]) in norm_answer(result["final"]) else ["gold_text_not_in_final_manual_check"]
                boxes = balanced_boxes(result["final"])
                if len(boxes) != 1:
                    flags.append(f"english_final_boxed_count_{len(boxes)}")
                elif norm_answer(boxes[0]) != norm_answer(row["gold_answer"]):
                    flags.append("english_boxed_text_mismatch_manual_check")
                status = "generation_truncated" if out_tok >= a.max_new_tokens else "pending_review"
                save(en_path, {"problem_id": key, "question_en": row["question_en"], "reference_en_original": row["reference_en_original"], "gold_answer": row["gold_answer"], "reference_en_short": result, "raw": raw, "status": status, "screening_flags": flags, "input_tokens": in_tok, "output_tokens": out_tok, "prompt_hash": sha(prompt), "generator_id": generator_id, "generator_revision": generator_revision})
            except Exception as exc:
                status = "generation_truncated" if out_tok is not None and out_tok >= a.max_new_tokens else "generation_error"
                save(en_path, {"problem_id": key, "question_en": row["question_en"], "reference_en_original": row["reference_en_original"], "gold_answer": row["gold_answer"], "status": status, "error": str(exc), "raw": raw, "input_tokens": in_tok, "output_tokens": out_tok, "prompt_hash": sha(prompt), "generator_id": generator_id, "generator_revision": generator_revision})
        en = json.loads(en_path.read_text())
        expected_en = {"problem_id": key, "question_en": row["question_en"], "reference_en_original": row["reference_en_original"], "gold_answer": row["gold_answer"], "prompt_hash": sha(prompt), "generator_id": generator_id, "generator_revision": generator_revision}
        if any(en.get(field) != value for field, value in expected_en.items()):
            raise ValueError(f"Existing English short label has stale identity or generator prompt: {en_path}")
        if en["status"] != "pending_review":
            continue
        if a.english_only:
            print(f"{n}/{len(examples)} {key} English {en['status']}", flush=True)
            continue
        for lang in a.languages:
            out = a.output_dir / lang / f"{key}.json"
            target_question = row["languages"][lang]["question_target"]
            local_hint = "In Swahili, use 'tarakimu' for digit and 'jumla ya tarakimu' for digit sum; avoid English words such as digit, solution, correct, and number in prose. " if lang == "sw" else ""
            prompt = f"Translate the following concise mathematical solution into natural {LANGUAGE_NAMES[lang]}. Copy every equation, number, condition and conclusion exactly. Do not add new steps, variables, metadata or explanations beyond the English source. Use the target question's terminology. Translate every explanatory sentence; mathematical symbols and proper names may stay. {local_hint}Do not repeat phrases. Return only <reasoning>...</reasoning><final>...</final>, with exactly one boxed final answer matching the source.\nTarget-language question: {target_question}\nEnglish reasoning: {en['reference_en_short']['reasoning']}\nEnglish final: {en['reference_en_short']['final']}\nGold answer: {row['gold_answer']}"
            if out.exists():
                existing = json.loads(out.read_text())
                expected = {"problem_id": key, "language": lang, "split": desired[key], "question_target": target_question, "gold_answer": row["gold_answer"], "generation_prompt_hash": sha(prompt), "generator_id": generator_id, "generator_revision": generator_revision}
                if any(existing.get(field) != value for field, value in expected.items()):
                    raise ValueError(f"Existing target label has stale split, identity or generator prompt: {out}")
                continue
            raw, in_tok, out_tok = None, None, None
            try:
                raw, in_tok, out_tok = generate(model, tokenizer, prompt, a.max_new_tokens, f"You translate mathematics into fluent {LANGUAGE_NAMES[lang]} faithfully and concisely.")
                result = parse_sections(raw)
                flags = [] if norm_answer(row["gold_answer"]) in norm_answer(result["final"]) else ["gold_text_not_in_final_manual_check"]
                if re.search(r"\b(?:Gold answer|English reasoning|English final|Original worked solution)\b", result["reasoning"] + " " + result["final"], re.I):
                    flags.append("prompt_metadata_echo")
                status = "generation_truncated" if out_tok >= a.max_new_tokens else "pending_review"
                final = result["final"].strip()
                if "\\boxed{" not in final:
                    final += " \\boxed{" + row["gold_answer"] + "}"
                boxes = balanced_boxes(final)
                if len(boxes) != 1:
                    flags.append(f"target_final_boxed_count_{len(boxes)}")
                elif norm_answer(boxes[0]) != norm_answer(row["gold_answer"]):
                    flags.append("target_boxed_text_mismatch_manual_check")
                completion = "<think>" + result["reasoning"].strip() + "</think>\n" + final
                student_prompt = student_tokenizer.apply_chat_template(messages(target_question, lang), tokenize=True, add_generation_prompt=True, enable_thinking=True)
                supervised_tokens = len(student_tokenizer(completion, add_special_tokens=False).input_ids) + 1
                full_tokens = len(student_prompt) + supervised_tokens
                if full_tokens > CONFIG["max_length"]:
                    status = "overlength"
                save(out, {"source_id": f"{CONFIG['train_dataset_revision']}:{row['languages'][lang]['source_row']}", "problem_id": key, "problem_hash": key, "source": row["source"], "language": lang, "question_en": row["question_en"], "reference_en_original": row["reference_en_original"], "reference_en_short": en["reference_en_short"], "question_target": target_question, "solution_target": result, "gold_answer": row["gold_answer"], "split": desired[key], "generator_id": generator_id, "generator_revision": generator_revision, "generation_prompt_hash": sha(prompt), "quality_status": status, "screening_flags": flags, "reasoning_target": result["reasoning"], "final_target": result["final"], "assistant_completion": completion, "input_tokens": len(student_prompt), "supervised_tokens": supervised_tokens, "full_tokens": full_tokens, "generator_input_tokens": in_tok, "generator_output_tokens": out_tok, "raw": raw})
            except Exception as exc:
                status = "generation_truncated" if out_tok is not None and out_tok >= a.max_new_tokens else "generation_error"
                save(out, {"problem_id": key, "language": lang, "split": desired[key], "question_target": target_question, "gold_answer": row["gold_answer"], "generation_prompt_hash": sha(prompt), "quality_status": status, "error": str(exc), "raw": raw, "generator_input_tokens": in_tok, "generator_output_tokens": out_tok, "generator_id": generator_id, "generator_revision": generator_revision})
        print(f"{n}/{len(examples)} {key}", flush=True)


if __name__ == "__main__":
    main()
