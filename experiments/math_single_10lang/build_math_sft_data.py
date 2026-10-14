"""Generate auditable short English solutions and target-language labels.

Every generation is persisted independently. Records remain pending_review until
an independent human/qualified review accepts reasoning and language quality.
"""
import argparse
import json
import os
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import CONFIG, LANGUAGE_NAMES, messages, sha


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


def generate(model, tokenizer, prompt, max_new_tokens=640):
    turns = [{"role": "system", "content": "You prepare accurate, concise math teaching solutions. Return only <reasoning>...</reasoning><final>...</final> sections. Never invent missing information."}, {"role": "user", "content": prompt}]
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
    p.add_argument("--student-tokenizer", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--languages", nargs="+", choices=list(CONFIG["languages"]), default=list(CONFIG["languages"]))
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    splits = json.loads(a.splits.read_text())
    desired = {key: name for name in ("train", "dev_tune", "dev_select") for key in splits[name]}
    examples = [x for x in read_jsonl(a.inventory) if x["problem_id"] in desired]
    order = {"train": 0, "dev_tune": 1, "dev_select": 2}
    examples.sort(key=lambda x: (order[desired[x["problem_id"]]], x["problem_id"]))
    if a.limit:
        examples = examples[:a.limit]
    tokenizer = AutoTokenizer.from_pretrained(str(a.model), local_files_only=True)
    student_tokenizer = AutoTokenizer.from_pretrained(str(a.student_tokenizer), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(a.model), local_files_only=True, torch_dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa").eval()
    generator_id = "Qwen/Qwen3-8B"
    generator_revision = "b968826d9c46dd6066d109eabc6255188de91218"
    for n, row in enumerate(examples, 1):
        key = row["problem_id"]
        en_path = a.output_dir / "en_short" / f"{key}.json"
        if not en_path.exists():
            prompt = f"Check the given worked solution against the gold answer. Rewrite it as a SHORT, COMPLETE English solution with all essential reasoning. If the solution appears wrong, say so in the final section. Use <reasoning> and <final> sections only.\nQuestion: {row['question_en']}\nOriginal worked solution: {row['reference_en_original']}\nGold answer: {row['gold_answer']}"
            try:
                raw, in_tok, out_tok = generate(model, tokenizer, prompt)
                result = parse_sections(raw)
                status = "pending_review" if norm_answer(row["gold_answer"]) in norm_answer(result["final"]) else "generator_mismatch"
                save(en_path, {"problem_id": key, "question_en": row["question_en"], "reference_en_original": row["reference_en_original"], "gold_answer": row["gold_answer"], "reference_en_short": result, "raw": raw, "status": status, "input_tokens": in_tok, "output_tokens": out_tok, "prompt_hash": sha(prompt), "generator_id": generator_id, "generator_revision": generator_revision})
            except Exception as exc:
                save(en_path, {"problem_id": key, "status": "generation_error", "error": str(exc), "generator_id": generator_id, "generator_revision": generator_revision})
        en = json.loads(en_path.read_text())
        if en["status"] != "pending_review":
            continue
        for lang in a.languages:
            out = a.output_dir / lang / f"{key}.json"
            if out.exists():
                continue
            target_question = row["languages"][lang]["question_target"]
            prompt = f"Translate the following draft English short mathematical reasoning and final explanation into {LANGUAGE_NAMES[lang]}. Keep every condition, number, equation and mathematical conclusion unchanged. Use natural {LANGUAGE_NAMES[lang]} for prose. Do not solve afresh. Return <reasoning> and <final> sections only; final must include the gold answer exactly.\nTarget-language question: {target_question}\nEnglish reasoning: {en['reference_en_short']['reasoning']}\nEnglish final: {en['reference_en_short']['final']}\nGold answer: {row['gold_answer']}"
            try:
                raw, in_tok, out_tok = generate(model, tokenizer, prompt)
                result = parse_sections(raw)
                status = "pending_review" if norm_answer(row["gold_answer"]) in norm_answer(result["final"]) else "generator_mismatch"
                final = result["final"].strip()
                if "\\boxed{" not in final:
                    final += " \\boxed{" + row["gold_answer"] + "}"
                completion = "<think>" + result["reasoning"].strip() + "</think>\n" + final
                student_prompt = student_tokenizer.apply_chat_template(messages(target_question, lang), tokenize=True, add_generation_prompt=True, enable_thinking=True)
                supervised_tokens = len(student_tokenizer(completion, add_special_tokens=False).input_ids) + 1
                full_tokens = len(student_prompt) + supervised_tokens
                if full_tokens > CONFIG["max_length"]:
                    status = "overlength"
                save(out, {"source_id": f"{CONFIG['train_dataset_revision']}:{row['languages'][lang]['source_row']}", "problem_id": key, "problem_hash": key, "source": row["source"], "language": lang, "question_en": row["question_en"], "reference_en_original": row["reference_en_original"], "reference_en_short": en["reference_en_short"], "question_target": target_question, "solution_target": result, "gold_answer": row["gold_answer"], "split": desired[key], "generator_id": generator_id, "generator_revision": generator_revision, "generation_prompt_hash": sha(prompt), "quality_status": status, "reasoning_target": result["reasoning"], "final_target": result["final"], "assistant_completion": completion, "input_tokens": len(student_prompt), "supervised_tokens": supervised_tokens, "full_tokens": full_tokens, "generator_input_tokens": in_tok, "generator_output_tokens": out_tok, "raw": raw})
            except Exception as exc:
                save(out, {"problem_id": key, "language": lang, "quality_status": "generation_error", "error": str(exc), "generator_id": generator_id, "generator_revision": generator_revision})
        print(f"{n}/{len(examples)} {key}", flush=True)


if __name__ == "__main__":
    main()
