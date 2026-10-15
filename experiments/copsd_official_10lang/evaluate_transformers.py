"""Single-GPU project evaluator using COPSD's released PolyMath prompt strings.

Generation backend is Transformers because vLLM is not installed on this
RTX-3090 host. Results are project metrics, never labeled vLLM reproduction.
"""
import argparse
import ast
import hashlib
import json
import time
from pathlib import Path

import torch
from math_verify import parse, verify
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def official_prompts(source):
    module = ast.parse(source.read_text())
    for node in module.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "LANGUAGE_PROMPTS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("Official LANGUAGE_PROMPTS not found")


def build_prompt(tokenizer, question, lang, prompts):
    cfg = prompts[lang.upper()]
    user = f"{question}\n\n{cfg['instruction']}"
    base = tokenizer.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=False)
    return base + "<|im_start|>assistant\n<think>\n" + cfg["think_prefix"]


def extract_boxed(text):
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    left = 0
    right = None
    for j in range(idx, len(text)):
        if text[j] == "{":
            left += 1
        elif text[j] == "}":
            left -= 1
            if left == 0:
                right = j
                break
    if right is None:
        return None
    boxed = text[idx:right + 1]
    if boxed.startswith("\\boxed{") and boxed.endswith("}"):
        return boxed[7:-1].strip()
    return None


def equal_answer(pred, gold):
    if pred is None:
        return False
    try:
        a = pred if "$" in pred or "\\boxed{" in pred else f"${pred}$"
        b = str(gold) if "$" in str(gold) or "\\boxed{" in str(gold) else f"${gold}$"
        return bool(verify(parse(b), parse(a), timeout_seconds=5))
    except Exception:
        return pred.replace("$", "").replace(" ", "").lower().strip() == str(gold).replace("$", "").replace(" ", "").lower().strip()


def load_items(path, kind, lang):
    if kind == "dev":
        for row in json.loads(path.read_text()):
            yield {"id": row["problem_id"], "question": row[f"problem_{lang}"], "answer": row["Answer"]}
    else:
        for i, line in enumerate(path.read_text().splitlines()):
            question, answer = line.split("\t", 1)
            yield {"id": str(i), "question": question, "answer": answer}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--official-eval-source", type=Path, required=True)
    p.add_argument("--base-model", type=Path, required=True)
    p.add_argument("--adapter", type=Path)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--input-kind", choices=["dev", "mgsm_rev2"], required=True)
    p.add_argument("--language", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-items", type=int, default=0)
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    if not torch.cuda.is_available() or not __import__("os").environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Explicit CUDA_VISIBLE_DEVICES and CUDA are required")
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise FileExistsError(a.output_dir)
    a.output_dir.mkdir(parents=True)
    lang = a.language.lower()
    prompts = official_prompts(a.official_eval_source)
    if lang.upper() not in prompts:
        raise ValueError(lang)
    items = list(load_items(a.input, a.input_kind, lang))
    if a.max_items:
        items = items[:a.max_items]
    if not items or a.samples < 1:
        raise ValueError("Empty evaluation or invalid samples")
    tokenizer = AutoTokenizer.from_pretrained(str(a.base_model), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(a.base_model), local_files_only=True,
                                                torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda:0").eval()
    if a.adapter:
        model = PeftModel.from_pretrained(model, str(a.adapter)).eval()
    config = {"input_kind": a.input_kind, "language": lang, "input": str(a.input),
              "input_sha256": hashlib.sha256(a.input.read_bytes()).hexdigest(),
              "official_eval_source": str(a.official_eval_source),
              "official_eval_source_sha256": hashlib.sha256(a.official_eval_source.read_bytes()).hexdigest(),
              "base_model": str(a.base_model), "adapter": str(a.adapter) if a.adapter else None,
              "samples": a.samples, "max_new_tokens": a.max_new_tokens, "seed": a.seed,
              "temperature": 1.0, "top_p": 0.95, "top_k": 0,
              "backend": "Transformers, project adaptation; COPSD official prompt strings and math_verify scorer"}
    (a.output_dir / "CONFIG.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    correct_first = pass_n = generated = finished = 0
    started = time.monotonic()
    with (a.output_dir / "ITEMS.jsonl").open("w") as f:
        for item in items:
            prompt = build_prompt(tokenizer, item["question"], lang, prompts)
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda:0")
            variants = []
            for n in range(a.samples):
                seed = int.from_bytes(hashlib.sha256(f"{a.seed}:{lang}:{item['id']}:{n}".encode()).digest()[:8], "big") % (2**32)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                tick = time.monotonic()
                with torch.inference_mode():
                    out = model.generate(**inputs, do_sample=True, temperature=1.0, top_p=.95, top_k=0,
                                         max_new_tokens=a.max_new_tokens, pad_token_id=tokenizer.eos_token_id)
                ids = out[0, inputs["input_ids"].shape[1]:]
                raw = tokenizer.decode(ids, skip_special_tokens=False)
                boxed = extract_boxed(raw)
                correct = equal_answer(boxed, item["answer"])
                terminated = int(ids[-1]) == tokenizer.eos_token_id
                variants.append({"sample": n, "seed": seed, "tokens": len(ids), "finished": terminated,
                                 "boxed_answer": boxed, "correct": correct, "seconds": time.monotonic() - tick,
                                 "raw_output": raw})
                generated += len(ids)
                finished += terminated
            correct_first += int(variants[0]["correct"])
            pass_n += int(any(x["correct"] for x in variants))
            f.write(json.dumps({**item, "prompt": prompt, "samples": variants}, ensure_ascii=False) + "\n")
            f.flush()
    result = {"items": len(items), "single_correct": correct_first, "single_accuracy": correct_first / len(items),
              "pass_n_correct": pass_n, "pass_n": pass_n / len(items), "n": a.samples,
              "generated_tokens": generated, "normal_terminations": finished,
              "generation_count": len(items) * a.samples, "seconds": time.monotonic() - started,
              "peak_gpu_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1)}
    (a.output_dir / "DONE.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
