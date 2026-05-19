#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GSM8K evaluation on NPU/CPU/CUDA, aligned as closely as possible with lm-eval task YAMLs.

Supported modes:
- gsm8k_cot       -> aligned to gsm8k-cot.yaml
- gsm8k           -> aligned to gsm8k.yaml

Default:
- mode = gsm8k_cot

Outputs:
- summary.json
- predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# =========================
# Official-aligned few-shot samples for gsm8k-cot
# From user's provided gsm8k-cot.yaml
# =========================
GSM8K_COT_FEWSHOT = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "target": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6."
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "target": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5."
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "target": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39."
    },
    {
        "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "target": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8."
    },
    {
        "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "target": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9."
    },
    {
        "question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "target": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29."
    },
    {
        "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "target": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33."
    },
    {
        "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "target": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8."
    },
]


def pick_device() -> str:
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def pick_dtype(device: str):
    if device == "cpu":
        return torch.float32
    return torch.bfloat16


def load_model_and_tokenizer(model_name_or_path: str, cache_dir: Optional[str]):
    device = pick_device()
    dtype = pick_dtype(device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=None,
        low_cpu_mem_usage=False,
    )
    model.to(device)
    model.eval()
    return model, tokenizer, device


def normalize_target_text(s: str) -> str:
    """
    Aligned to exact_match regex ignore behavior from official YAML:
    ignore ',' '$' trailing '.' and text before ####
    """
    s = s.strip()
    s = re.sub(r"(?s).*####\s*", "", s)
    s = s.replace(",", "")
    s = s.replace("$", "")
    s = re.sub(r"\.$", "", s.strip())
    return s.strip()


def extract_gold_answer(answer: str) -> str:
    """
    For gsm8k-cot.yaml:
    doc_to_target: answer.split('####')[-1].strip()
    """
    return normalize_target_text(answer.split("####")[-1].strip())


def build_prompt_gsm8k_cot(question: str) -> str:
    """
    Aligned to gsm8k-cot.yaml:
    doc_to_text: 'Q: {{question}}\n\nA:'
    num_fewshot: 8
    """
    parts: List[str] = []
    for ex in GSM8K_COT_FEWSHOT:
        parts.append(f"Q: {ex['question']}\n\nA: {ex['target']}")
    parts.append(f"Q: {question}\n\nA:")
    return "\n\n".join(parts)


def build_prompt_gsm8k(question: str, fewshot_examples: List[Dict[str, str]]) -> str:
    """
    Approx aligned to gsm8k.yaml:
    doc_to_text: "Question: {{question}}\\nAnswer:"
    num_fewshot: 5
    Since gsm8k.yaml itself relies on train fewshot sampling in harness,
    here we mimic with first N train examples directly.
    """
    parts: List[str] = []
    for ex in fewshot_examples:
        parts.append(
            f"Question: {ex['question']}\nAnswer: {extract_gold_answer(ex['answer'])}"
        )
    parts.append(f"Question: {question}\nAnswer:")
    return "\n\n".join(parts)


def build_input_text(prompt: str, tokenizer, apply_chat_template: bool) -> str:
    if not apply_chat_template:
        return prompt
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return prompt


@torch.no_grad()
def generate_one(
    model,
    tokenizer,
    device: str,
    prompt: str,
    apply_chat_template: bool,
    max_new_tokens: int,
) -> str:
    input_text = build_input_text(prompt, tokenizer, apply_chat_template)
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    return text.strip()


def extract_prediction_strict_gsm8k_cot(text: str) -> Optional[str]:
    """
    Official gsm8k-cot.yaml strict regex:
    The answer is (\-?[0-9\.\,]+).
    """
    m = re.search(r"The answer is (\-?[0-9\.\,]+)\.", text)
    if not m:
        return None
    return normalize_target_text(m.group(1))


def extract_prediction_flexible(text: str) -> Optional[str]:
    """
    Official flexible-extract style:
    (-?[$0-9.,]{2,})|(-?[0-9]+)
    take last captured effective number
    """
    matches = re.findall(r"(-?[$0-9.,]{2,})|(-?[0-9]+)", text)
    if not matches:
        return None

    # Keep the last non-empty group to mimic group_select=-1 flavor
    last_val = None
    for g1, g2 in matches:
        val = g1 if g1 else g2
        if val:
            last_val = val

    if last_val is None:
        return None
    return normalize_target_text(last_val)


def exact_match(pred: Optional[str], gold: str) -> bool:
    if pred is None:
        return False
    return normalize_target_text(pred) == normalize_target_text(gold)


def parse_args():
    parser = argparse.ArgumentParser(description="GSM8K evaluation on NPU")

    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=None)

    parser.add_argument(
        "--mode",
        type=str,
        default="gsm8k_cot",
        choices=["gsm8k_cot", "gsm8k"],
        help="Which official-style task to mimic.",
    )

    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Apply tokenizer chat template before generation.",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Generation max_new_tokens.",
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="0 means use all test samples.",
    )

    parser.add_argument(
        "--save_raw_prompt",
        action="store_true",
        help="Whether to save full prompt in predictions.jsonl",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("✅ Start GSM8K evaluation")
    print(f"model_name_or_path : {args.model_name_or_path}")
    print(f"output_dir         : {output_dir}")
    print(f"mode               : {args.mode}")
    print(f"apply_chat_template: {args.apply_chat_template}")
    print(f"max_new_tokens     : {args.max_new_tokens}")
    print(f"max_samples        : {args.max_samples}")
    print("=" * 60)

    model, tokenizer, device = load_model_and_tokenizer(
        args.model_name_or_path,
        args.cache_dir,
    )
    print(f"🖥️ device = {device}")

    test_ds = load_dataset(
        "gsm8k",
        "main",
        cache_dir=args.cache_dir,
        split="test",
    )

    if args.max_samples and args.max_samples > 0:
        test_ds = test_ds.select(range(min(args.max_samples, len(test_ds))))

    # For gsm8k non-cot mode, harness fewshot_split=train, num_fewshot=5
    gsm8k_train_fewshot = None
    if args.mode == "gsm8k":
        train_ds = load_dataset(
            "gsm8k",
            "main",
            cache_dir=args.cache_dir,
            split="train",
        )
        gsm8k_train_fewshot = [train_ds[i] for i in range(5)]

    print(f"📥 loaded test samples = {len(test_ds)}")

    details_path = output_dir / "predictions.jsonl"

    strict_hits = 0
    flexible_hits = 0
    total = 0

    with details_path.open("w", encoding="utf-8") as fout:
        for idx, doc in enumerate(test_ds, start=1):
            question = doc["question"]
            gold = extract_gold_answer(doc["answer"])

            if args.mode == "gsm8k_cot":
                prompt = build_prompt_gsm8k_cot(question)
            else:
                prompt = build_prompt_gsm8k(question, gsm8k_train_fewshot)

            response = generate_one(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompt=prompt,
                apply_chat_template=args.apply_chat_template,
                max_new_tokens=args.max_new_tokens,
            )

            if args.mode == "gsm8k_cot":
                pred_strict = extract_prediction_strict_gsm8k_cot(response)
            else:
                # gsm8k.yaml doesn't enforce "The answer is ...", so strict is same as flexible here
                pred_strict = extract_prediction_flexible(response)

            pred_flexible = extract_prediction_flexible(response)

            strict_ok = exact_match(pred_strict, gold)
            flexible_ok = exact_match(pred_flexible, gold)

            strict_hits += int(strict_ok)
            flexible_hits += int(flexible_ok)
            total += 1

            record = {
                "index": idx - 1,
                "question": question,
                "gold_answer": gold,
                "response": response,
                "pred_strict": pred_strict,
                "pred_flexible": pred_flexible,
                "strict_match": strict_ok,
                "flexible_match": flexible_ok,
            }
            if args.save_raw_prompt:
                record["prompt"] = prompt

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            if idx % 20 == 0 or idx == len(test_ds):
                print(
                    f"[{idx}/{len(test_ds)}] "
                    f"strict={strict_hits / total:.4f} "
                    f"flexible={flexible_hits / total:.4f}"
                )

    summary = {
        "task": args.mode,
        "model_name_or_path": args.model_name_or_path,
        "num_samples": total,
        "apply_chat_template": args.apply_chat_template,
        "strict_match_acc": strict_hits / total if total else 0.0,
        "flexible_extract_acc": flexible_hits / total if total else 0.0,
        "details_path": str(details_path),
    }

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("🎉 GSM8K evaluation finished")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary = {summary_path}")
    print(f"details = {details_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()