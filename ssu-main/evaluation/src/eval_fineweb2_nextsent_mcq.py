#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评测 FineWeb2“真实后续句选择题”。

思路：
- 给定上下文 context
- 对每个候选句 option_i，计算 P(option_i | context)
- 选择平均 token log-prob 最大的候选句
- 输出 accuracy

优点：
- 完全贴近 causal LM 训练目标
- 不依赖额外 prompt engineering
- 对小模型更稳
"""

import os
import json
import math
import argparse
from typing import List, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_device():
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def score_candidate(model, tokenizer, device, context: str, candidate: str) -> float:
    """返回 candidate 在 context 条件下的平均 token log-prob。"""
    # 用非常朴素的 continuation 方式，不加复杂提示词
    prompt = context.strip() + "\n"
    continuation = candidate.strip()

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    cont_ids = tokenizer(continuation, add_special_tokens=False)["input_ids"]
    if len(cont_ids) == 0:
        return -1e9

    input_ids = prompt_ids + cont_ids

    # 超长时从左侧裁 prompt，保证 continuation 完整保留
    max_len = min(getattr(tokenizer, "model_max_length", 4096), 4096)
    if len(input_ids) > max_len:
        overflow = len(input_ids) - max_len
        prompt_ids = prompt_ids[overflow:]
        input_ids = prompt_ids + cont_ids

    input_ids_t = torch.tensor([input_ids], dtype=torch.long, device=device)
    labels = input_ids_t.clone()
    labels[:, :len(prompt_ids)] = -100

    with torch.no_grad():
        outputs = model(input_ids=input_ids_t)
        logits = outputs.logits[:, :-1, :]
        target = input_ids_t[:, 1:]
        target_labels = labels[:, 1:]

        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)

        mask = (target_labels != -100).float()
        total = (token_log_probs * mask).sum().item()
        count = mask.sum().item()

    if count == 0:
        return -1e9
    return total / count


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print("========================================")
    print("✅ 开始评测 FineWeb2 真实后续句选择题")
    print(f"model_name_or_path : {args.model_name_or_path}")
    print(f"dataset_path       : {args.dataset_path}")
    print(f"output_dir         : {args.output_dir}")
    print(f"max_samples        : {args.max_samples}")
    print("========================================")

    device = get_device()
    print(f"🖥️ device = {device}")

    data = load_jsonl(args.dataset_path)
    if args.max_samples > 0:
        data = data[:args.max_samples]
    print(f"📥 载入样本数 = {len(data)}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device.type != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()

    correct = 0
    results = []

    for idx, ex in enumerate(data, start=1):
        context = ex["context"]
        options = ex["options"]
        gold_idx = ex["answer_idx"]

        scores = []
        for opt in options:
            s = score_candidate(model, tokenizer, device, context, opt)
            scores.append(s)

        pred_idx = max(range(len(scores)), key=lambda i: scores[i])
        is_correct = int(pred_idx == gold_idx)
        correct += is_correct

        results.append({
            "sample_id": idx,
            "doc_id": ex.get("doc_id"),
            "context": context,
            "options": options,
            "gold_idx": gold_idx,
            "gold_letter": ex.get("answer_letter"),
            "pred_idx": pred_idx,
            "pred_letter": ["A", "B", "C", "D", "E", "F"][pred_idx],
            "scores": scores,
            "correct": is_correct,
        })

        if idx % 20 == 0:
            acc = correct / idx
            print(f"[{idx}/{len(data)}] 当前 Accuracy = {acc:.4f}")

    acc = correct / max(len(data), 1)
    summary = {
        "task": "fineweb2_next_sentence_mcq",
        "num_samples": len(data),
        "accuracy": acc,
        "num_correct": correct,
        "model_name_or_path": args.model_name_or_path,
        "dataset_path": args.dataset_path,
    }

    summary_path = os.path.join(args.output_dir, "summary.json")
    detail_path = os.path.join(args.output_dir, "predictions.jsonl")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(detail_path, "w", encoding="utf-8") as f:
        for x in results:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    print("========================================")
    print("🎉 评测完成")
    print(f"Accuracy = {acc:.4f}")
    print(f"summary   = {summary_path}")
    print(f"details   = {detail_path}")
    print("========================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="/home/HwHiAiUser/cl_workspace/hf_cache")
    parser.add_argument("--max_samples", type=int, default=200)
    args = parser.parse_args()
    main(args)
