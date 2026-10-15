"""One-GPU Qwen3-4B SW OPD engineering smoke; never a formal run.

The independent frozen teacher scores on CPU. A full-parameter FP32 CPU
master copy receives plain SGD updates, then BF16 student weights are copied
back to the GPU. This deliberately tests actual 4B rollouts and gradients on
one available 24GB card; optimizer, rollout length and data are not formal.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_math import balanced_boxes, final_section, language_status
from sw_opd_core import aligned_reverse_kl_tail
from train_sw_opd_fsdp_smoke import digest, read_jsonl, student_prompt, teacher_prompt, write_json


def sample_answer(student, tokenizer, prompt, max_new_tokens, seed, device):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    inputs = torch.tensor([prompt], dtype=torch.long, device=device)
    student.eval()
    started = time.monotonic()
    with torch.no_grad():
        outputs = student.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs), do_sample=True, temperature=1.1,
                                   top_p=.95, top_k=20, max_new_tokens=max_new_tokens,
                                   pad_token_id=tokenizer.eos_token_id, use_cache=True)
    torch.cuda.synchronize(device)
    answer = outputs[:, inputs.shape[1]:].detach()
    student.train()
    if answer.shape[1] == 0:
        raise RuntimeError("Empty student rollout")
    return answer, time.monotonic() - started


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--source-manifest", type=Path, required=True)
    p.add_argument("--source-review", type=Path, required=True)
    p.add_argument("--split-file", type=Path, required=True)
    p.add_argument("--train-ids-file", type=Path, required=True)
    p.add_argument("--dev-ids-file", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--model-revision", required=True)
    p.add_argument("--polymath-code", type=Path, required=True)
    p.add_argument("--scorer-deps", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-steps", type=int, default=5)
    p.add_argument("--rollout-max-new-tokens", type=int, default=64)
    p.add_argument("--eval-max-new-tokens", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true", required=True)
    a = p.parse_args()
    if not a.smoke or not 1 <= a.max_steps <= 20 or not 1 <= a.rollout_max_new_tokens <= 256:
        raise ValueError("Only explicit short engineering smoke is supported")
    if os.environ.get("GPU_QUEUE_GPU_COUNT") != "1" or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Use the manual one-GPU queue with explicit CUDA_VISIBLE_DEVICES")
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise FileExistsError(a.output_dir)
    splits = json.loads(a.split_file.read_text())["ids"]
    train_ids = json.loads(a.train_ids_file.read_text())
    dev_ids = json.loads(a.dev_ids_file.read_text())
    if not train_ids or not dev_ids or not set(train_ids) <= set(splits["train"]) or not set(dev_ids) <= set(splits["dev_tune"]):
        raise ValueError("Smoke IDs do not match the fixed SW split")
    source = {r["problem_id"]: r for r in read_jsonl(a.source_manifest)}
    review = {r["problem_id"]: r for r in read_jsonl(a.source_review)}
    rows = {r["problem_id"]: r for r in read_jsonl(a.inventory) if r["problem_id"] in set(train_ids + dev_ids)}
    if set(rows) != set(train_ids + dev_ids):
        raise ValueError("Missing inventory IDs")
    for key in train_ids + dev_ids:
        r, s = rows[key], source[key]
        if s.get("triage_flags") or s["question_sw"] != r["languages"]["sw"]["question_target"] or s["question_en"] != r["question_en"] or s["gold_answer"] != r["gold_answer"]:
            raise ValueError(f"Stale/flagged source: {key}")
    for key in train_ids:
        if review.get(key, {}).get("math_status") != "model_review_pass":
            raise ValueError(f"Train source lacks Codex math precheck: {key}")
    sys.path[:0] = [str(a.scorer_deps), str(a.polymath_code / "eval")]
    from scripts import math_equal
    if not math_equal("2", "2") or math_equal("2", "3"):
        raise RuntimeError("Official scorer self-check failed")
    tokenizer = AutoTokenizer.from_pretrained(str(a.model), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    a.output_dir.mkdir(parents=True)
    write_json(a.output_dir / "RUN_CONFIG.json", {
        "protocol": "single-GPU full-parameter engineering smoke only", "model": str(a.model),
        "model_revision": a.model_revision, "teacher": "independent frozen CPU Qwen3-4B",
        "optimizer": "FP32 CPU-master full-parameter plain SGD; not formal AdamW", "learning_rate": a.learning_rate,
        "max_steps": a.max_steps, "rollout_max_new_tokens": a.rollout_max_new_tokens,
        "eval_max_new_tokens": a.eval_max_new_tokens, "seed": a.seed,
        "source_status": "candidate, Codex math precheck only; not accepted Q_full",
        "train_ids": train_ids, "dev_ids": dev_ids,
        "files_sha256": {str(path): digest(path) for path in (a.inventory, a.source_manifest, a.source_review, a.split_file, a.train_ids_file, a.dev_ids_file)},
        "tokenizer_sha256": digest(a.model / "tokenizer.json"),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
    })
    device = torch.device("cuda:0")
    teacher = AutoModelForCausalLM.from_pretrained(str(a.model), local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa").eval()
    teacher.requires_grad_(False)
    teacher_sample = next(teacher.parameters()).flatten()[:1024].detach().float().clone()
    student = AutoModelForCausalLM.from_pretrained(str(a.model), local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
    student.config.use_cache = False
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    student.train()
    master = [p.detach().cpu().float().clone() for p in student.parameters()]
    writer = SummaryWriter(str(a.output_dir / "tensorboard"))

    def evaluate(step):
        items = []
        for i, key in enumerate(dev_ids):
            row = rows[key]
            answer, seconds = sample_answer(student, tokenizer, student_prompt(tokenizer, row), a.eval_max_new_tokens, a.seed + 100000 + step * 100 + i, device)
            raw = tokenizer.decode(answer[0], skip_special_tokens=False)
            final = final_section(raw)
            boxes = balanced_boxes(final) if final else []
            found = boxes[0] if len(boxes) == 1 else None
            items.append({"problem_id": key, "gold_answer": row["gold_answer"], "final_answer": found,
                          "correct": bool(math_equal(found, row["gold_answer"])) if found else False,
                          "thinking_language": language_status(raw.split("</think>", 1)[0].removeprefix("<think>"), "sw"),
                          "final_language": language_status(final or "", "sw"),
                          "finished": int(answer[0, -1]) == tokenizer.eos_token_id,
                          "generated_tokens": int(answer.shape[1]), "generation_seconds": seconds, "raw_output": raw})
        write_json(a.output_dir / f"EVAL_STEP_{step}.json", {"step": step, "items": items,
                   "correct": sum(x["correct"] for x in items), "count": len(items),
                   "note": "short smoke generation only; not fixed formal evaluation"})
        writer.add_scalar("eval_smoke/math_correct", sum(x["correct"] for x in items) / len(items), step)
        writer.flush()

    evaluate(0)
    records = []
    total_effective = 0
    for step in range(a.max_steps):
        key = train_ids[step % len(train_ids)]
        row = rows[key]
        s_prompt = student_prompt(tokenizer, row)
        t_prompt = teacher_prompt(tokenizer, row)
        answer, generation_seconds = sample_answer(student, tokenizer, s_prompt, a.rollout_max_new_tokens, a.seed + step, device)
        n = answer.shape[1]
        student_ids = torch.cat((torch.tensor([s_prompt], dtype=torch.long, device=device), answer[:, :-1]), dim=1)
        teacher_ids = torch.cat((torch.tensor([t_prompt], dtype=torch.long), answer[:, :-1].cpu()), dim=1)
        started = time.monotonic()
        with torch.inference_mode():
            teacher_logits = teacher(input_ids=teacher_ids, use_cache=False, logits_to_keep=n).logits.to(device)
        teacher_seconds = time.monotonic() - started
        student_logits = student(input_ids=student_ids, use_cache=False, logits_to_keep=n).logits
        loss = aligned_reverse_kl_tail(student_logits, teacher_logits, torch.ones((1, n), dtype=torch.bool, device=device), chunk_tokens=8)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite KL at step {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        if not torch.isfinite(grad_norm) or grad_norm <= 0 or any(p.grad is not None for p in teacher.parameters()):
            raise RuntimeError(f"Invalid gradient or unfrozen teacher at step {step}")
        updated_tensors = 0
        bf16_changed_tensors = 0
        with torch.no_grad():
            for p, m in zip(student.parameters(), master):
                if p.grad is None:
                    continue
                m.add_(p.grad.detach().cpu().float(), alpha=-a.learning_rate)
                new_value = m.to(device=device, dtype=p.dtype)
                bf16_changed_tensors += int(torch.any(new_value != p).item())
                p.copy_(new_value)
                updated_tensors += 1
        if bf16_changed_tensors == 0:
            raise RuntimeError("FP32 master changed but BF16 student policy did not; on-policy update invalid")
        student.zero_grad(set_to_none=True)
        total_effective += n
        record = {"step": step + 1, "policy_version": step, "problem_id": key,
                  "rollout_reuse": 1, "student_prompt_tokens": len(s_prompt), "teacher_prompt_tokens": len(t_prompt),
                  "rollout_tokens": n, "teacher_scoring_tokens": n, "effective_loss_tokens": n,
                  "rollout_terminated_eos": int(answer[0, -1]) == tokenizer.eos_token_id,
                  "reverse_kl": float(loss.detach()), "grad_norm": float(grad_norm.detach()),
                  "updated_parameter_tensors_in_fp32_master": updated_tensors,
                  "bf16_student_tensors_changed": bf16_changed_tensors,
                  "generation_seconds": generation_seconds, "teacher_scoring_seconds": teacher_seconds,
                  "peak_gpu_memory_mib": round(torch.cuda.max_memory_allocated(device) / 2**20, 1)}
        records.append(record)
        with (a.output_dir / "TRAIN_STEPS.jsonl").open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        writer.add_scalar("train/reverse_kl", record["reverse_kl"], step + 1)
        writer.add_scalar("train/effective_loss_tokens", n, step + 1)
        writer.flush()
        del student_logits, teacher_logits, student_ids, teacher_ids, loss
    evaluate(a.max_steps)
    teacher_unchanged = torch.equal(teacher_sample, next(teacher.parameters()).flatten()[:1024].detach().float())
    if not teacher_unchanged:
        raise RuntimeError("Frozen teacher sample changed")
    write_json(a.output_dir / "SMOKE_DONE.json", {"status": "engineering_smoke_completed", "steps": a.max_steps,
               "total_effective_loss_tokens": total_effective, "teacher_frozen_sample_unchanged": teacher_unchanged,
               "peak_gpu_memory_mib": round(torch.cuda.max_memory_allocated(device) / 2**20, 1),
               "formal_opd_completed": False, "source_accepted": False, "optimizer_matches_formal": False})
    writer.close()


if __name__ == "__main__":
    main()
