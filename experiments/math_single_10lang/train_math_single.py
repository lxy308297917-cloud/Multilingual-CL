"""Qwen3 full-parameter assistant-only SFT with fixed 2048 token sequences.

Formal runs require independently accepted labels for every shared problem ID.
Use torchrun with four visible GPUs for FSDP; each language starts from Base.
"""
import argparse
import json
import math
import os
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

from common import CONFIG, messages, dump_json, sha


def prepare_record(row, tokenizer, language):
    question = row["question_target"]
    prompt_ids = tokenizer.apply_chat_template(messages(question, language), tokenize=True, add_generation_prompt=True, enable_thinking=True)
    prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)
    if row["question_en"] in prompt_text and row["question_en"] != question:
        raise ValueError("English question leaked into student prompt")
    if row["gold_answer"] in prompt_text and row["gold_answer"] not in question:
        raise ValueError("Gold answer leaked into student prompt")
    completion = row["assistant_completion"]
    if not completion.startswith("<think>") or "</think>" not in completion or "\\boxed{" not in completion:
        raise ValueError("Invalid completion structure")
    if prompt_text.rstrip().endswith("<think>"):
        completion = completion[len("<think>"):]
    completion_ids = tokenizer(completion, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
    total = len(prompt_ids) + len(completion_ids)
    if total > CONFIG["max_length"]:
        raise ValueError(f"Overlength complete sequence: {total}")
    padding = CONFIG["max_length"] - total
    return {"input_ids": prompt_ids + completion_ids + [tokenizer.pad_token_id] * padding,
            "attention_mask": [1] * total + [0] * padding,
            "labels": [-100] * len(prompt_ids) + completion_ids + [-100] * padding,
            "input_tokens": total, "supervised_tokens": len(completion_ids), "padding_tokens": padding}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--language", choices=list(CONFIG["languages"]), required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--labels-dir", type=Path, required=True)
    p.add_argument("--splits", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke-examples", type=int, default=64)
    p.add_argument("--smoke-dev-examples", type=int, default=16)
    a = p.parse_args()
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0 and a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {a.output_dir}")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if not a.smoke and world_size != 4:
        raise ValueError("Formal 4B FFT requires four GPU FSDP workers on this 24GB/GPU machine")
    set_seed(a.seed)
    splits = json.loads(a.splits.read_text())
    ids = splits["train"] + splits["dev_tune"]
    if a.smoke:
        def available_for_smoke(key):
            file = a.labels_dir / a.language / f"{key}.json"
            return file.exists() and json.loads(file.read_text()).get("quality_status") in ("pending_review", "accepted")
        train_ids = [key for key in splits["train"] if available_for_smoke(key)][:a.smoke_examples]
        dev_ids = [key for key in splits["dev_tune"] if available_for_smoke(key)][:a.smoke_dev_examples]
        if len(train_ids) != a.smoke_examples or len(dev_ids) != a.smoke_dev_examples:
            raise ValueError(f"Insufficient valid smoke labels: train={len(train_ids)}/{a.smoke_examples}, dev={len(dev_ids)}/{a.smoke_dev_examples}")
        ids = train_ids + dev_ids
    tokenizer = AutoTokenizer.from_pretrained(str(a.model), local_files_only=True)
    rows = {}
    for key in ids:
        file = a.labels_dir / a.language / f"{key}.json"
        if not file.exists():
            raise FileNotFoundError(file)
        row = json.loads(file.read_text())
        if row.get("problem_id") != key or row.get("language") != a.language:
            raise ValueError(f"Label identity mismatch: {file}")
        if row.get("quality_status") != "accepted" and not (a.smoke and row.get("quality_status") == "pending_review"):
            raise ValueError(f"Label not accepted: {file}: {row.get('quality_status')}")
        rows[key] = prepare_record(row, tokenizer, a.language)
    train = Dataset.from_list([rows[k] for k in ids if k in splits["train"]])
    dev = Dataset.from_list([rows[k] for k in ids if k in splits["dev_tune"]])
    train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    dev.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    if world_size == 4:
        accum = CONFIG["global_batch"] // (CONFIG["micro_batch"] * world_size)
    else:
        accum = CONFIG["global_batch"] // CONFIG["micro_batch"]
    if CONFIG["global_batch"] != CONFIG["micro_batch"] * world_size * accum:
        raise ValueError("Effective batch is not exact")
    steps_half_epoch = math.ceil(len(train) / CONFIG["global_batch"] / 2)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        dump_json(a.output_dir / "RUN_CONFIG.json", {"config": CONFIG, "language": a.language, "train_seed": a.seed, "data_seed": a.seed, "model_path": str(a.model), "model_revision": CONFIG["model_revision"], "smoke": a.smoke, "world_size": world_size, "gradient_accumulation_steps": accum, "train_ids": [k for k in ids if k in splits["train"]], "dev_ids": [k for k in ids if k in splits["dev_tune"]], "train_count": len(train), "dev_count": len(dev), "supervised_tokens_train": sum(rows[k]["supervised_tokens"] for k in ids if k in splits["train"]), "input_tokens_train": sum(rows[k]["input_tokens"] for k in ids if k in splits["train"]), "padding_tokens_train": sum(rows[k]["padding_tokens"] for k in ids if k in splits["train"]), "split_sha256": sha(a.splits.read_text()), "label_status_required": "pending_review allowed only for smoke" if a.smoke else "accepted"})
        with (a.output_dir / "PREPARED_SAMPLES.jsonl").open("w") as sample_file:
            for key in ids[:5]:
                row = rows[key]
                active = row["input_ids"][:row["input_tokens"]]
                first_label = next(i for i, value in enumerate(row["labels"]) if value != -100)
                sample_file.write(json.dumps({"problem_id": key, "prompt_token_ids": active[:first_label], "completion_token_ids": active[first_label:], "decoded_full_sequence": tokenizer.decode(active, skip_special_tokens=False), "supervised_tokens": row["supervised_tokens"], "input_tokens": row["input_tokens"]}, ensure_ascii=False) + "\n")
    model = AutoModelForCausalLM.from_pretrained(str(a.model), local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    args = TrainingArguments(output_dir=str(a.output_dir), overwrite_output_dir=False,
        per_device_train_batch_size=CONFIG["micro_batch"], per_device_eval_batch_size=1,
        gradient_accumulation_steps=accum, num_train_epochs=CONFIG["epochs"], max_steps=20 if a.smoke else -1,
        learning_rate=CONFIG["learning_rate"], optim="adamw_torch", adam_beta1=.9, adam_beta2=.999,
        adam_epsilon=1e-8, weight_decay=CONFIG["weight_decay"], lr_scheduler_type="cosine",
        warmup_ratio=CONFIG["warmup_ratio"], max_grad_norm=1.0, bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, logging_steps=5, logging_first_step=True,
        eval_strategy="steps", eval_steps=10 if a.smoke else steps_half_epoch,
        save_strategy="steps", save_steps=20 if a.smoke else steps_half_epoch,
        eval_on_start=True, report_to=["tensorboard"], save_safetensors=True,
        save_only_model=True, seed=a.seed, data_seed=a.seed,
        fsdp="full_shard auto_wrap" if world_size == 4 else "",
        fsdp_config={"transformer_layer_cls_to_wrap": ["Qwen3DecoderLayer"]} if world_size == 4 else {},
        remove_unused_columns=False, dataloader_num_workers=0)
    trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=dev, processing_class=tokenizer)
    trainer.train()
    if not a.smoke:
        trainer.save_model(str(a.output_dir / "final"))
    if rank == 0:
        dump_json(a.output_dir / "TRAINING_DONE.json", {"language": a.language, "global_step": trainer.state.global_step, "model_revision": CONFIG["model_revision"], "smoke": a.smoke, "final_checkpoint": "checkpoint-20" if a.smoke else "final", "note": "save_only_model checkpoints contain weights only; interrupted run must restart from Base"})


if __name__ == "__main__":
    main()
