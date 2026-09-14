#!/usr/bin/env python3
"""Modern-PEFT compatible O-LoRA training for continual CPT.

Previous LoRA deltas are merged into the incoming full checkpoint.  Their A
subspaces are retained in a compact state file and constrain the new adapter
with the official O-LoRA L1 orthogonality term.
"""

import argparse
import json
import os
import random
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)


def lora_modules(model):
    result = {}
    for name, module in model.named_modules():
        lora_a = getattr(module, "lora_A", None)
        if lora_a is not None and "default" in lora_a:
            result[name] = module
    return result


def load_old_state(path):
    if not path:
        return {}
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "olora_a_state_v1":
        raise ValueError(f"unexpected O-LoRA state format: {path}")
    return payload["a_bases"]


class OrthogonalLoraTrainer(Trainer):
    def __init__(self, *args, old_a_bases=None, orth_lambda=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.old_a_bases = old_a_bases or {}
        self.orth_lambda = orth_lambda
        self._olora_modules = lora_modules(self.model)
        missing = sorted(set(self.old_a_bases) - set(self._olora_modules))
        if missing:
            raise KeyError(f"old A-state modules absent from model: {missing[:5]}")

    def log(self, logs, start_time=None):
        # Keep the paper-compatible summed regularizer, but expose its scale
        # separately so the Trainer's total loss is not mistaken for LM loss.
        if "loss" in logs and hasattr(self, "_last_task_loss"):
            logs = dict(logs)
            logs["task_loss"] = self._last_task_loss
            logs["orth_loss_unscaled"] = self._last_orth_loss
        return super().log(logs, start_time=start_time)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        task_loss = outputs.loss
        orth_loss = task_loss.new_zeros(())
        if self.old_a_bases:
            for name, old_a in self.old_a_bases.items():
                new_a = self._olora_modules[name].lora_A["default"].weight
                old_a = old_a.to(device=new_a.device, dtype=new_a.dtype)
                orth_loss = orth_loss + torch.abs(old_a @ new_a.T).sum()
        loss = task_loss + self.orth_lambda * orth_loss
        self._last_orth_loss = float(orth_loss.detach())
        self._last_task_loss = float(task_loss.detach())
        return (loss, outputs) if return_outputs else loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--train-dataset", required=True)
    parser.add_argument("--eval-dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prior-state")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--orth-lambda", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for path in (args.model, args.tokenizer, args.train_dataset, args.eval_dataset):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train = load_from_disk(args.train_dataset).shuffle(seed=args.seed)
    valid = load_from_disk(args.eval_dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    )
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    model = get_peft_model(model, config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    old_a = load_old_state(args.prior_state)
    train_args = TrainingArguments(
        output_dir=str(output),
        logging_dir=str(output / "logs"),
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="no",
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant",
        warmup_steps=0,
        weight_decay=0.0,
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        bf16=True,
        tf32=False,
        gradient_checkpointing=True,
        logging_steps=1,
        report_to=["tensorboard"],
        remove_unused_columns=True,
        seed=args.seed,
        data_seed=args.seed,
        prediction_loss_only=True,
    )
    trainer = OrthogonalLoraTrainer(
        model=model,
        args=train_args,
        train_dataset=train,
        eval_dataset=valid,
        data_collator=default_data_collator,
        old_a_bases=old_a,
        orth_lambda=args.orth_lambda,
    )
    trainer.train()

    combined = {name: value.cpu() for name, value in old_a.items()}
    for name, module in lora_modules(model).items():
        current = module.lora_A["default"].weight.detach().float().cpu()
        combined[name] = (
            current if name not in combined
            else torch.cat((combined[name].float(), current), dim=0)
        )
    state_path = output / "olora_state.pt"
    torch.save(
        {
            "format": "olora_a_state_v1",
            "rank_per_task": args.rank,
            "target_modules": ["q_proj", "v_proj"],
            "a_bases": combined,
        },
        state_path,
    )

    merged = model.merge_and_unload()
    merged.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    metadata = {
        "format": "olora_compatible_cpt_v1",
        "previous_model": args.model,
        "prior_state": args.prior_state,
        "rank": args.rank,
        "alpha": args.alpha,
        "dropout": args.dropout,
        "orth_lambda": args.orth_lambda,
        "learning_rate": args.learning_rate,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "implementation_note": (
            "Previous LoRA deltas are merged into the full checkpoint; "
            "historical A bases are retained for the official L1 orthogonality loss."
        ),
    }
    (output / "OLORA_METADATA.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    (output / "TRAINING_DONE").touch()
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
