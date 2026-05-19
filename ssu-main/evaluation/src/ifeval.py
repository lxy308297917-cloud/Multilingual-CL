#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IFEval on NPU, aligned as closely as possible with the official lm-eval task.

Official alignment:
- dataset_path: google/IFEval
- test_split: train
- num_fewshot: 0
- doc_to_text: prompt
- metrics:
    - prompt_level_strict_acc
    - inst_level_strict_acc
    - prompt_level_loose_acc
    - inst_level_loose_acc

This script reuses the official instruction-checking code from:
- instructions.py
- instructions_registry.py
- instructions_util.py

Save this file as:
    evaluation/src/ifeval.py

Place these sibling files in the same directory:
    instructions.py
    instructions_registry.py
    instructions_util.py
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# =========================
# Dynamic import of official IFEval files
# =========================

def _register_fake_package_hierarchy() -> None:
    """
    Create fake package entries so that the official files can keep imports like:
        from lm_eval.tasks.ifeval import instructions_util
    """
    if "lm_eval" not in sys.modules:
        sys.modules["lm_eval"] = types.ModuleType("lm_eval")
    if "lm_eval.tasks" not in sys.modules:
        sys.modules["lm_eval.tasks"] = types.ModuleType("lm_eval.tasks")
    if "lm_eval.tasks.ifeval" not in sys.modules:
        sys.modules["lm_eval.tasks.ifeval"] = types.ModuleType("lm_eval.tasks.ifeval")


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_official_ifeval_modules(base_dir: Path):
    _register_fake_package_hierarchy()

    util_path = base_dir / "instructions_util.py"
    instr_path = base_dir / "instructions.py"
    registry_path = base_dir / "instructions_registry.py"

    if not util_path.exists():
        raise FileNotFoundError(f"Missing file: {util_path}")
    if not instr_path.exists():
        raise FileNotFoundError(f"Missing file: {instr_path}")
    if not registry_path.exists():
        raise FileNotFoundError(f"Missing file: {registry_path}")

    instructions_util = _load_module(
        "lm_eval.tasks.ifeval.instructions_util", util_path
    )
    instructions = _load_module(
        "lm_eval.tasks.ifeval.instructions", instr_path
    )
    instructions_registry = _load_module(
        "lm_eval.tasks.ifeval.instructions_registry", registry_path
    )
    return instructions_util, instructions, instructions_registry


# =========================
# Official-style scoring logic
# =========================

InstructionArgs = Optional[Dict[str, Optional[Union[str, int]]]]


@dataclasses.dataclass
class InputExample:
    key: int
    instruction_id_list: List[str]
    prompt: str
    kwargs: List[InstructionArgs]


@dataclasses.dataclass
class OutputExample:
    instruction_id_list: List[str]
    prompt: str
    response: str
    follow_all_instructions: bool
    follow_instruction_list: List[bool]


def test_instruction_following_strict(
    inp: InputExample,
    response: str,
    instructions_registry_module,
) -> OutputExample:
    instruction_list = inp.instruction_id_list
    is_following_list: List[bool] = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry_module.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        kwargs = {k: v for k, v in inp.kwargs[index].items() if v is not None}
        instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp.prompt)

        if response.strip() and instruction.check_following(response):
            is_following_list.append(True)
        else:
            is_following_list.append(False)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def test_instruction_following_loose(
    inp: InputExample,
    response: str,
    instructions_registry_module,
) -> OutputExample:
    r = response.split("\n")
    response_remove_first = "\n".join(r[1:]).strip()
    response_remove_last = "\n".join(r[:-1]).strip()
    response_remove_both = "\n".join(r[1:-1]).strip()

    revised_response = response.replace("*", "")
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")

    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]

    instruction_list = inp.instruction_id_list
    is_following_list: List[bool] = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry_module.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        kwargs = {k: v for k, v in inp.kwargs[index].items() if v is not None}
        instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp.prompt)

        is_following = False
        for candidate in all_responses:
            if candidate.strip() and instruction.check_following(candidate):
                is_following = True
                break

        is_following_list.append(is_following)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def agg_inst_level_acc(items: List[List[bool]]) -> float:
    flat_items = [item for sublist in items for item in sublist]
    return sum(flat_items) / len(flat_items) if flat_items else 0.0


# =========================
# Model + generation
# =========================

def pick_device() -> str:
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def pick_dtype(device: str):
    if device == "cpu":
        return torch.float32
    # 对齐你当前环境，NPU 这里优先 bfloat16
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
    max_new_tokens: int = 1280,
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


# =========================
# Main evaluation
# =========================

def parse_args():
    parser = argparse.ArgumentParser(description="IFEval evaluation on NPU")

    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=None)

    parser.add_argument(
        "--official_files_dir",
        type=str,
        default=None,
        help="Directory containing instructions.py / instructions_registry.py / instructions_util.py. "
             "Default: same directory as this script.",
    )

    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Apply tokenizer chat template before generation.",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1280,
        help="Aligned with official ifeval.yaml max_gen_toks=1280.",
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="0 means use all samples.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_dir = Path(args.official_files_dir) if args.official_files_dir else (Path(__file__).resolve().parent / "utils")
    _, _, instructions_registry_module = load_official_ifeval_modules(base_dir)

    print("=" * 48)
    print("✅ 开始评测 IFEval")
    print(f"model_name_or_path : {args.model_name_or_path}")
    print(f"output_dir         : {output_dir}")
    print(f"official_files_dir : {base_dir}")
    print(f"apply_chat_template: {args.apply_chat_template}")
    print(f"max_new_tokens     : {args.max_new_tokens}")
    print(f"max_samples        : {args.max_samples}")
    print("=" * 48)

    model, tokenizer, device = load_model_and_tokenizer(
        args.model_name_or_path,
        args.cache_dir,
    )
    print(f"🖥️ device = {device}")

    # Official config:
    # dataset_path: google/IFEval
    # test_split: train
    ds = load_dataset(
        "google/IFEval",
        cache_dir=args.cache_dir,
        split="train",
    )

    if args.max_samples and args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    print(f"📥 载入样本数 = {len(ds)}")

    details_path = output_dir / "predictions.jsonl"
    prompt_level_strict_list: List[bool] = []
    prompt_level_loose_list: List[bool] = []
    inst_level_strict_items: List[List[bool]] = []
    inst_level_loose_items: List[List[bool]] = []

    with details_path.open("w", encoding="utf-8") as fout:
        for idx, doc in enumerate(ds, start=1):
            prompt = doc["prompt"]

            response = generate_one(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompt=prompt,
                apply_chat_template=args.apply_chat_template,
                max_new_tokens=args.max_new_tokens,
            )

            inp = InputExample(
                key=doc["key"],
                instruction_id_list=doc["instruction_id_list"],
                prompt=doc["prompt"],
                kwargs=doc["kwargs"],
            )

            out_strict = test_instruction_following_strict(
                inp, response, instructions_registry_module
            )
            out_loose = test_instruction_following_loose(
                inp, response, instructions_registry_module
            )

            prompt_level_strict_list.append(out_strict.follow_all_instructions)
            prompt_level_loose_list.append(out_loose.follow_all_instructions)
            inst_level_strict_items.append(out_strict.follow_instruction_list)
            inst_level_loose_items.append(out_loose.follow_instruction_list)

            fout.write(
                json.dumps(
                    {
                        "key": doc["key"],
                        "prompt": doc["prompt"],
                        "instruction_id_list": doc["instruction_id_list"],
                        "kwargs": doc["kwargs"],
                        "response": response,
                        "prompt_level_strict_acc": out_strict.follow_all_instructions,
                        "inst_level_strict_acc": out_strict.follow_instruction_list,
                        "prompt_level_loose_acc": out_loose.follow_all_instructions,
                        "inst_level_loose_acc": out_loose.follow_instruction_list,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            if idx % 20 == 0 or idx == len(ds):
                cur_prompt_strict = sum(prompt_level_strict_list) / len(prompt_level_strict_list)
                cur_prompt_loose = sum(prompt_level_loose_list) / len(prompt_level_loose_list)
                print(
                    f"[{idx}/{len(ds)}] "
                    f"strict_prompt={cur_prompt_strict:.4f} "
                    f"loose_prompt={cur_prompt_loose:.4f}"
                )

    summary = {
        "task": "ifeval",
        "model_name_or_path": args.model_name_or_path,
        "num_samples": len(ds),
        "apply_chat_template": args.apply_chat_template,
        "prompt_level_strict_acc": sum(prompt_level_strict_list) / len(prompt_level_strict_list),
        "inst_level_strict_acc": agg_inst_level_acc(inst_level_strict_items),
        "prompt_level_loose_acc": sum(prompt_level_loose_list) / len(prompt_level_loose_list),
        "inst_level_loose_acc": agg_inst_level_acc(inst_level_loose_items),
        "details_path": str(details_path),
    }

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 48)
    print("🎉 IFEval 评测完成")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary = {summary_path}")
    print(f"details = {details_path}")
    print("=" * 48)


if __name__ == "__main__":
    main()