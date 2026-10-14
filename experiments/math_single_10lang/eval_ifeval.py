"""Fixed English IFEval retention check using the repository's official-style scorer."""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import dump_json, sha


def load_scorer(path):
    spec = importlib.util.spec_from_file_location("math_single_ifeval_scorer", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _, _, registry = module.load_official_ifeval_modules(path.parent / "utils")
    return module, registry


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--items", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, help="Pinned Base tokenizer; defaults to model directory")
    p.add_argument("--scorer-code", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise FileExistsError(a.output_dir)
    rows = [json.loads(s) for s in a.items.read_text().splitlines() if s]
    if len(rows) != 541 or len({r["key"] for r in rows}) != 541:
        raise ValueError("Expected 541 unique pinned IFEval prompts")
    if a.limit:
        rows = rows[:a.limit]
    scorer, registry = load_scorer(a.scorer_code)
    a.output_dir.mkdir(parents=True)
    score_files = [a.scorer_code, *(a.scorer_code.parent / "utils" / name for name in ("instructions.py", "instructions_registry.py", "instructions_util.py"))]
    dump_json(a.output_dir / "EVAL_CONFIG.json", {"model": str(a.model), "tokenizer": str(a.tokenizer or a.model), "items": str(a.items), "items_sha256": sha(a.items.read_text()), "scorer_code": str(a.scorer_code), "scorer_file_sha256": {str(file): sha(file.read_text()) for file in score_files}, "count": len(rows), "generation": {"enable_thinking": False, "do_sample": False, "max_new_tokens": 1280}})
    tokenizer = AutoTokenizer.from_pretrained(str(a.tokenizer or a.model), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(a.model), local_files_only=True, torch_dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa").eval()
    outcomes = []
    with (a.output_dir / "predictions.jsonl").open("w") as f:
        for i, row in enumerate(rows, 1):
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": row["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            torch.cuda.synchronize()
            started = time.monotonic()
            with torch.inference_mode():
                output = model.generate(**inputs, do_sample=False, max_new_tokens=1280, pad_token_id=tokenizer.eos_token_id)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - started
            new_ids = output[0, inputs["input_ids"].shape[-1]:]
            raw = tokenizer.decode(new_ids, skip_special_tokens=False)
            response = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            inp = scorer.InputExample(key=row["key"], instruction_id_list=row["instruction_id_list"], prompt=row["prompt"], kwargs=row["kwargs"])
            strict = scorer.test_instruction_following_strict(inp, response, registry)
            loose = scorer.test_instruction_following_loose(inp, response, registry)
            result = {"key": row["key"], "prompt": row["prompt"], "prompt_tokens": int(inputs["input_ids"].shape[-1]), "instruction_id_list": row["instruction_id_list"], "kwargs": row["kwargs"], "response": response, "raw_output": raw, "generated_tokens": len(new_ids), "generation_seconds": elapsed, "truncated": len(new_ids) >= 1280, "prompt_level_strict_acc": strict.follow_all_instructions, "inst_level_strict_acc": strict.follow_instruction_list, "prompt_level_loose_acc": loose.follow_all_instructions, "inst_level_loose_acc": loose.follow_instruction_list}
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            outcomes.append(result)
            print(f"{i}/{len(rows)} key={row['key']} strict={strict.follow_all_instructions}", flush=True)
    dump_json(a.output_dir / "DONE.json", {"count": len(outcomes), "prompt_level_strict_acc": sum(x["prompt_level_strict_acc"] for x in outcomes)/len(outcomes), "inst_level_strict_acc": scorer.agg_inst_level_acc([x["inst_level_strict_acc"] for x in outcomes]), "prompt_level_loose_acc": sum(x["prompt_level_loose_acc"] for x in outcomes)/len(outcomes), "inst_level_loose_acc": scorer.agg_inst_level_acc([x["inst_level_loose_acc"] for x in outcomes]), "prompt_tokens": sum(x["prompt_tokens"] for x in outcomes), "generated_tokens": sum(x["generated_tokens"] for x in outcomes), "generation_seconds": sum(x["generation_seconds"] for x in outcomes)})


if __name__ == "__main__":
    main()
