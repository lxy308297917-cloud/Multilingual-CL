#!/usr/bin/env python3
"""Run the aligned PLND downstream suite in the locked evaluation environment."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
TASKS = PROJECT / "evaluation" / "src"


def load_subjects() -> tuple[str, ...]:
    spec = importlib.util.spec_from_file_location("plnd_mmlu_local", TASKS / "mmlu_local.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return tuple(module.SUBJECTS)


def run(command: list[str], marker: Path, environment: dict) -> None:
    if marker.is_file():
        print(f"[SKIP] {marker}")
        return
    marker.parent.mkdir(parents=True, exist_ok=True)
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True, env=environment)
    marker.touch()


def lighteval(
    config, model: Path, task: str, custom: str, output: Path, batch: int,
    environment: dict, max_samples: int | None = None,
):
    command = [
        config["lighteval_executable"], "accelerate",
        f"model_name={model},batch_size={batch},dtype=bfloat16,override_chat_template=true",
        task, "--custom-tasks", str(TASKS / custom), "--save-details",
        "--output-dir", str(output),
    ]
    if max_samples is not None:
        command.extend(["--max-samples", str(max_samples)])
    run(command, output / "DONE", environment)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    environment = os.environ.copy()
    environment.update({
        "HF_HOME": config["hf_cache"],
        "HF_HUB_CACHE": config["hf_cache"],
        "HF_DATASETS_CACHE": str(Path(config["hf_cache"]) / "datasets"),
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "1",
    })
    environment.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    model, output = args.model.resolve(), args.output.resolve()
    jobs = [
        ("mt:en2ig|3", "mt.py", "target/mt_en_to_ig", 16),
        ("mt:ig2en|3", "mt.py", "target/mt_ig_to_en", 16),
        ("sum:ig|0", "sum.py", "target/sum_ig", 16),
        ("sum:en|0", "sum.py", "english/sum_en", 16),
        ("belebele_ibo_Latn_mcf|3", "belebele.py", "target/belebele_ig", 8),
        ("belebele_eng_Latn_mcf|3", "belebele.py", "english/belebele_en", 8),
    ]
    sample_limit = 1 if args.smoke else None
    for task, custom, directory, batch in jobs:
        lighteval(
            config, model, task, custom, output / directory, batch, environment,
            max_samples=sample_limit,
        )

    subjects = load_subjects()
    if args.smoke:
        subjects = subjects[:1]
    gmmlu = ",".join(f"gmmlu_ibo_mcf:{subject}|5" for subject in subjects)
    mmlu = ",".join(f"mmlu_local:{subject}|5" for subject in subjects)
    lighteval(
        config, model, gmmlu, "gmmlu.py", output / "target/gmmlu_ig", 4,
        environment, max_samples=sample_limit,
    )
    lighteval(
        config, model, mmlu, "mmlu_local.py", output / "english/mmlu_en", 8,
        environment, max_samples=sample_limit,
    )

    ifeval = [
        config["evaluation_python"], "-u", str(TASKS / "ifeval.py"),
        "--model_name_or_path", str(model), "--output_dir", str(output / "general/ifeval"),
        "--dataset_jsonl",
        "/root/.cache/huggingface/hub/datasets--google--IFEval/snapshots/966cd89545d6b6acfd7638bc708b98261ca58e84/ifeval_input_data.jsonl",
        "--official_files_dir", str(TASKS / "utils"), "--apply_chat_template", "--batch_size", "12",
    ]
    if args.smoke:
        ifeval.extend(["--max_samples", "1"])
    run(ifeval, output / "general/ifeval/DONE", environment)
    gsm8k = [
        config["evaluation_python"], "-u", str(TASKS / "gsm8k.py"),
        "--model_name_or_path", str(model), "--output_dir", str(output / "general/gsm8k"),
        "--test_arrow",
        "/root/.cache/huggingface/datasets/openai___gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866/gsm8k-test.arrow",
        "--mode", "gsm8k_cot", "--apply_chat_template", "--max_new_tokens", "256", "--batch_size", "12",
    ]
    if args.smoke:
        gsm8k.extend(["--max_samples", "1"])
    run(gsm8k, output / "general/gsm8k/DONE", environment)


if __name__ == "__main__":
    main()
