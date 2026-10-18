"""Compare rollout backends on frozen official student prompts; never trains.

prepare runs on CPU. GPU modes require a manually started one-GPU queue.
Backend/batch changes need not produce identical sampled text with the same seed.
This measures standalone inference, not weight-sync or colocated OPD overhead.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["prepare", "transformers", "vllm"], required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", type=Path, default=Path("/data0/users/lixiaoyang/models/Qwen3-1.7B"))
    p.add_argument("--source", type=Path, default=Path("/data0/users/lixiaoyang/COPSD-official-20261015"))
    p.add_argument("--data", type=Path)
    p.add_argument("--language", default="sw")
    p.add_argument("--count", type=int, default=4)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--adapter", type=Path)
    p.add_argument("--batch-size", type=int, choices=[1, 2, 4], default=1)
    a = p.parse_args()
    if a.count < 1:
        p.error("count must be positive")
    if a.backend != "prepare":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if os.environ.get("GPU_QUEUE_GPU_COUNT") != "1" or not visible.startswith("GPU-") or "," in visible:
            p.error("GPU modes require a one-GPU queue and an explicit GPU UUID")
        if not a.manifest:
            p.error("GPU modes require --manifest")
    elif not a.data:
        p.error("prepare requires --data")
    a.output_dir.mkdir(parents=True, exist_ok=False)
    from transformers import AutoTokenizer, GenerationConfig
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    generation = GenerationConfig.from_pretrained(a.model, local_files_only=True)
    eos = generation.eos_token_id
    eos = [eos] if isinstance(eos, int) else eos
    if not eos:
        raise ValueError("The frozen model must define EOS tokens")
    if a.backend == "prepare":
        sys.path.insert(0, str(a.source))
        from language_config import LANGUAGE_CONFIG
        from multilingual_data_collator import MultilingualSelfDistillationDataCollator
        if a.language.upper() not in LANGUAGE_CONFIG:
            raise ValueError("Unsupported language")
        collator = MultilingualSelfDistillationDataCollator(
            tokenizer, max_length=20000, reason_first=False, student_enable_thinking=True
        )
        rows = json.loads(a.data.read_text())
        if len(rows) < a.count:
            raise ValueError("Insufficient benchmark questions")
        prompts = []
        for row in rows[:a.count]:
            if not row.get("problem_" + a.language.lower()):
                raise ValueError("Missing target-language question; no English fallback")
            batch = collator([{**row, "target_lang": a.language.upper()}])
            ids = batch["student_prompts"][0].tolist()
            prompts.append({"id": row["problem_id"], "token_ids": ids,
                            "text": tokenizer.decode(ids, skip_special_tokens=False)})
        save(a.output_dir / "PROMPTS.json", {
            "model": str(a.model), "model_lock_sha256": digest(a.model / "MODEL_LOCK.json"),
            "data": str(a.data), "data_sha256": digest(a.data), "language": a.language,
            "collator_sha256": digest(a.source / "multilingual_data_collator.py"),
            "language_config_sha256": digest(a.source / "language_config.py"),
            "sampling": {"temperature": 1.1, "top_p": 0.95, "top_k": 20,
                         "max_tokens": 2048, "seed": 42, "eos_token_ids": eos},
            "prompts": prompts,
        })
        return
    manifest = json.loads(a.manifest.read_text())
    if digest(a.model / "MODEL_LOCK.json") != manifest["model_lock_sha256"]:
        raise ValueError("Model identity differs from prepared prompts")
    if eos != manifest["sampling"]["eos_token_ids"]:
        raise ValueError("EOS protocol differs from prepared prompts")
    import torch
    torch.cuda.set_device(0)
    telemetry, phase, stop_telemetry = [], ["load"], threading.Event()

    def sample_device():
        while not stop_telemetry.is_set():
            try:
                result = subprocess.run([
                    "nvidia-smi", "--id=" + os.environ["CUDA_VISIBLE_DEVICES"],
                    "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits",
                ], capture_output=True, text=True, timeout=3, check=True)
                memory, utilization = (float(v.strip()) for v in result.stdout.strip().split(","))
                telemetry.append({"phase": phase[0], "time": time.monotonic(),
                                  "memory_mib": memory, "utilization_percent": utilization})
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            stop_telemetry.wait(0.5)

    telemetry_thread = threading.Thread(target=sample_device, daemon=True)
    telemetry_thread.start()
    prompts = manifest["prompts"]
    cfg = manifest["sampling"]
    before = time.monotonic()
    adapter = str(a.adapter) if a.adapter else None
    if a.backend == "transformers":
        from transformers import AutoModelForCausalLM, set_seed
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            a.model, local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa", device_map={"": 0},
        )
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
        model.eval()

        def generate(batch, budget):
            inputs = tokenizer.pad({"input_ids": [r["token_ids"] for r in batch]},
                                   padding=True, return_tensors="pt").to("cuda:0")
            with torch.inference_mode():
                result = model.generate(**inputs, max_new_tokens=budget, do_sample=True,
                    temperature=cfg["temperature"], top_p=cfg["top_p"], top_k=cfg["top_k"],
                    eos_token_id=eos, pad_token_id=tokenizer.pad_token_id, use_cache=True)
            output = []
            for row in result[:, inputs["input_ids"].shape[1]:].tolist():
                stop = next((i + 1 for i, token in enumerate(row) if token in eos), len(row))
                tokens = row[:stop]
                output.append((tokens, "eos" if any(t in eos for t in tokens) else "length"))
            return output
    else:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        model = LLM(model=str(a.model), tokenizer=str(a.model), dtype="bfloat16", seed=42,
                    generation_config="vllm", gpu_memory_utilization=0.75,
                    max_model_len=max(len(r["token_ids"]) for r in prompts) + cfg["max_tokens"],
                    max_num_seqs=a.batch_size, enable_lora=bool(adapter), max_lora_rank=64)
        request = LoRARequest("student_snapshot", 1, adapter) if adapter else None

        def generate(batch, budget):
            sampling = SamplingParams(n=1, temperature=cfg["temperature"], top_p=cfg["top_p"],
                top_k=cfg["top_k"], max_tokens=budget, seed=cfg["seed"], stop_token_ids=eos)
            outputs = model.generate([{"prompt_token_ids": r["token_ids"]} for r in batch],
                                     sampling, lora_request=request, use_tqdm=False)
            return [(list(row.outputs[0].token_ids), row.outputs[0].finish_reason) for row in outputs]
    torch.cuda.synchronize()
    load_seconds = time.monotonic() - before
    phase[0] = "warmup"
    before = time.monotonic()
    generate(prompts[:1], 64)  # Warmup only; excluded from measured 2048-token budget.
    torch.cuda.synchronize()
    warmup_seconds = time.monotonic() - before
    if a.backend == "transformers":
        set_seed(cfg["seed"])
    results = []
    torch.cuda.reset_peak_memory_stats()
    phase[0] = "generation"
    before = time.monotonic()
    for start in range(0, len(prompts), a.batch_size):
        batch = prompts[start:start+a.batch_size]
        outputs = generate(batch, cfg["max_tokens"])
        for row, (tokens, finish_reason) in zip(batch, outputs):
            results.append({"id": row["id"], "tokens": len(tokens), "token_ids": tokens,
                "contains_eos": any(t in eos for t in tokens), "finish_reason": finish_reason,
                "text": tokenizer.decode(tokens, skip_special_tokens=False)})
    torch.cuda.synchronize()
    seconds = time.monotonic() - before
    stop_telemetry.set()
    telemetry_thread.join(timeout=4)
    measured = [r for r in telemetry if r["phase"] == "generation"]
    save(a.output_dir / "GPU_TELEMETRY.json", telemetry)
    save(a.output_dir / "RESULTS.json", {
        "scope": "standalone rollout only; no teacher, loss, backward, or weight-sync timing",
        "backend": a.backend, "batch_size": a.batch_size, "adapter": adapter,
        "adapter_sha256": digest(a.adapter / "adapter_model.safetensors") if adapter else None,
        "manifest_sha256": digest(a.manifest), "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "sampling": cfg, "load_seconds": load_seconds, "warmup_seconds": warmup_seconds,
        "python": sys.executable,
        "packages": {name: importlib.metadata.version(name) for name in
                     (["torch", "transformers", "peft", "vllm"] if a.backend == "vllm"
                      else ["torch", "transformers", "peft"])},
        "generation_seconds": seconds, "actual_tokens": sum(r["tokens"] for r in results),
        "actual_tokens_per_second": sum(r["tokens"] for r in results) / seconds,
        "torch_process_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "observed_device_peak_generation_mib": max((r["memory_mib"] for r in measured), default=None),
        "observed_device_mean_generation_utilization_percent":
            sum(r["utilization_percent"] for r in measured) / len(measured) if measured else None,
        "telemetry_note": "Device-wide samples every 0.5s, including vLLM workers; observed peak can miss transient spikes.",
        "memory_note": "Torch peak excludes separate vLLM worker allocations; use external GPU telemetry for vLLM.",
        "items": results,
    })


if __name__ == "__main__":
    main()
