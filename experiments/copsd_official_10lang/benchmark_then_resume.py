"""One authorized four-GPU handoff: bounded rollout benchmarks, then TH resume.

The queue selects four UUIDs. Benchmark children use subsets of those UUIDs;
the parent's selection is unchanged and all four are passed to resumed training.
No GPU outside this allocation is selected. Failed benchmarks are not retried.
"""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import xml.etree.ElementTree as ET


def save(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def occupied(uuids):
    xml = subprocess.run(["nvidia-smi", "-q", "-x"], check=True,
                         capture_output=True, text=True, timeout=10).stdout
    busy, seen = [], set()
    for gpu in ET.fromstring(xml).findall("gpu"):
        uuid = gpu.findtext("uuid")
        if uuid not in uuids:
            continue
        seen.add(uuid)
        processes = gpu.find("processes")
        if processes is None:
            raise RuntimeError("GPU process information unavailable")
        for proc in processes.findall("process_info"):
            memory = proc.findtext("used_memory") or ""
            ignored = (proc.findtext("process_name") == "/usr/lib/xorg/Xorg"
                       and proc.findtext("type") == "G" and memory.endswith(" MiB")
                       and memory.split()[0].isdigit() and int(memory.split()[0]) <= 64)
            if not ignored:
                busy.append({"gpu_uuid": uuid, "pid": proc.findtext("pid")})
        memory = int(gpu.findtext("fb_memory_usage/used").split()[0])
        utilization = int(gpu.findtext("utilization/gpu_util").split()[0])
        if memory > 512 or utilization > 5:
            busy.append({"gpu_uuid": uuid, "memory_mib": memory,
                         "utilization_percent": utilization})
    if seen != set(uuids):
        raise RuntimeError("Some queue-assigned GPUs are missing from the snapshot")
    return busy


def main():
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--handoff-run-id", required=True)
    parser.add_argument("--resume-run-id", required=True)
    args = parser.parse_args()
    for name in [args.handoff_run_id, args.resume_run_id]:
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
            parser.error("Invalid run ID")
    uuids = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if (os.environ.get("GPU_QUEUE_GPU_COUNT") != "4" or len(set(uuids)) != 4
            or any(not uuid.startswith("GPU-") for uuid in uuids)):
        parser.error("Requires four UUIDs selected by the queue")
    root = Path("/data0/users/lixiaoyang")
    scripts = Path(__file__).resolve().parent
    checkpoint = args.checkpoint.resolve()
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    if not 0 < state["global_step"] < 100:
        raise ValueError("Not a resumable intermediate checkpoint")
    adapter = root / "models/checkpoint/copsd_official_17b_single_v1/sw_17b_project2900_seed42_20261015_v1/checkpoint-20"
    manifest = root / "logs/copsd_official_10lang/rollout_benchmark/sw_train4_prompts_20261018/PROMPTS.json"
    output = root / "logs/copsd_official_10lang/handoff" / args.handoff_run_id
    output.mkdir(parents=True, exist_ok=False)
    save(output / "MANIFEST.json", {
        "created_at": datetime.datetime.now().astimezone().isoformat(),
        "parent_checkpoint": str(checkpoint), "parent_global_step": state["global_step"],
        "checkpoint_adapter_sha256": digest(checkpoint / "adapter_model.safetensors"),
        "benchmark_adapter": str(adapter), "benchmark_adapter_sha256": digest(adapter / "adapter_model.safetensors"),
        "prompts_sha256": digest(manifest), "queue_gpu_uuids": uuids,
        "resume_run_id": args.resume_run_id, "benchmark_time_limit_seconds": 1800,
        "source_sha256": {name: digest(scripts / name) for name in [
            "benchmark_then_resume.py", "run_benchmark_then_resume.sh",
            "benchmark_rollout.py", "run_rollout_benchmark.sh",
            "run_four_gpu_17b_resume_v2.sh", "resume_torch28_entry.py"]},
        "versions": {p: importlib.metadata.version(p) for p in
                     ["torch", "transformers", "trl", "peft", "accelerate", "datasets", "numpy", "vllm"]},
        "scope": "Standalone rollout benchmark, not end-to-end OPD throughput; resumed runtime differs from original Torch2.5 environment",
    })
    if occupied(uuids):
        raise RuntimeError("A selected GPU acquired a compute process before benchmarking")
    deadline = time.monotonic() + 1800
    results = []

    def stop_group(process):
        # Only process groups created and tracked by this program are signalled.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=15)

    def run_cases(cases):
        running = []
        try:
            for backend, batch, index, label in cases:
                run_id = args.handoff_run_id + "_" + label
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuids[index], GPU_QUEUE_GPU_COUNT="1")
                command = ["bash", str(scripts / "run_rollout_benchmark.sh"),
                           backend, str(batch), run_id, str(adapter)]
                handle = (output / (label + ".log")).open("x")
                process = subprocess.Popen(command, env=env, stdout=handle,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                item = {"backend": backend, "batch_size": batch, "label": label,
                        "gpu_uuid": uuids[index], "pid": process.pid, "run_id": run_id}
                running.append((process, handle, item))
                print(json.dumps({"benchmark_started": item}), flush=True)
            for process, handle, item in running:
                try:
                    code = process.wait(timeout=max(0.1, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    stop_group(process)
                    code = 124
                item["exit_code"] = code
                path = root / "logs/copsd_official_10lang/rollout_benchmark" / item["run_id"] / "RESULTS.json"
                item["results_path"] = str(path)
                item["results_present"] = path.is_file()
                results.append(item)
                print(json.dumps({"benchmark_finished": item}), flush=True)
        finally:
            for process, handle, item in running:
                # Also remove any child workers left in our recorded group.
                stop_group(process)
                handle.close()

    run_cases([("transformers", 2, 0, "hf2"), ("transformers", 4, 1, "hf4"),
               ("vllm", 2, 2, "vllm2"), ("vllm", 4, 3, "vllm4")])
    by_label = {item["label"]: item for item in results}
    # A same-card, reversed-order comparison for batch2 if the initial run is quick.
    if (all(by_label[label]["exit_code"] == 0 for label in ["hf2", "vllm2"])
            and deadline - time.monotonic() > 600):
        run_cases([("transformers", 2, 2, "hf2_cross"), ("vllm", 2, 0, "vllm2_cross")])
    save(output / "BENCHMARK_DONE.json", {
        "finished_at": datetime.datetime.now().astimezone().isoformat(), "cases": results,
        "scope": "Generation only; no teacher, backward, loss, or online weight synchronization timing",
    })
    if not any(item["backend"] == "transformers" and item["exit_code"] == 0
               and item["results_present"] for item in results):
        raise RuntimeError("No successful Torch2.8 GPU reference benchmark; resume not started")
    for _ in range(6):
        busy = occupied(uuids)
        if not busy:
            break
        time.sleep(5)
    if busy:
        save(output / "RESUME_BLOCKED.json", {"occupied_selected_gpus": busy})
        raise RuntimeError("A selected GPU has a compute process; cannot start resumed training")
    command = ["bash", str(scripts / "run_four_gpu_17b_resume_v2.sh"),
               "th", args.resume_run_id, str(checkpoint)]
    save(output / "RESUME_LAUNCH.json", {"command": command, "queue_gpu_uuids": uuids,
                                        "runtime_change_recorded": True})
    print(json.dumps({"resume_launch": command}), flush=True)
    os.execv("/bin/bash", command)


if __name__ == "__main__":
    main()
