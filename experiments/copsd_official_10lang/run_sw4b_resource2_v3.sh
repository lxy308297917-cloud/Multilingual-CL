#!/usr/bin/env bash
set -euo pipefail
root="${COPSD_PROJECT_ROOT:-$HOME}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="$root/envs/copsd-vllm011-20261018/bin/python"
export COPSD_OFFICIAL_SOURCE="$root/COPSD-ddp-teacher-unwrapped-20261018"
export COPSD_VLLM_UUID_COMPAT=1 PYTHONPATH="$script_dir:$script_dir/compat/vllm011_uuid"
export COPSD_CACHE_ROOT="$root/cache/copsd_official"
export WANDB_MODE=offline WANDB_DIR="$root/logs/copsd_official_10lang/wandb"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 COPSD_STOP_AFTER_UPDATES=2
unset COPSD_EXPECT_INITIAL_SYNC_JSON
"$python_bin" "$script_dir/sw4b_resource2_v3_entry.py" --root "$root" --prepare-only
if [[ "${COPSD_SW4B_RESOURCE_PREPARE_ONLY:-0}" == 1 ]]; then exit 0; fi
test "${GPU_QUEUE_GPU_COUNT:-}" = 1
[[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* && "$CUDA_VISIBLE_DEVICES" != *,* ]]
telemetry="$root/logs/copsd_official_10lang/sw4b_resource2_20261022_v3.gpu.csv"
test ! -e "$telemetry"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits -l 1 > "$telemetry" &
telemetry_pid=$!
trap 'kill "$telemetry_pid" 2>/dev/null || true; wait "$telemetry_pid" 2>/dev/null || true' EXIT
"$python_bin" -c 'import os; from benchmark_eval_concurrency8192 import wait_for_empty_GPU; wait_for_empty_GPU(os.environ["CUDA_VISIBLE_DEVICES"])'
"$python_bin" "$script_dir/check_mask_reduction_memory.py" --upstream "$COPSD_OFFICIAL_SOURCE/multilingual_opsd_trainer.py" --device cuda --output "$script_dir/mask_reduction_memory_CUDA_20261022.PUBLIC.json"
"$python_bin" "$script_dir/sw4b_resource2_v3_entry.py" --root "$root"
