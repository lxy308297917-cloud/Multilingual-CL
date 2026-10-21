#!/usr/bin/env bash
set -euo pipefail
root="${COPSD_PROJECT_ROOT:-$HOME}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
config="${1:?frozen matched-single config}"
config="$(realpath "$config")"
export COPSD_OFFICIAL_SOURCE="$root/COPSD-ddp-teacher-unwrapped-20261018"
export COPSD_VLLM_UUID_COMPAT=1
export PYTHONPATH="$script_dir/compat/vllm011_uuid"
export COPSD_CACHE_ROOT="$root/cache/copsd_official"
export WANDB_MODE=offline WANDB_DIR="$root/logs/copsd_official_10lang/wandb"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
args=(--root "$root" --config "$config")
if [[ "${COPSD_MATCHED_PREPARE_ONLY:-0}" == 1 ]]; then
  exec "$root/envs/copsd-vllm011-20261018/bin/python" "$script_dir/matched_pool_single_entry.py" "${args[@]}" --prepare-only
fi
test "${GPU_QUEUE_GPU_COUNT:-}" = 1
[[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* && "$CUDA_VISIBLE_DEVICES" != *,* ]]
run_id="$("$root/envs/copsd-vllm011-20261018/bin/python" -c 'import json,sys;print(json.load(open(sys.argv[1]))["run_id"])' "$config")"
[[ "$run_id" =~ ^[A-Za-z0-9_-]+$ ]]
telemetry="$root/logs/copsd_official_10lang/$run_id.gpu.csv"
test ! -e "$telemetry"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 1 > "$telemetry" &
telemetry_pid=$!
trap 'kill "$telemetry_pid" 2>/dev/null || true; wait "$telemetry_pid" 2>/dev/null || true' EXIT
cd "$COPSD_OFFICIAL_SOURCE"
"$root/envs/copsd-vllm011-20261018/bin/python" "$script_dir/matched_pool_single_entry.py" "${args[@]}"
