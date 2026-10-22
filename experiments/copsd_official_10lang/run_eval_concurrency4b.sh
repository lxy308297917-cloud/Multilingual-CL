#!/usr/bin/env bash
set -euo pipefail
root="${COPSD_PROJECT_ROOT:-$HOME}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export COPSD_VLLM_UUID_COMPAT=1
export PYTHONPATH="$script_dir/compat/vllm011_uuid"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export HF_HOME="$root/cache/copsd_official/huggingface"
export VLLM_CACHE_ROOT="$root/cache/copsd_official/vllm"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
args=(--root "$root")
if [[ "${COPSD_CONCURRENCY4B_PREPARE_ONLY:-0}" == 1 ]]; then args+=(--prepare-only); fi
exec "$root/envs/copsd-vllm011-20261018/bin/python" "$script_dir/benchmark_eval_concurrency4b.py" "${args[@]}"
