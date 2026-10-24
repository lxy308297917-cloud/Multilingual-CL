#!/usr/bin/env bash
set -euo pipefail
root="${COPSD_PROJECT_ROOT:-$HOME}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="$root/envs/copsd-vllm011-20261018/bin/python"
freeze="${1:?Frozen formal4B plan required}"
case_id="${2:?Frozen case identifier required}"
export COPSD_VLLM_UUID_COMPAT=1
export PYTHONPATH="$script_dir:$script_dir/compat/vllm011_uuid"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export HF_HOME="$root/cache/copsd_official/huggingface"
export VLLM_CACHE_ROOT="$root/cache/copsd_official/vllm"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
"$python_bin" "$script_dir/evaluate_formal4b.py" --root "$root" --freeze "$freeze" --case "$case_id" --prepare-only
if [[ "${COPSD_FORMAL4B_PREPARE_ONLY:-0}" == 1 ]]; then exit 0; fi
test "${GPU_QUEUE_GPU_COUNT:-}" = 1
[[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* && "$CUDA_VISIBLE_DEVICES" != *,* ]]
exec "$python_bin" "$script_dir/evaluate_formal4b.py" --root "$root" --freeze "$freeze" --case "$case_id"
