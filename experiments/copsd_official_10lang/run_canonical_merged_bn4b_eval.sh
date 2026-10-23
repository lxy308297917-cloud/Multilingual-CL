#!/usr/bin/env bash
set -euo pipefail
root="${COPSD_PROJECT_ROOT:-$HOME}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export COPSD_VLLM_UUID_COMPAT=1 PYTHONPATH="$script_dir:$script_dir/compat/vllm011_uuid"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python_bin="$root/envs/copsd-vllm011-20261018/bin/python"
case_id="${1:?Base or BN20}"
if [[ "${COPSD_CANONICAL_BN4B_PREPARE_ONLY:-0}" == 1 ]]; then
  exec "$python_bin" "$script_dir/canonical_merged_bn4b_eval_job.py" --root "$root" --case "$case_id" --prepare-only
fi
test "${GPU_QUEUE_GPU_COUNT:-}" = 1
exec "$python_bin" "$script_dir/canonical_merged_bn4b_eval_job.py" --root "$root" --case "$case_id"
