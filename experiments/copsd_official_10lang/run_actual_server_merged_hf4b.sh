#!/usr/bin/env bash
set -euo pipefail
root="${COPSD_PROJECT_ROOT:-$HOME}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export COPSD_VLLM_UUID_COMPAT=1 PYTHONPATH="$script_dir:$script_dir/compat/vllm011_uuid"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python_bin="$root/envs/copsd-vllm011-20261018/bin/python"
"$python_bin" "$script_dir/check_actual_server_merged_hf_4b.py" --root "$root" --prepare-only
if [[ "${COPSD_MERGED_HF4B_PREPARE_ONLY:-0}" == 1 ]]; then exit 0; fi
test "${GPU_QUEUE_GPU_COUNT:-}" = 1
exec "$python_bin" "$script_dir/check_actual_server_merged_hf_4b.py" --root "$root"
