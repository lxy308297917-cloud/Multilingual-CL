#!/usr/bin/env bash
set -euo pipefail
test "${GPU_QUEUE_GPU_COUNT:-}" = 1
[[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* && "$CUDA_VISIBLE_DEVICES" != *,* ]]
root="${COPSD_PROJECT_ROOT:-$HOME}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export COPSD_VLLM_UUID_COMPAT=1
export PYTHONPATH="$script_dir/compat/vllm011_uuid"
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 OMP_NUM_THREADS=4
exec "$root/envs/copsd-vllm011-20261018/bin/python" "$script_dir/check_cumem_none_ref_gpu.py" \
  --lock "$script_dir/cumem_none_ref_runtime_20261020.CONFIG.json" \
  --output "$root/logs/copsd_official_10lang/errors/cumem_none_ref_GPU_20261020.PUBLIC.json"
