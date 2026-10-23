#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
root="${COPSD_PROJECT_ROOT:-$HOME}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="$root/envs/copsd-vllm011-20261018/bin/python"
export COPSD_OFFICIAL_SOURCE="$root/COPSD-ddp-teacher-unwrapped-20261018"
export COPSD_VLLM_UUID_COMPAT=1 PYTHONPATH="$script_dir:$script_dir/compat/vllm011_uuid"
export COPSD_CACHE_ROOT="$root/cache/copsd_official"
export WANDB_MODE=offline WANDB_DIR="$root/logs/copsd_official_10lang/wandb"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset COPSD_EXPECT_INITIAL_SYNC_JSON
"$python_bin" "$script_dir/bn4b_unclip20_job.py" --root "$root" --prepare-only
if [[ "${COPSD_BN4B_UNCLIP_PREPARE_ONLY:-0}" == 1 ]]; then exit 0; fi
test "${GPU_QUEUE_GPU_COUNT:-}" = 2
exec "$python_bin" "$script_dir/bn4b_unclip20_job.py" --root "$root"
