#!/usr/bin/env bash
set -euo pipefail
test "${GPU_QUEUE_GPU_COUNT:-}" = 1
[[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* && "$CUDA_VISIBLE_DEVICES" != *,* ]]
root="${COPSD_PROJECT_ROOT:-$HOME}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lang="${1:?language}"
python_bin="$root/envs/copsd-vllm011-20261018/bin/python"
"$python_bin" "$script_dir/run_dev_checkpoint_curve.py" "$lang" --root "$root" --check-only
"$python_bin" "$script_dir/run_dev_checkpoint_curve.py" "$lang" --root "$root"
