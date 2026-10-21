#!/usr/bin/env bash
set -euo pipefail
root="${COPSD_PROJECT_ROOT:-$HOME}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
plan="${1:?frozen evaluation plan}"
python_bin="$root/envs/copsd-vllm011-20261018/bin/python"
"$python_bin" "$script_dir/sequential_evaluation.py" --root "$root" prepare --plan "$plan"
if [[ "${COPSD_SEQUENCE_EVAL_PREPARE_ONLY:-0}" == 1 ]]; then exit 0; fi
test "${GPU_QUEUE_GPU_COUNT:-}" = 1
[[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* && "$CUDA_VISIBLE_DEVICES" != *,* ]]
exec "$python_bin" "$script_dir/sequential_evaluation.py" --root "$root" run --plan "$plan"
