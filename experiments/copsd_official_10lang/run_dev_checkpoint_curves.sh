#!/usr/bin/env bash
set -euo pipefail
test "${GPU_QUEUE_GPU_COUNT:-}" = 1
root="${COPSD_PROJECT_ROOT:-$HOME}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="$root/envs/copsd-vllm011-20261018/bin/python"
# Check all inputs before spending GPU time; never restart incomplete outputs.
for lang in sw th bn zh de fr; do
  "$python_bin" "$script_dir/run_dev_checkpoint_curve.py" "$lang" --root "$root" --check-only
done
for lang in sw th bn zh de fr; do
  "$python_bin" "$script_dir/run_dev_checkpoint_curve.py" "$lang" --root "$root"
done
