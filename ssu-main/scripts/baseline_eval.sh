#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${BASELINE_TRAINING_PYTHON:-/root/miniconda3/envs/cl/bin/python}" "$script_dir/src/evaluate_cl.py" --config "${BASELINE_EVAL_CONFIG:-$script_dir/../configs/ig_baseline_eval_v3.json}" "$@"
