#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec /root/miniconda3/envs/cl/bin/python "$script_dir/evaluate_cl.py" --config "$script_dir/../configs/ig_baseline_eval_v3.json" "$@"
