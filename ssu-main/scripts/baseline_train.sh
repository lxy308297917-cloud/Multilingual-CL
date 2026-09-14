#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 METHOD [--plan | --gpu 2 | --gpu 3]" >&2
  exit 2
fi
method="$1"
shift
exec /root/miniconda3/envs/cl/bin/python "$script_dir/train_cl.py" --config "$script_dir/../configs/ig_baseline_train_v1.json" --method "$method" "$@"
