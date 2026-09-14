#!/usr/bin/env bash
# Runtime-only launcher; evaluation task/config/code identities remain frozen in v3.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
if [[ $# -eq 1 && "$1" != -* ]]; then
  /root/miniconda3/envs/cl/bin/python "$script_dir/src/baseline_language_first.py" "$1"
fi
exec bash "$script_dir/baseline_eval.sh" "$@"
