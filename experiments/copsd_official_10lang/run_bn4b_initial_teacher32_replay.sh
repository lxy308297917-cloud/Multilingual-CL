#!/usr/bin/env bash
set -euo pipefail
root="${COPSD_PROJECT_ROOT:-$HOME}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python_bin="$root/envs/copsd-vllm011-20261018/bin/python"
"$python_bin" "$script_dir/replay_bn4b_initial_teacher32.py" --root "$root" --mode check
if [[ "${COPSD_REPLAY_PREPARE_ONLY:-0}" == 1 ]]; then exit 0; fi
"$python_bin" "$script_dir/replay_bn4b_initial_teacher32.py" --root "$root" --mode run
