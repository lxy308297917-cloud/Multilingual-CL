#!/usr/bin/env bash
set -euo pipefail
test "${GPU_QUEUE_GPU_COUNT:-}" = 1
[[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* && "$CUDA_VISIBLE_DEVICES" != *,* ]]
lang="${1:?language}"
run_id="${2:?new run ID}"
root="${COPSD_PROJECT_ROOT:-$HOME}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="$root/envs/copsd-vllm011-20261018/bin/python"
args=(--root "$root" --language "$lang" --run-id "$run_id")
if [[ -n "${3:-}" ]]; then args+=(--adapter "$3" --expected-step "${4:?checkpoint step}"); fi
"$python_bin" "$script_dir/evaluate_dev8192.py" "${args[@]}" --prepare-only
export COPSD_VLLM_UUID_COMPAT=1
export PYTHONPATH="$script_dir/compat/vllm011_uuid"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export HF_HOME="$root/cache/copsd_official/huggingface"
export VLLM_CACHE_ROOT="$root/cache/copsd_official/vllm"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
input_lang="$lang"
if [[ "$lang" == en ]]; then input_lang=sw; fi
"$python_bin" "$script_dir/freeze_vllm_eval.py" verify --root "$root" \
  --lock "$root/logs/copsd_official_10lang/dev_vllm_runtime_20261020.CONFIG.json" \
  --input "$root/data/copsd_official_20261015/adapted_v2_ordered/project_dev_100_${input_lang}.json" \
  --source "$root/COPSD-official-20261015/polymath_eval/evaluate_math.py"
"$python_bin" "$script_dir/evaluate_dev8192.py" "${args[@]}"
