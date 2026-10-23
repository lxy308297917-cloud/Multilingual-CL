#!/usr/bin/env bash
set -euo pipefail
root="${COPSD_PROJECT_ROOT:-$HOME}"
code="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="$root/envs/copsd-vllm011-20261018/bin/python"
freeze="${1:?Frozen evaluation config path required}"
case_id="${2:?Evaluation case required}"
export COPSD_VLLM_UUID_COMPAT=1 PYTHONPATH="$code/compat/vllm011_uuid"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export HF_HOME="$root/cache/copsd_official/huggingface" VLLM_CACHE_ROOT="$root/cache/copsd_official/vllm"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mapfile -t languages < <("$python_bin" -c 'import json,sys; f=json.load(open(sys.argv[1])); print("\n".join(f["cases"][sys.argv[2]]["evaluation_languages"]))' "$freeze" "$case_id")
test "${#languages[@]}" -gt 0
for language in "${languages[@]}"; do
  "$python_bin" "$code/evaluate_single4b_development.py" --root "$root" --freeze "$freeze" --case "$case_id" --language "$language" --prepare-only
done
if [[ "${COPSD_SINGLE4B_EVAL_PREPARE_ONLY:-0}" == 1 ]]; then exit 0; fi
test "${GPU_QUEUE_GPU_COUNT:-}" = 1
[[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* && "$CUDA_VISIBLE_DEVICES" != *,* ]]
for language in "${languages[@]}"; do
  "$python_bin" "$code/evaluate_single4b_development.py" --root "$root" --freeze "$freeze" --case "$case_id" --language "$language"
done
