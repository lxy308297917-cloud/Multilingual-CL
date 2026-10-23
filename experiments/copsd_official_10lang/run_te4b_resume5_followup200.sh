#!/usr/bin/env bash
set -euo pipefail
root="${COPSD_PROJECT_ROOT:-$HOME}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="$root/envs/copsd-vllm011-20261018/bin/python"
export COPSD_VLLM_UUID_COMPAT=1 PYTHONPATH="$script_dir/compat/vllm011_uuid"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export HF_HOME="$root/cache/copsd_official/huggingface" VLLM_CACHE_ROOT="$root/cache/copsd_official/vllm"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
case_id="${1:?TE20R}"
case "$case_id" in TE20R) languages=(te en) ;; *) exit 2 ;; esac
for language in "${languages[@]}"; do
  "$python_bin" "$script_dir/evaluate_te4b_resume5_followup200.py" --root "$root" --case "$case_id" --language "$language" --prepare-only
done
if [[ "${COPSD_TE4B_RESUME5_EVAL_PREPARE_ONLY:-0}" == 1 ]]; then exit 0; fi
test "${GPU_QUEUE_GPU_COUNT:-}" = 1
[[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* && "$CUDA_VISIBLE_DEVICES" != *,* ]]
for language in "${languages[@]}"; do
  "$python_bin" "$script_dir/evaluate_te4b_resume5_followup200.py" --root "$root" --case "$case_id" --language "$language"
done
