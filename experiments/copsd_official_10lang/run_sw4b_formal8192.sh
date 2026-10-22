#!/usr/bin/env bash
set -euo pipefail
root="${COPSD_PROJECT_ROOT:-$HOME}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="$root/envs/copsd-vllm011-20261018/bin/python"
export COPSD_VLLM_UUID_COMPAT=1 PYTHONPATH="$script_dir/compat/vllm011_uuid"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export HF_HOME="$root/cache/copsd_official/huggingface" VLLM_CACHE_ROOT="$root/cache/copsd_official/vllm"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
case_id="${1:?Base or SW20}"
benchmark="${2:?mgsm_rev2 or gsm8k}"
case "$case_id" in Base|SW20) ;; *) exit 2 ;; esac
case "$benchmark" in mgsm_rev2) lang=sw ;; gsm8k) lang=en ;; *) exit 2 ;; esac
run_id="sw4b_formal_${case_id}_${benchmark}_${lang}_20261022_v1"
args=(--root "$root" --case "$case_id" --benchmark "$benchmark" --language "$lang" --run-id "$run_id")
if [[ "$case_id" == SW20 ]]; then
 args+=(--adapter "$root/models/checkpoint/copsd_sw4b_server_pilot20_v1/sw_4b_server_pilot20_safe2700_horizon100_seed42_20261022_v1/checkpoint-20" --expected-step 20)
fi
"$python_bin" "$script_dir/evaluate_sw4b_formal8192.py" "${args[@]}" --prepare-only
if [[ "${COPSD_SW4B_FORMAL_PREPARE_ONLY:-0}" == 1 ]]; then exit 0; fi
test "${GPU_QUEUE_GPU_COUNT:-}" = 1
[[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* && "$CUDA_VISIBLE_DEVICES" != *,* ]]
exec "$python_bin" "$script_dir/evaluate_sw4b_formal8192.py" "${args[@]}"
