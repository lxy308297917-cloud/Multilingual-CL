#!/usr/bin/env bash
set -euo pipefail
root="${COPSD_PROJECT_ROOT:-$HOME}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export COPSD_VLLM_UUID_COMPAT=1
export PYTHONPATH="$script_dir/compat/vllm011_uuid"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export HF_HOME="$root/cache/copsd_official/huggingface"
export VLLM_CACHE_ROOT="$root/cache/copsd_official/vllm"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python_bin="$root/envs/copsd-vllm011-20261018/bin/python"
"$python_bin" "$script_dir/freeze_vllm_eval.py" verify --root "$root" \
 --lock "$root/logs/copsd_official_10lang/dev_vllm_runtime_20261020.CONFIG.json" \
 --input "$root/data/copsd_official_20261015/adapted_v2_ordered/project_dev_100_sw.json" \
 --source "$root/COPSD-official-20261015/polymath_eval/evaluate_math.py"
if [[ "${COPSD_SW_RECOVERY_PREPARE_ONLY:-0}" == 1 ]]; then
 "$python_bin" "$script_dir/evaluate_sw_teacher_recovery32.py" --root "$root" --prepare-only
else
 test "${GPU_QUEUE_GPU_COUNT:-}" = 1
 [[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* && "$CUDA_VISIBLE_DEVICES" != *,* ]]
 "$python_bin" "$script_dir/evaluate_sw_teacher_recovery32.py" --root "$root"
fi
