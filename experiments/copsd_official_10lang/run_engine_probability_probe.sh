#!/usr/bin/env bash
set -euo pipefail
root="${COPSD_PROJECT_ROOT:-$HOME}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="$root/envs/copsd-vllm011-20261018/bin/python"
export COPSD_VLLM_UUID_COMPAT=1 PYTHONPATH="$script_dir/compat/vllm011_uuid"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export HF_HOME="$root/cache/copsd_official/huggingface" VLLM_CACHE_ROOT="$root/cache/copsd_official/vllm"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
"$python_bin" "$script_dir/freeze_vllm_eval.py" verify --root "$root" \
 --lock "$root/logs/copsd_official_10lang/dev_vllm_runtime_20261020.CONFIG.json" \
 --input "$root/data/copsd_official_20261015/adapted_v2_ordered/project_dev_100_sw.json" \
 --source "$root/COPSD-official-20261015/polymath_eval/evaluate_math.py"
for stage in hf native dense; do
 "$python_bin" "$script_dir/engine_probability_probe.py" --root "$root" --stage "$stage" --prepare-only
done
if [[ "${COPSD_ENGINE_PROBE_PREPARE_ONLY:-0}" == 1 ]]; then exit 0; fi
test "${GPU_QUEUE_GPU_COUNT:-}" = 1
[[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* && "$CUDA_VISIBLE_DEVICES" != *,* ]]
for stage in hf native dense; do
 "$python_bin" "$script_dir/engine_probability_probe.py" --root "$root" --stage "$stage"
done
