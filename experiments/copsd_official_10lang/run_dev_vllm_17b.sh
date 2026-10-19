#!/usr/bin/env bash
set -euo pipefail
test "${GPU_QUEUE_GPU_COUNT:-}" = 1
[[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* && "$CUDA_VISIBLE_DEVICES" != *,* ]]
lang="${1:?usage: run_dev_vllm_17b.sh LANG base|adapter NEW_RUN_ID [CHECKPOINT]}"
mode="${2:?base or adapter}"
run_id="${3:?new run id}"
checkpoint="${4:-}"
[[ "$run_id" =~ ^[a-zA-Z0-9_-]+$ ]]
case "$lang" in sw|te|th|bn|zh|ja|ru|de|fr|es|en) ;; *) exit 2 ;; esac
root="${COPSD_PROJECT_ROOT:-$HOME}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="$root/envs/copsd-vllm011-20261018/bin/python"
source_file="$root/COPSD-official-20261015/polymath_eval/evaluate_math.py"
input_lang="$lang"
if [[ "$lang" == en ]]; then input_lang=sw; fi
input="$root/data/copsd_official_20261015/adapted_v2_ordered/project_dev_100_${input_lang}.json"
args=()
case "$mode" in
  base) test -z "$checkpoint" ;;
  adapter)
    [[ "$checkpoint" == "$root/models/checkpoint/"* ]]
    test -s "$checkpoint/adapter_model.safetensors"
    args+=(--adapter "$checkpoint")
    ;;
  *) exit 2 ;;
esac
output="$root/logs/copsd_official_10lang/development_vllm/$run_id"
test ! -e "$output"
export COPSD_VLLM_UUID_COMPAT=1
export PYTHONPATH="$script_dir/compat/vllm011_uuid"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export HF_HOME="$root/cache/copsd_official/huggingface"
export VLLM_CACHE_ROOT="$root/cache/copsd_official/vllm"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
"$python_bin" "$script_dir/freeze_vllm_eval.py" verify --root "$root" \
  --lock "$root/logs/copsd_official_10lang/dev_vllm_runtime_20261020.CONFIG.json" \
  --input "$input" --source "$source_file"
telemetry="$root/logs/copsd_official_10lang/$run_id.gpu.csv"
test ! -e "$telemetry"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 1 > "$telemetry" &
telemetry_pid=$!
trap 'kill "$telemetry_pid" 2>/dev/null || true; wait "$telemetry_pid" 2>/dev/null || true' EXIT
"$python_bin" "$script_dir/evaluate_dev_vllm.py" \
  --source "$source_file" --model "$root/models/Qwen3-1.7B" \
  --input "$input" --language "$lang" --output "$output" "${args[@]}"
