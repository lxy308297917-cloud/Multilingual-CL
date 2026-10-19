#!/usr/bin/env bash
set -euo pipefail
test "${GPU_QUEUE_GPU_COUNT:-}" = 1
[[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* && "$CUDA_VISIBLE_DEVICES" != *,* ]]
benchmark="${1:?usage: run_vllm_math_eval.sh BENCHMARK LANG base|adapter NEW_RUN_ID [CHECKPOINT]}"
lang="${2:?language required}"
mode="${3:?base or adapter required}"
run_id="${4:?new run ID required}"
checkpoint="${5:-}"
[[ "$run_id" =~ ^[a-zA-Z0-9_-]+$ ]]
case "$lang" in sw|te|th|bn|zh|ja|ru|de|fr|es|en|swa) ;; *) exit 2 ;; esac
root="${COPSD_PROJECT_ROOT:-$HOME}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="$root/envs/copsd-vllm011-20261018/bin/python"
source_file="$root/COPSD-official-20261015/polymath_eval/evaluate_math.py"
protocol=project_single_vllm_v1
case "$benchmark" in
  afrimgsm)
    [[ "$lang" == sw || "$lang" == swa ]]
    lang=swa
    input="$root/data/copsd_official_20261015/afrimgsm_8e4268d2b949/data/swa/test.tsv"
    kind=afrimgsm
    protocol=afrimgsm_release_vllm_v1
    source_file="$root/COPSD-official-20261015/african_langs_eval/evaluate_math.py"
    ;;
  mgsm_rev2)
    [[ "$lang" != swa ]]
    input="$root/data/math_single_10lang_20261015/mgsm_rev2/MGSM-Rev2/mgsm_${lang}.tsv"
    kind=mgsm_rev2
    ;;
  polymath)
    [[ "$lang" != en && "$lang" != swa ]]
    input="$root/data/copsd_official_20261015/polymath_project_eval_v1/polymath_all500_${lang}.json"
    kind=polymath_project
    ;;
  gsm8k)
    [[ "$lang" == en ]]
    input="$root/data/copsd_official_20261015/gsm8k_project_en_zeroshot_v1/gsm8k_en_test_1319.json"
    kind=polymath_project
    ;;
  *) exit 2 ;;
esac
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
output="$root/logs/copsd_official_10lang/evaluation_vllm/$run_id"
test ! -e "$output"
export COPSD_VLLM_UUID_COMPAT=1
export PYTHONPATH="$script_dir/compat/vllm011_uuid"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export HF_HOME="$root/cache/copsd_official/huggingface"
export VLLM_CACHE_ROOT="$root/cache/copsd_official/vllm"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
"$python_bin" "$script_dir/freeze_vllm_eval.py" verify --root "$root" \
  --lock "$root/logs/copsd_official_10lang/eval_vllm_runtime_20261019.CONFIG.json" \
  --input "$input" --source "$source_file"
telemetry="$root/logs/copsd_official_10lang/$run_id.gpu.csv"
test ! -e "$telemetry"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 1 > "$telemetry" &
telemetry_pid=$!
trap 'kill "$telemetry_pid" 2>/dev/null || true; wait "$telemetry_pid" 2>/dev/null || true' EXIT
"$python_bin" "$script_dir/evaluate_vllm.py" \
  --official-source "$source_file" --base-model "$root/models/Qwen3-1.7B" \
  --input "$input" --input-kind "$kind" --benchmark "$benchmark" \
  --protocol "$protocol" --language "$lang" --output-dir "$output" \
  --gpu-memory-utilization 0.9 --seed 42 "${args[@]}"
