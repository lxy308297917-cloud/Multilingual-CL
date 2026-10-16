#!/usr/bin/env bash
set -euo pipefail
test "${GPU_QUEUE_GPU_COUNT:-}" = 1
test -n "${CUDA_VISIBLE_DEVICES:-}"
lang="${1:?usage: run_mgsm_rev2_17b.sh LANG base|adapter RUN_ID [CHECKPOINT]}"
mode="${2:?mode required}"
run_id="${3:?run ID required}"
checkpoint="${4:-}"
case "$lang" in sw|te|th|bn|zh|ja|ru|de|fr|es) ;; *) echo "invalid language: $lang" >&2; exit 2 ;; esac
[[ "$run_id" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "invalid run ID" >&2; exit 2; }
root=/data0/users/lixiaoyang
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
test -f "$root/models/Qwen3-1.7B/MODEL_LOCK.json"
input="$root/data/math_single_10lang_20261015/mgsm_rev2/MGSM-Rev2/mgsm_${lang}.tsv"
test -s "$input"
test "$(wc -l < "$input")" = 250
args=()
case "$mode" in
  base) test -z "$checkpoint" ;;
  adapter) test -s "$checkpoint/adapter_model.safetensors"; args+=(--adapter "$checkpoint") ;;
  *) echo "invalid mode: $mode" >&2; exit 2 ;;
esac
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
exec "$root/envs/copsd-official-20261015/bin/python" "$script_dir/evaluate_transformers.py" \
  --official-eval-source "$root/COPSD-official-20261015/polymath_eval/evaluate_math.py" \
  --base-model "$root/models/Qwen3-1.7B" "${args[@]}" \
  --input "$input" --input-kind mgsm_rev2 --language "$lang" \
  --samples 1 --max-new-tokens 2048 --seed 42 \
  --output-dir "$root/logs/copsd_official_10lang/evaluation/$run_id"
