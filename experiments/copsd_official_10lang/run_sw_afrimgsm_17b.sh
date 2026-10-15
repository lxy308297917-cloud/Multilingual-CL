#!/usr/bin/env bash
set -euo pipefail
test "${GPU_QUEUE_GPU_COUNT:-}" = 1
test -n "${CUDA_VISIBLE_DEVICES:-}"
mode="${1:?usage: run_sw_afrimgsm_17b.sh base|adapter RUN_ID [CHECKPOINT]}"
run_id="${2:?run ID required}"
checkpoint="${3:-}"
[[ "$run_id" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "invalid run ID" >&2; exit 2; }
root=/data0/users/lixiaoyang
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
test -f "$root/models/Qwen3-1.7B/MODEL_LOCK.json"
input="$root/data/copsd_official_20261015/afrimgsm_8e4268d2b949/data/swa/test.tsv"
test -s "$input"
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
  --official-eval-source "$root/COPSD-official-20261015/african_langs_eval/evaluate_math.py" \
  --base-model "$root/models/Qwen3-1.7B" "${args[@]}" \
  --input "$input" --input-kind afrimgsm --language swa \
  --samples 12 --max-new-tokens 1024 --seed 42 \
  --output-dir "$root/logs/copsd_official_10lang/evaluation/$run_id"
