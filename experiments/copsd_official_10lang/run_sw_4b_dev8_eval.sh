#!/usr/bin/env bash
set -euo pipefail
test "${GPU_QUEUE_GPU_COUNT:-}" = 1
test -n "${CUDA_VISIBLE_DEVICES:-}"
mode="${1:-}"
case "$mode" in
  base) adapter=; output=sw_4b_dev8_base_v1 ;;
  smoke1) adapter=/data0/users/lixiaoyang/models/checkpoint/copsd_official_sw_4b_smoke1_v2/sw_4b_official_smoke1_v2_sw/checkpoint-1; output=sw_4b_dev8_smoke1_v1 ;;
  *) echo "Usage: $0 base|smoke1" >&2; exit 2 ;;
esac
root=/data0/users/lixiaoyang
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
args=()
if [[ -n "$adapter" ]]; then args+=(--adapter "$adapter"); fi
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
exec "$root/envs/copsd-official-20261015/bin/python" "$script_dir/evaluate_transformers.py" \
  --official-eval-source "$root/COPSD-official-20261015/polymath_eval/evaluate_math.py" \
  --base-model "$root/models/Qwen3-4B" "${args[@]}" \
  --input "$root/data/copsd_official_20261015/adapted_v2_ordered/project_dev_100_sw.json" \
  --input-kind dev --language sw --max-items 8 --samples 1 --max-new-tokens 2048 --seed 42 \
  --output-dir "$root/logs/copsd_official_10lang/$output"
