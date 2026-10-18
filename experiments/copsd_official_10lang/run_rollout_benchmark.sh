#!/usr/bin/env bash
set -euo pipefail
test "${GPU_QUEUE_GPU_COUNT:-}" = 1
test -n "${CUDA_VISIBLE_DEVICES:-}"
backend="${1:?usage: run_rollout_benchmark.sh transformers|vllm BATCH RUN_ID [ADAPTER]}"
batch="${2:?batch required}"
run_id="${3:?run ID required}"
adapter="${4:-}"
case "$backend" in transformers|vllm) ;; *) exit 2 ;; esac
case "$batch" in 1|2|4) ;; *) exit 2 ;; esac
[[ "$run_id" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
root=/data0/users/lixiaoyang
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
manifest="$root/logs/copsd_official_10lang/rollout_benchmark/sw_train4_prompts_20261018/PROMPTS.json"
output="$root/logs/copsd_official_10lang/rollout_benchmark/$run_id"
test -s "$manifest"
test ! -e "$output"
args=()
if test -n "$adapter"; then
  test -s "$adapter/adapter_model.safetensors"
  args+=(--adapter "$adapter")
fi
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
# Both backends use the same isolated environment and frozen token IDs.
# Do not infer end-to-end OPD speed from standalone rollout throughput.
exec "$root/envs/copsd-vllm011-20261018/bin/python" "$script_dir/benchmark_rollout.py" \
  --backend "$backend" --batch-size "$batch" --manifest "$manifest" \
  --output-dir "$output" "${args[@]}"
