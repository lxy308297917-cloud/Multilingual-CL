#!/usr/bin/env bash
set -euo pipefail
shard="${1:?shard index required}"
case "$shard" in 0|1|2|3) ;; *) echo "invalid shard: $shard" >&2; exit 2 ;; esac
test "${GPU_QUEUE_GPU_COUNT:-1}" = 1
test -n "${CUDA_VISIBLE_DEVICES:-}"
root=/data0/users/lixiaoyang
cd "$root/Multilingual-CL-math-single-20261015"
run="$root/logs/math_single_10lang/dev_select_en_base_seed17_v8_8192_20261015"
exec "$root/ssu-migration/envs/cl/bin/python" experiments/math_single_10lang/eval_math.py \
  --items "$run/shard${shard}_items.jsonl" \
  --model "$root/models/Qwen3-4B" \
  --tokenizer "$root/models/Qwen3-4B" \
  --output-dir "$run/shard${shard}/result" \
  --seed 17 \
  --max-new-tokens 8192 \
  --polymath-code "$root/data/math_single_10lang_20261015/polymath_code" \
  --scorer-deps "$root/data/math_single_10lang_20261015/scorer_deps"
