#!/usr/bin/env bash
set -euo pipefail
test "${GPU_QUEUE_GPU_COUNT:-}" = 2
test -n "${CUDA_VISIBLE_DEVICES:-}"
root=/data0/users/lixiaoyang
data="$root/data/math_single_10lang_20261015"
cd "$root/Multilingual-CL-math-single-20261015"
exec "$root/ssu-migration/envs/cl/bin/python" experiments/math_single_10lang/build_math_sft_data.py \
  --inventory "$data/audit_v2/inventory.jsonl" \
  --splits "$data/split_v8_screened/problem_ids.json" \
  --model "$root/models/Qwen3-14B" \
  --generator-id Qwen/Qwen3-14B \
  --generator-revision 40c069824f4251a91eefaf281ebe4c544efd3e18 \
  --device-map auto \
  --max-new-tokens 640 \
  --student-tokenizer "$root/models/Qwen3-4B" \
  --output-dir "$data/labels_v8_14b_reviewed4_target_probe" \
  --problem-ids-file "$data/labels_v8_14b_reviewed4_target_probe/problem_ids.json"
