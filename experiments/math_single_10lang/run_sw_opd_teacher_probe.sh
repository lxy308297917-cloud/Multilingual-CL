#!/usr/bin/env bash
set -euo pipefail
test "${GPU_QUEUE_GPU_COUNT:-}" = 1
test -n "${CUDA_VISIBLE_DEVICES:-}"
root=/data0/users/lixiaoyang
cd "$root/Multilingual-CL-math-single-20261015"
exec "$root/ssu-migration/envs/cl/bin/python" experiments/math_single_10lang/sw_opd_teacher_probe.py \
  --inventory "$root/data/math_single_10lang_20261015/audit_v2/inventory.jsonl" \
  --ids-file "$root/logs/math_sw_opd_20261015/teacher_probe2_ids.json" \
  --model "$root/models/Qwen3-4B" \
  --output-dir "$root/logs/math_sw_opd_20261015/teacher_probe2/result" \
  --polymath-code "$root/data/math_single_10lang_20261015/polymath_code" \
  --scorer-deps "$root/data/math_single_10lang_20261015/scorer_deps" \
  --seed 17 --max-new-tokens 4096
