#!/usr/bin/env bash
set -euo pipefail
test "${GPU_QUEUE_GPU_COUNT:-}" = 4
test -n "${CUDA_VISIBLE_DEVICES:-}"
root=/data0/users/lixiaoyang
cd "$root/Multilingual-CL-math-single-20261015"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
exec "$root/ssu-migration/envs/cl/bin/torchrun" --standalone --nproc_per_node=4 \
  experiments/math_single_10lang/train_sw_opd_dual_fsdp_smoke.py \
  --inventory "$root/data/math_single_10lang_20261015/audit_v2/inventory.jsonl" \
  --source-manifest "$root/logs/math_sw_opd_20261015/SW_STATIC_TRIAGE_V1.jsonl" \
  --source-review "$root/logs/math_sw_opd_20261015/SW_SOURCE_REVIEW_FIRST10_CODEX.jsonl" \
  --split-file "$root/logs/math_sw_opd_20261015/SW_SPLIT_CANDIDATE_V1.json" \
  --train-ids-file "$root/logs/math_sw_opd_20261015/SW_OPD_FSDP_SMOKE_TRAIN_IDS_V1.json" \
  --dev-ids-file "$root/logs/math_sw_opd_20261015/teacher_probe2_ids.json" \
  --model "$root/models/Qwen3-4B" \
  --model-revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --polymath-code "$root/data/math_single_10lang_20261015/polymath_code" \
  --scorer-deps "$root/data/math_single_10lang_20261015/scorer_deps" \
  --output-dir "$root/logs/math_sw_opd_20261015/opd_dual_fsdp_smoke_v1" \
  --max-steps 5 --rollout-max-new-tokens 64 --eval-max-new-tokens 128 \
  --seed 42 --smoke
