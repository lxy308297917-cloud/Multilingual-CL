#!/usr/bin/env bash
set -euo pipefail
test "${GPU_QUEUE_GPU_COUNT:-}" = 1
root="${COPSD_PROJECT_ROOT:-$HOME}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Preserve the ready-BN development check and frozen English retention comparisons.
bash "$script_dir/run_english_retention_with_ready_dev.sh"
# These are frozen final-checkpoint reports, never language/checkpoint selection.
for lang in sw th; do
  case "$lang" in
    sw) checkpoint="$root/models/checkpoint/copsd_official_17b_single_v1/sw_17b_project2900_seed42_20261015_v1/checkpoint-100" ;;
    th) checkpoint="$root/models/checkpoint/copsd_official_17b_single_4gpu_v1/th_17b_project2900_seed42_4gpu_resume70_torch28_uuid_20261019_v2/checkpoint-100" ;;
  esac
  "$root/envs/copsd-vllm011-20261018/bin/python" - "$checkpoint" <<'CHECK'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
assert json.loads((p/'trainer_state.json').read_text())['global_step']==100
assert (p/'adapter_model.safetensors').stat().st_size>0
CHECK
  # SW MGSM-Rev2 already completed; retain its original scores and outputs.
  if [[ "$lang" == th ]]; then
    bash "$script_dir/run_vllm_math_eval.sh" mgsm_rev2 th base th_17b_mgsm_rev2_base_20261020_v1
    bash "$script_dir/run_vllm_math_eval.sh" mgsm_rev2 th adapter th_17b_mgsm_rev2_ckpt100_20261020_v1 "$checkpoint"
  fi
  bash "$script_dir/run_vllm_math_eval.sh" polymath "$lang" base "${lang}_17b_polymath_base_20261020_v1"
  bash "$script_dir/run_vllm_math_eval.sh" polymath "$lang" adapter "${lang}_17b_polymath_ckpt100_20261020_v1" "$checkpoint"
done
printf 'PIPELINE_PHASE completed_single_formal_suite_done\n'
