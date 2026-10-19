#!/usr/bin/env bash
set -euo pipefail
test "${GPU_QUEUE_GPU_COUNT:-}" = 1
[[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* && "$CUDA_VISIBLE_DEVICES" != *,* ]]
root="${COPSD_PROJECT_ROOT:-$HOME}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
checkpoint="$root/models/checkpoint/copsd_vllm_seeded_cost_single_v2/zh_17b_project2900_vllm_seeded_cost_seed42_20261020_v2/checkpoint-100"
"$root/envs/copsd-vllm011-20261018/bin/python" - "$checkpoint" "$root" <<'CHECK'
import json,sys
from pathlib import Path
p,root=map(Path,sys.argv[1:])
assert json.loads((p/'trainer_state.json').read_text())['global_step']==100
assert (p/'adapter_model.safetensors').stat().st_size>0
assert json.loads((p.parent/'COST_SEGMENT_DONE.json').read_text())['loss_calls']==3188
base=root/'logs/copsd_official_10lang/evaluation_vllm/en_17b_gsm8k_base_20261020_v1'
done=json.loads((base/'DONE.json').read_text())
assert done['items']==1319 and done['formal_full_test'] and done['n']==1
print('Frozen ZH100 reference verified; existing English Base retained for later strict pairing.',flush=True)
CHECK
# Formal reports only. No checkpoint/language selection from these test outputs.
for benchmark in mgsm_rev2 polymath; do
  bash "$script_dir/run_vllm_math_eval.sh" "$benchmark" zh base "zh_17b_${benchmark}_base_20261020_v1"
  bash "$script_dir/run_vllm_math_eval.sh" "$benchmark" zh adapter "zh_17b_${benchmark}_ckpt100_20261020_v2" "$checkpoint"
done
bash "$script_dir/run_vllm_math_eval.sh" gsm8k en adapter en_17b_gsm8k_zh100_20261020_v2 "$checkpoint"
printf 'PIPELINE_PHASE zh_completed_formal_suite_done\n'
