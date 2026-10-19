#!/usr/bin/env bash
set -euo pipefail
test "${GPU_QUEUE_GPU_COUNT:-}" = 1
test -n "${CUDA_VISIBLE_DEVICES:-}"
root="${COPSD_PROJECT_ROOT:-$HOME}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
checkpoint="$root/models/checkpoint/copsd_official_17b_single_v1/sw_17b_project2900_seed42_20261015_v1/checkpoint-100"
"$root/envs/copsd-vllm011-20261018/bin/python" - "$checkpoint" <<'CHECK'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);assert json.loads((p/'trainer_state.json').read_text())['global_step']==100
assert (p/'adapter_model.safetensors').stat().st_size>0
print('Frozen100-update SW reference; intermediate checkpoints remain development-only.',flush=True)
CHECK
# Two separate processes on the same allocated GPU. Each engine starts fresh.
bash "$script_dir/run_vllm_math_eval.sh" mgsm_rev2 sw base sw_17b_mgsm_rev2_base_20261020_v1
bash "$script_dir/run_vllm_math_eval.sh" mgsm_rev2 sw adapter sw_17b_mgsm_rev2_ckpt100_20261020_v1 "$checkpoint"
