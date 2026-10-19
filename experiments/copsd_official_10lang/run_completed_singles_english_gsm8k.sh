#!/usr/bin/env bash
set -euo pipefail
test "${GPU_QUEUE_GPU_COUNT:-}" = 1
root="${COPSD_PROJECT_ROOT:-$HOME}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sw="$root/models/checkpoint/copsd_official_17b_single_v1/sw_17b_project2900_seed42_20261015_v1/checkpoint-100"
th="$root/models/checkpoint/copsd_official_17b_single_4gpu_v1/th_17b_project2900_seed42_4gpu_resume70_torch28_uuid_20261019_v2/checkpoint-100"
"$root/envs/copsd-vllm011-20261018/bin/python" - "$sw" "$th" <<'CHECK'
import json,sys
from pathlib import Path
for value in sys.argv[1:]:
 p=Path(value);assert json.loads((p/'trainer_state.json').read_text())['global_step']==100
 assert (p/'adapter_model.safetensors').stat().st_size>0
CHECK
# All models use the frozen project zero-shot single-answer retention protocol.
bash "$script_dir/run_vllm_math_eval.sh" gsm8k en base en_17b_gsm8k_base_20261020_v1
bash "$script_dir/run_vllm_math_eval.sh" gsm8k en adapter en_17b_gsm8k_sw100_20261020_v1 "$sw"
bash "$script_dir/run_vllm_math_eval.sh" gsm8k en adapter en_17b_gsm8k_th100_20261020_v1 "$th"
