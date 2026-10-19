#!/usr/bin/env bash
set -euo pipefail
test "${GPU_QUEUE_GPU_COUNT:-}" = 1
root="${COPSD_PROJECT_ROOT:-$HOME}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bn="$root/models/checkpoint/copsd_vllm_update_batch_single_v1/bn_17b_project2900_vllm_update_batch_seed42_20261020_v1/checkpoint-5"
te="$root/models/checkpoint/copsd_vllm_single_v1/te_17b_project2900_vllm_seed42_20261019_v1/checkpoint-10"
"$root/envs/copsd-vllm011-20261018/bin/python" - "$bn" "$te" <<'CHECK'
import json,sys
from pathlib import Path
for name,step in zip(sys.argv[1:],[5,10]):
    p=Path(name);assert json.loads((p/'trainer_state.json').read_text())['global_step']==step
    assert (p/'adapter_model.safetensors').stat().st_size>0
CHECK
bash "$script_dir/run_dev_vllm_17b.sh" bn adapter bn_17b_dev100_vllm_ckpt5_20261020_v1 "$bn"
bash "$script_dir/run_dev_vllm_17b.sh" te adapter te_17b_dev100_vllm_ckpt10_20261020_v1 "$te"
