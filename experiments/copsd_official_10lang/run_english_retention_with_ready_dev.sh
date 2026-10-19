#!/usr/bin/env bash
set -euo pipefail
test "${GPU_QUEUE_GPU_COUNT:-}" = 1
root="${COPSD_PROJECT_ROOT:-$HOME}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if bn_checkpoint="$("$root/envs/copsd-vllm011-20261018/bin/python" - "$root" <<'READY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);state=json.loads((root/'gpu-queue/queues/copsd-bn-17b-vllm-update-batch100-v1/state.json').read_text())
if state['phase']!='completed' or state['job']['exit_code']!=0:raise SystemExit(1)
rid=state['job']['config']['job']['args'][1]
p=root/'models/checkpoint/copsd_vllm_update_batch_single_v1'/rid/'checkpoint-100'
if not (p/'trainer_state.json').exists() or json.loads((p/'trainer_state.json').read_text())['global_step']!=100:raise SystemExit(1)
if not (p/'adapter_model.safetensors').is_file():raise SystemExit(1)
print(p)
READY
)"; then
  printf 'PIPELINE_PHASE completed_BN_final_development\n'
  bash "$script_dir/run_dev_vllm_17b.sh" bn adapter bn_17b_dev100_vllm_ckpt100_20261020_v1 "$bn_checkpoint"
fi
printf 'PIPELINE_PHASE formal_english_retention\n'
exec bash "$script_dir/run_completed_singles_english_gsm8k.sh"
