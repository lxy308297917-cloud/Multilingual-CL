#!/usr/bin/env bash
set -euo pipefail
test "${GPU_QUEUE_GPU_COUNT:-}" = 1
lang="${1:?language}"
run_id="${2:?fresh training run id}"
base_id="${3:?new Base development run id}"
final_id="${4:?new final development run id}"
root="${COPSD_PROJECT_ROOT:-$HOME}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for output in "$root/models/checkpoint/copsd_vllm_seeded_cost_single_v2/$run_id" "$root/logs/copsd_official_10lang/development_vllm/$base_id" "$root/logs/copsd_official_10lang/development_vllm/$final_id"; do test ! -e "$output"; done
printf 'PIPELINE_PHASE base_development\n'
bash "$script_dir/run_dev_vllm_17b.sh" "$lang" base "$base_id"
printf 'PIPELINE_PHASE fresh_training100\n'
bash "$script_dir/run_seeded_cost_batch_17b.sh" "$lang" "$run_id" 100
checkpoint="$root/models/checkpoint/copsd_vllm_seeded_cost_single_v2/$run_id/checkpoint-100"
"$root/envs/copsd-vllm011-20261018/bin/python" - "$checkpoint" <<'CHECK'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);assert json.loads((p/'trainer_state.json').read_text())['global_step']==100
assert json.loads((p.parent/'RUNTIME_SHUTDOWN_DONE.json').read_text())['ordered_pool_release_completed']
assert json.loads((p.parent/'COST_SEGMENT_DONE.json').read_text())['loss_calls']==3188
for name in ['adapter_model.safetensors','optimizer.pt','scheduler.pt','rng_state.pth']:assert (p/name).stat().st_size>0
CHECK
printf 'PIPELINE_PHASE final_development\n'
bash "$script_dir/run_dev_vllm_17b.sh" "$lang" adapter "$final_id" "$checkpoint"
printf 'PIPELINE_PHASE done\n'
