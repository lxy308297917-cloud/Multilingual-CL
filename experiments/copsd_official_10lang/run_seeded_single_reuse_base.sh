#!/usr/bin/env bash
set -euo pipefail
test "${GPU_QUEUE_GPU_COUNT:-}" = 1
lang="${1:?language}"
run_id="${2:?new fresh training id}"
base_id="${3:?existing Base dev id}"
final_id="${4:?new final dev id}"
root="${COPSD_PROJECT_ROOT:-$HOME}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$root/envs/copsd-vllm011-20261018/bin/python" "$script_dir/verify_cached_development.py" --root "$root" --language "$lang" --run-id "$base_id"
bash "$script_dir/run_seeded_cost_batch_17b.sh" "$lang" "$run_id" 100
checkpoint="$root/models/checkpoint/copsd_vllm_seeded_cost_single_v2/$run_id/checkpoint-100"
"$root/envs/copsd-vllm011-20261018/bin/python" - "$checkpoint" <<'CHECK'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);assert json.loads((p/'trainer_state.json').read_text())['global_step']==100
assert json.loads((p.parent/'RUNTIME_SHUTDOWN_DONE.json').read_text())['ordered_pool_release_completed']
assert json.loads((p.parent/'COST_SEGMENT_DONE.json').read_text())['loss_calls']==3188
CHECK
bash "$script_dir/run_dev_vllm_17b.sh" "$lang" adapter "$final_id" "$checkpoint"
