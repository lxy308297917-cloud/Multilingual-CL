#!/usr/bin/env bash
set -euo pipefail
lang="${1:?language}"
run_id="${2:?new independent continuation run id}"
checkpoint="${3:?saved full checkpoint}"
reference="${4:?private parent inference-sync JSON}"
root="${COPSD_PROJECT_ROOT:-$HOME}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$root/envs/copsd-vllm011-20261018/bin/python" - "$checkpoint" "$reference" <<'CHECK'
import json,sys
from pathlib import Path
p,ref=map(Path,sys.argv[1:]);state=json.loads((p/'trainer_state.json').read_text())
step=state['global_step'];assert 0<step<100
for name in ['adapter_model.safetensors','optimizer.pt','scheduler.pt','rng_state.pth','training_args.bin']:
    assert (p/name).stat().st_size>0,name
rows=json.loads(ref.read_text())['syncs']
assert len([r for r in rows if r['global_step']==step])==1
print('Saved checkpoint and exact parent policy reference verified.',flush=True)
CHECK
export COPSD_EXPECT_INITIAL_SYNC_JSON="$reference"
exec bash "$script_dir/run_seeded_cost_batch_17b.sh" "$lang" "$run_id" 100 "$checkpoint"
