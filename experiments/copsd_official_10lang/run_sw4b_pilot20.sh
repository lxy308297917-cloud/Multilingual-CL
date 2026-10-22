#!/usr/bin/env bash
set -euo pipefail
root="${COPSD_PROJECT_ROOT:-$HOME}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="$root/envs/copsd-vllm011-20261018/bin/python"
export COPSD_OFFICIAL_SOURCE="$root/COPSD-ddp-teacher-unwrapped-20261018"
export COPSD_VLLM_UUID_COMPAT=1 PYTHONPATH="$script_dir/compat/vllm011_uuid"
export COPSD_CACHE_ROOT="$root/cache/copsd_official"
export WANDB_MODE=offline WANDB_DIR="$root/logs/copsd_official_10lang/wandb"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 COPSD_STOP_AFTER_UPDATES=20
unset COPSD_EXPECT_INITIAL_SYNC_JSON
"$python_bin" "$script_dir/sw4b_pilot20_entry.py" --root "$root" --prepare-only
if [[ "${COPSD_SW4B_PILOT_PREPARE_ONLY:-0}" == 1 ]]; then exit 0; fi
test "${GPU_QUEUE_GPU_COUNT:-}" = 1
[[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* && "$CUDA_VISIBLE_DEVICES" != *,* ]]
"$python_bin" "$script_dir/sw4b_pilot20_entry.py" --root "$root"
"$python_bin" - "$root" "$script_dir" <<'VERIFY'
import json,sys,torch
from pathlib import Path
root,code=map(Path,sys.argv[1:3]);f=json.loads((code/'sw4b_pilot20_frozen_20261022.PUBLIC.json').read_text());p=root/f['checkpoint_relative'];cp=p/'checkpoint-20'
assert json.loads((cp/'trainer_state.json').read_text())['global_step']==20
assert torch.load(cp/'scheduler.pt',map_location='cpu',weights_only=False)['last_epoch']==20
assert {int(v['step']) for v in torch.load(cp/'optimizer.pt',map_location='cpu',weights_only=False)['state'].values()}=={20}
assert (cp/'rng_state.pth').stat().st_size>0
proof=json.loads((p/'COLLOCATE_TRAIN_UPDATES_DONE.json').read_text());assert proof['micro_steps']==640 and proof['base_unchanged_bitwise'] and len(proof['syncs'])==21
assert json.loads((p/'MASK_REDUCTION_ACTUAL.json').read_text())['position_reduction_calls']==640
assert len(json.loads((p/'EXPOSURE_PROMPTS.PRIVATE.json').read_text()))==640
assert json.loads((p/'RUNTIME_SHUTDOWN_DONE.json').read_text())['ordered_pool_release_completed']
print('Fresh SW4B20 checkpoint,640 exposures,frozenBase and shutdown accepted.',flush=True)
VERIFY
for lang in sw en; do
 mapfile -t pair < <("$python_bin" - "$script_dir" "$root" "$lang" <<'READ'
import json,sys
from pathlib import Path
code,root=map(Path,sys.argv[1:3]);lang=sys.argv[3];f=json.loads((code/'sw4b_pilot20_frozen_20261022.PUBLIC.json').read_text())
print(f['evaluation_run_ids'][lang]);print(root/f['checkpoint_relative']/'checkpoint-20')
READ
 )
 bash "$script_dir/run_sw4b_dev8192.sh" "$lang" "${pair[0]}" "${pair[1]}" 20
done
