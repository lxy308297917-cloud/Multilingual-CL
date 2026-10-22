#!/usr/bin/env bash
set -euo pipefail
condition="${1:?reference or lower_lr}"
[[ "$condition" == reference || "$condition" == lower_lr ]]
root="${COPSD_PROJECT_ROOT:-$HOME}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="$root/envs/copsd-vllm011-20261018/bin/python"
export COPSD_OFFICIAL_SOURCE="$root/COPSD-ddp-teacher-unwrapped-20261018"
export COPSD_VLLM_UUID_COMPAT=1 PYTHONPATH="$script_dir/compat/vllm011_uuid"
export COPSD_CACHE_ROOT="$root/cache/copsd_official"
export WANDB_MODE=offline WANDB_DIR="$root/logs/copsd_official_10lang/wandb"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 COPSD_STOP_AFTER_UPDATES=20
export COPSD_EXPECT_INITIAL_SYNC_JSON="$root/models/checkpoint/copsd_prefix20_replicates_v1/sw_17b_prefix20_horizon100_seed42_20261020_v1/COLLOCATE_TRAIN_UPDATES_DONE.json"
"$python_bin" "$script_dir/sw_lr20_entry.py" --root "$root" --condition "$condition" --prepare-only
if [[ "${COPSD_SW_LR_PREPARE_ONLY:-0}" == 1 ]]; then exit 0; fi
test "${GPU_QUEUE_GPU_COUNT:-}" = 1
[[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* && "$CUDA_VISIBLE_DEVICES" != *,* ]]
"$python_bin" "$script_dir/sw_lr20_entry.py" --root "$root" --condition "$condition"
"$python_bin" - "$root" "$script_dir" "$condition" <<'VERIFY'
import json,sys,torch
from pathlib import Path
root,code=map(Path,sys.argv[1:3]);condition=sys.argv[3]
f=json.loads((code/'sw_lr20_frozen_20261022.PUBLIC.json').read_text());s=f['conditions'][condition];p=root/s['checkpoint_relative'];cp=p/'checkpoint-20'
assert json.loads((cp/'trainer_state.json').read_text())['global_step']==20
args=torch.load(cp/'training_args.bin',map_location='cpu',weights_only=False)
assert args.learning_rate==s['learning_rate'] and args.max_steps==100 and args.seed==args.data_seed==42
assert torch.load(cp/'scheduler.pt',map_location='cpu',weights_only=False)['last_epoch']==20
assert {int(v['step']) for v in torch.load(cp/'optimizer.pt',map_location='cpu',weights_only=False)['state'].values()}=={20}
assert (cp/'rng_state.pth').stat().st_size>0
proof=json.loads((p/'COLLOCATE_TRAIN_UPDATES_DONE.json').read_text());assert proof['global_step']==20 and proof['micro_steps']==640 and proof['base_unchanged_bitwise'] and len(proof['syncs'])==21
assert len(json.loads((p/'LR_EXPOSURE_PROMPTS.PRIVATE.json').read_text()))==640
assert json.loads((p/'RUNTIME_SHUTDOWN_DONE.json').read_text())['ordered_pool_release_completed']
print('Fresh SW LR20 checkpoint, optimizer, scheduler, RNG, frozen Base,640 exposure hashes accepted.',flush=True)
VERIFY
for lang in sw en; do
 mapfile -t pair < <("$python_bin" - "$script_dir" "$root" "$condition" "$lang" <<'READ'
import json,sys
from pathlib import Path
c,root=map(Path,sys.argv[1:3]);condition,lang=sys.argv[3:5]
s=json.loads((c/'sw_lr20_frozen_20261022.PUBLIC.json').read_text())['conditions'][condition]
print(s['evaluation_run_ids'][lang]);print(root/s['checkpoint_relative']/'checkpoint-20')
READ
 )
 bash "$script_dir/run_dev8192.sh" "$lang" "${pair[0]}" "${pair[1]}" 20
done
