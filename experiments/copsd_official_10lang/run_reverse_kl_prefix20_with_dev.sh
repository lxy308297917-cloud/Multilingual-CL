#!/usr/bin/env bash
set -euo pipefail
test "${GPU_QUEUE_GPU_COUNT:-}" = 1
# A completed matched reference is required before this GPU job does any work.
lang="${1:?language}"
seed="${2:?training seed}"
run_id="${3:?fresh20-update run id}"
dev_id="${4:?fresh development run id}"
en_id="${5:?English development run id}"
smoke_id=""
[[ -z "$smoke_id" && "$lang" == sw && "$seed" == 42 ]]
root="${COPSD_PROJECT_ROOT:-$HOME}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for output in "$root/models/checkpoint/copsd_kl_direction_ablation_v1/$run_id" "$root/logs/copsd_official_10lang/development_vllm/$dev_id"; do test ! -e "$output"; done
verify_prefix() {
  CUDA_VISIBLE_DEVICES='' COPSD_VLLM_UUID_COMPAT=0 "$root/envs/copsd-vllm011-20261018/bin/python" - "$1" "$2" "$seed" <<'CHECK'
import json,sys
from pathlib import Path
import torch
from safetensors.torch import load_file
root=Path(sys.argv[1]);stop,seed=map(int,sys.argv[2:]);cp=root/f'checkpoint-{stop}'
state=json.loads((cp/'trainer_state.json').read_text())
assert state['global_step']==stop and state['max_steps']==100
for name in ['adapter_model.safetensors','optimizer.pt','scheduler.pt','rng_state.pth','training_args.bin']:assert (cp/name).stat().st_size>0
args=torch.load(cp/'training_args.bin',map_location='cpu',weights_only=False)
assert args.max_steps==100 and args.seed==args.data_seed==seed
assert torch.load(cp/'scheduler.pt',map_location='cpu',weights_only=False)['last_epoch']==stop
opt=torch.load(cp/'optimizer.pt',map_location='cpu',weights_only=False)
assert {int(v['step']) for v in opt['state'].values()}=={stop}
assert json.loads((root/'RUNTIME_SHUTDOWN_DONE.json').read_text())['ordered_pool_release_completed']
proof=json.loads((root/'COLLOCATE_TRAIN_UPDATES_DONE.json').read_text())
assert proof['global_step']==stop and proof['scheduler_horizon']==100
assert proof['micro_steps']==stop*32 and proof['base_unchanged_bitwise']
assert len(proof['syncs'])==stop+1
cost=json.loads((root/'COST_SEGMENT_DONE.json').read_text())
assert cost['global_step']==stop and cost['loss_calls']==stop*32
first,last=[load_file(str(p/'adapter_model.safetensors')) for p in [root,cp]]
assert set(first)==set(last) and all(torch.equal(first[k],last[k]) for k in first)
print(f'PREFIX_ACCEPTED stop={stop}, scheduler_horizon=100, fresh seed={seed}, loss_calls={stop*32}',flush=True)
CHECK
}
if [[ -n "$smoke_id" ]]; then
  printf 'PIPELINE_PHASE independent_prefix2_smoke\n'
  if [[ ! -d "$root/models/checkpoint/copsd_prefix20_checks_v1/$smoke_id" ]]; then
    bash "$script_dir/run_reverse_kl_prefix20_17b.sh" "$lang" "$smoke_id" "$seed" 2
  fi
  verify_prefix "$root/models/checkpoint/copsd_prefix20_checks_v1/$smoke_id" 2
fi
"$root/envs/copsd-vllm011-20261018/bin/python" - "$root" <<'BASELINE'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);q=json.loads((root/'gpu-queue/queues/copsd-sw-clip-ablation20-v1/state.json').read_text())
assert q['phase']=='completed' and q['job']['exit_code']==0, 'Matched beta0/unclipped reference must complete first'
p=root/'models/checkpoint/copsd_clip_ablation_v1/sw_17b_prefix20_horizon100_unclipped_seed42_20261020_v1'
assert json.loads((p/'COLLOCATE_TRAIN_UPDATES_DONE.json').read_text())['global_step']==20
assert json.loads((p/'COST_SEGMENT_DONE.json').read_text())['loss_calls']==640
assert json.loads((root/'logs/copsd_official_10lang/development_vllm/sw_17b_dev100_vllm_prefix20_unclipped_seed42_20261020_v1/DONE.json').read_text())['items']==100
print('Matched beta0/unclipped reference train and independent development completed; controlled ablation may start.',flush=True)
BASELINE
printf 'PIPELINE_PHASE fresh_prefix20_training\n'
bash "$script_dir/run_reverse_kl_prefix20_17b.sh" "$lang" "$run_id" "$seed" 20
verify_prefix "$root/models/checkpoint/copsd_kl_direction_ablation_v1/$run_id" 20
printf 'PIPELINE_PHASE heldout_development100\n'
bash "$script_dir/run_dev_vllm_17b.sh" "$lang" adapter "$dev_id" "$root/models/checkpoint/copsd_kl_direction_ablation_v1/$run_id/checkpoint-20"
printf 'PIPELINE_PHASE prefix20_repeat_done\n'

printf 'PIPELINE_PHASE English_development100\n'
bash "$script_dir/run_dev_vllm_17b.sh" en adapter "$en_id" "$root/models/checkpoint/copsd_kl_direction_ablation_v1/$run_id/checkpoint-20"
