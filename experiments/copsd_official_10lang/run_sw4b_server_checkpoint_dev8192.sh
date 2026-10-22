#!/usr/bin/env bash
set -euo pipefail
root="${COPSD_PROJECT_ROOT:-$HOME}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="$root/envs/copsd-vllm011-20261018/bin/python"
"$python_bin" - "$root" "$script_dir" <<'VERIFY'
import sys,json,torch
from pathlib import Path
root,code=map(Path,sys.argv[1:3]);f=json.loads((code/'sw4b_server_pilot20_frozen_20261022.PUBLIC.json').read_text());directory=root/f['checkpoint_relative'];cp=directory/'checkpoint-20'
state=json.loads((root/'gpu-queue/queues/copsd-sw4b-server-pilot20-v1/state.json').read_text());assert state['phase']=='completed' and state['job']['exit_code']==0
proof=json.loads((directory/'SERVER_PILOT20_UPDATES_DONE.json').read_text());assert proof['completed_updates']==20 and proof['micro_steps']==640 and proof['base_unchanged_bitwise'] and len(proof['syncs'])==21
assert json.loads((directory/'SERVER_PARENT_DONE.json').read_text())['two_role_job_completed']
assert json.loads((cp/'trainer_state.json').read_text())['global_step']==20
assert torch.load(cp/'scheduler.pt',map_location='cpu',weights_only=False)['last_epoch']==20
assert {int(v['step']) for v in torch.load(cp/'optimizer.pt',map_location='cpu',weights_only=False)['state'].values()}=={20}
assert (cp/'rng_state.pth').stat().st_size
actual=json.loads((directory/'EXPOSURE_PROMPTS.PRIVATE.json').read_text());prior=json.loads((code/'sw_lr20_frozen_20261022.PUBLIC.json').read_text())
reference=root/prior['conditions']['reference']['checkpoint_relative']/'LR_EXPOSURE_PROMPTS.PRIVATE.json'
assert len(actual)==640 and actual==json.loads(reference.read_text()),'4B vs fresh1.7B actual prompt sequence differs; do not silently pool protocols'
print('SW4B20 trained/checkpoint/restoration-state/input-order acceptance passed, not yet efficacy.',flush=True)
VERIFY
for lang in sw en; do
 mapfile -t pair < <("$python_bin" - "$root" "$script_dir" "$lang" <<'READ'
import sys,json
from pathlib import Path
root,code=map(Path,sys.argv[1:3]);lang=sys.argv[3];f=json.loads((code/'sw4b_server_pilot20_frozen_20261022.PUBLIC.json').read_text())
print(f['evaluation_run_ids'][lang]);print(root/f['checkpoint_relative']/'checkpoint-20')
READ
 )
 bash "$script_dir/run_sw4b_server_dev8192.sh" "$lang" "${pair[0]}" "${pair[1]}" 20
done
