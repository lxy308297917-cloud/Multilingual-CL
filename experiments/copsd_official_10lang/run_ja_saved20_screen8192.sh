#!/usr/bin/env bash
set -euo pipefail
root="${COPSD_PROJECT_ROOT:-$HOME}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="$root/envs/copsd-vllm011-20261018/bin/python"
checkpoint="$root/models/checkpoint/copsd_vllm_ja_data_repair_v4/ja_17b_project2900_prefixrepair_nonefixed_seed42_20261020_v4/checkpoint-20"
"$python_bin" - "$root" "$script_dir" "$checkpoint" <<'CHECK'
from pathlib import Path
import json,hashlib,sys
root,script,cp=map(Path,sys.argv[1:])
frozen=json.loads((script/'ja_saved20_screen8192_frozen_20261020.PUBLIC.json').read_text())
for name,sha in frozen['code_sha256'].items():
    assert hashlib.sha256((script/name).read_bytes()).hexdigest()==sha,name
assert hashlib.sha256((cp/'adapter_model.safetensors').read_bytes()).hexdigest()==frozen['adapter_sha256']
state=json.loads((cp/'trainer_state.json').read_text());cost=json.loads((cp/'COST_SEGMENT_PROGRESS.json').read_text())
assert state['global_step']==cost['global_step']==20 and state['max_steps']==100
assert cost['loss_calls']==640 and cost['start_step']==0
for name in ['optimizer.pt','scheduler.pt','rng_state.pth']:
    assert (cp/name).stat().st_size>0
path=root/'data/copsd_official_20261015/adapted_v2_ordered/project_dev_100_ja.json'
assert hashlib.sha256(path.read_bytes()).hexdigest()==frozen['development_sha256']
rows=json.loads(path.read_text());english=json.loads((path.with_name('project_dev_100_sw.json')).read_text())
assert len(rows)==len(english)==100
assert all(all(a[k]==b[k] for k in ['problem_id','problem','Answer']) for a,b in zip(rows,english))
print('Saved JA20 checkpoint/cost marker/source/input identities verified; training is separate.',flush=True)
CHECK
"$python_bin" "$script_dir/evaluate_dev8192.py" --root "$root" --language ja --run-id ja_v4_saved20_base_dev100_8192_20261020_v1 --prepare-only
"$python_bin" "$script_dir/evaluate_dev8192.py" --root "$root" --language ja --run-id ja_v4_saved20_target_dev100_8192_20261020_v1 --adapter "$checkpoint" --expected-step 20 --prepare-only
"$python_bin" "$script_dir/evaluate_dev8192.py" --root "$root" --language en --run-id ja_v4_saved20_English_dev100_8192_20261020_v1 --adapter "$checkpoint" --expected-step 20 --prepare-only
if [[ "${COPSD_SCREEN_PREPARE_ONLY:-0}" == 1 ]]; then exit 0; fi
test "${GPU_QUEUE_GPU_COUNT:-}" = 1
[[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* && "$CUDA_VISIBLE_DEVICES" != *,* ]]
bash "$script_dir/run_dev8192.sh" ja ja_v4_saved20_base_dev100_8192_20261020_v1
bash "$script_dir/run_dev8192.sh" ja ja_v4_saved20_target_dev100_8192_20261020_v1 "$checkpoint" 20
bash "$script_dir/run_dev8192.sh" en ja_v4_saved20_English_dev100_8192_20261020_v1 "$checkpoint" 20
