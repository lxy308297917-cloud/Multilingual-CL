#!/usr/bin/env bash
set -euo pipefail
test "${GPU_QUEUE_GPU_COUNT:-}" = 1
lang="${1:?language}"
run_id="${2:?fresh training run id}"
base_id="${3:?new Base development run id}"
final_id="${4:?new final development run id}"
root="${COPSD_PROJECT_ROOT:-$HOME}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for output in "$root/models/checkpoint/copsd_vllm_seeded_cost_single_v2/$run_id" "$root/logs/copsd_official_10lang/development_vllm/$final_id"; do test ! -e "$output"; done
printf 'PIPELINE_PHASE verify_cached_Base_development\n'
"$root/envs/copsd-vllm011-20261018/bin/python" - "$root" "$script_dir" "$lang" "$base_id" <<'BASE'
import hashlib,json,sys
from pathlib import Path
root,script=map(Path,sys.argv[1:3]);lang,base_id=sys.argv[3:]
p=root/'logs/copsd_official_10lang/development_vllm'/base_id
c=json.loads((p/'CONFIG.json').read_text());m=json.loads((p/'DONE.json').read_text())
assert m['items']==100 and not m['formal_test'] and c['adapter'] is None
assert c['language']==lang and c['protocol']=='project_dev_single_vllm_v1'
assert c['model']==str(root/'models/Qwen3-1.7B')
assert c['samples']==1 and c['max_new_tokens']==2048
for field,path in [('input_sha256',Path(c['input'])),('official_source_sha256',Path(c['source'])),('evaluator_sha256',script/'evaluate_dev_vllm.py')]:
 assert hashlib.sha256(path.read_bytes()).hexdigest()==c[field],field
print('Cached Base development identity verified; no repeated Base generation.',flush=True)
BASE
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
"$root/envs/copsd-vllm011-20261018/bin/python" - "$root" "$base_id" "$final_id" <<'PAIR'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])/'logs/copsd_official_10lang/development_vllm'
paths=[root/x for x in sys.argv[2:]]
configs=[json.loads((p/'CONFIG.json').read_text()) for p in paths]
assert [{k:v for k,v in c.items() if k not in ['adapter','output']} for c in configs][0]=={k:v for k,v in configs[1].items() if k not in ['adapter','output']}
rows=[[json.loads(l) for l in (p/'ITEMS.jsonl').read_text().splitlines()] for p in paths]
assert len(rows[0])==len(rows[1])==100
for a,b in zip(*rows):
 assert all(a[k]==b[k] for k in ['id','question','answer','prompt'])
 assert a['samples'][0]['seed']==b['samples'][0]['seed']
print('Cached Base and final development inputs/prompts/seeds/configs paired.',flush=True)
PAIR
