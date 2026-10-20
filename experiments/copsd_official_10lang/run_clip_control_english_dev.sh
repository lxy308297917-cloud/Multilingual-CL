#!/usr/bin/env bash
set -euo pipefail
test "${GPU_QUEUE_GPU_COUNT:-}" = 1
[[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* && "$CUDA_VISIBLE_DEVICES" != *,* ]]
root="${COPSD_PROJECT_ROOT:-$HOME}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="$root/envs/copsd-vllm011-20261018/bin/python"
ref="$root/models/checkpoint/copsd_prefix20_replicates_v1/sw_17b_prefix20_horizon100_seed42_20261020_v1/checkpoint-20"
alt="$root/models/checkpoint/copsd_clip_ablation_v1/sw_17b_prefix20_horizon100_unclipped_seed42_20261020_v1/checkpoint-20"
base_id=en_17b_dev100_vllm_base_20261020_v1
ref_id=en_17b_dev100_vllm_sw_clip20_seed42_20261020_v1
alt_id=en_17b_dev100_vllm_sw_unclipped20_seed42_20261020_v1
# Two new evaluations only. Reuse the completed English Base on the same100 IDs.
CUDA_VISIBLE_DEVICES='' "$python_bin" - "$root" "$ref" "$alt" "$base_id" <<'CHECK'
import hashlib,json,sys
from pathlib import Path
root,ref,alt=map(Path,sys.argv[1:4]);name=sys.argv[4]
for q in ['copsd-sw-prefix20-match-v1','copsd-sw-clip-ablation20-v1']:
 s=json.loads((root/'gpu-queue/queues'/q/'state.json').read_text())
 assert s['phase']=='completed' and s['job']['exit_code']==0
for p in [ref,alt]:
 assert json.loads((p/'trainer_state.json').read_text())['global_step']==20
 assert (p/'adapter_model.safetensors').stat().st_size>0
p=root/'logs/copsd_official_10lang/development_vllm'/name
c=json.loads((p/'CONFIG.json').read_text());m=json.loads((p/'DONE.json').read_text())
assert m['items']==100 and not m['formal_test'] and c['language']=='en' and c['adapter'] is None
assert c['protocol']=='project_dev_single_vllm_v1' and c['samples']==1 and c['max_new_tokens']==2048
for k in ['input','source']:
 assert hashlib.sha256(Path(c[k]).read_bytes()).hexdigest()==c[k+'_sha256' if k=='input' else 'official_source_sha256']
print('Completed pair and cached English Base identity verified.',flush=True)
CHECK
bash "$script_dir/run_dev_vllm_17b.sh" en adapter "$ref_id" "$ref"
bash "$script_dir/run_dev_vllm_17b.sh" en adapter "$alt_id" "$alt"
CUDA_VISIBLE_DEVICES='' "$python_bin" - "$root" "$base_id" "$ref_id" "$alt_id" <<'PAIR'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);paths=[root/'logs/copsd_official_10lang/development_vllm'/n for n in sys.argv[2:]]
cfg=[json.loads((p/'CONFIG.json').read_text()) for p in paths]
canonical=[{k:v for k,v in c.items() if k not in ['adapter','output']} for c in cfg]
assert canonical[0]==canonical[1]==canonical[2]
rows=[[json.loads(l) for l in (p/'ITEMS.jsonl').read_text().splitlines()] for p in paths]
metrics=[json.loads((p/'DONE.json').read_text()) for p in paths]
for rr,m in zip(rows,metrics):
 assert len(rr)==len({r['id'] for r in rr})==m['items']==100
 assert sum(r['samples'][0]['correct'] for r in rr)==m['single_correct']
 assert sum(len(r['samples'][0]['token_ids']) for r in rr)==m['generated_tokens']
for triplet in zip(*rows):
 assert all(all(r[k]==triplet[0][k] for k in ['id','question','answer','prompt']) for r in triplet)
 assert len({r['samples'][0]['seed'] for r in triplet})==1
output=root/'logs/copsd_official_10lang/sw_clip_control_english_dev_20261020.PUBLIC.json'
result={'protocol':'project_dev_single_vllm_v1','language':'en','training_language':'sw',
 'correct_out_of100':dict(zip(['Base','clipped20','unclipped20'],[m['single_correct'] for m in metrics])),
 'metrics':dict(zip(['Base','clipped20','unclipped20'],metrics)),
 'inputs_prompts_answers_saved_seeds_and_configs_paired':True,
 'new_inference':{'responses':200,'generated_tokens':sum(m['generated_tokens'] for m in metrics[1:]),
 'generation_seconds':sum(m['generation_seconds'] for m in metrics[1:]),'cached_Base_cost_excluded':True},
 'scope':'English independent development supplement on the same original100 problems; not GSM8K formal test, stable effect or sequential forgetting.'}
with output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result['correct_out_of100']),flush=True)
PAIR
