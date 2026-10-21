#!/usr/bin/env bash
set -euo pipefail
root="${COPSD_PROJECT_ROOT:-$HOME}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$root/envs/copsd-vllm011-20261018/bin/python" - "$root" "$script_dir" <<'CHECK'
import hashlib,json,shutil,sys
from pathlib import Path
root,code=map(Path,sys.argv[1:]);lock=json.loads((code/'sw20_afrimgsm_frozen_20261021.PUBLIC.json').read_text())
assert shutil.disk_usage(root).free>=20*2**30
for name,sha in lock['code_sha256'].items():assert hashlib.sha256((code/name).read_bytes()).hexdigest()==sha,name
for name,sha in lock['cached_Base_sha256'].items():
 p=root/'logs/copsd_official_10lang/evaluation_vllm'/lock['cached_Base_run_id']/name
 assert hashlib.sha256(p.read_bytes()).hexdigest()==sha,name
for stage in lock['stages']:
 p=root/stage['checkpoint_relative']
 assert json.loads((p/'trainer_state.json').read_text())['global_step']==20
 assert hashlib.sha256((p/'adapter_model.safetensors').read_bytes()).hexdigest()==stage['adapter_sha256']
 assert not (root/'logs/copsd_official_10lang/evaluation_vllm'/stage['run_id']).exists()
print('SW20 release pass12 source/checkpoint/cache identities verified; no model initialized.',flush=True)
CHECK
if [[ "${COPSD_SW20_AFRI_PREPARE_ONLY:-0}" == 1 ]]; then exit 0; fi
test "${GPU_QUEUE_GPU_COUNT:-}" = 1
[[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* && "$CUDA_VISIBLE_DEVICES" != *,* ]]
bash "$script_dir/run_vllm_math_eval.sh" afrimgsm sw adapter sw_17b_afrimgsm_release_prefix20_seed42_20261021_v1 "$root/models/checkpoint/copsd_prefix20_replicates_v1/sw_17b_prefix20_horizon100_seed42_20261020_v1/checkpoint-20"
bash "$script_dir/run_vllm_math_eval.sh" afrimgsm sw adapter sw_17b_afrimgsm_release_prefix20_seed43_20261021_v1 "$root/models/checkpoint/copsd_prefix20_replicates_v1/sw_17b_prefix20_horizon100_seed43_20261020_v1/checkpoint-20"
