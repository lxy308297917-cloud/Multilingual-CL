#!/usr/bin/env bash
set -euo pipefail
root="${COPSD_PROJECT_ROOT:-$HOME}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
model="${1:?seed42 or seed43}"
case "$model" in
 seed42) checkpoint="$root/models/checkpoint/copsd_vllm_seeded_cost_single_v2/zh_17b_project2900_vllm_seeded_cost_seed42_20261020_v2/checkpoint-10" ;;
 seed43) checkpoint="$root/models/checkpoint/copsd_prefix20_replicates_v1/zh_17b_prefix20_horizon100_seed43_20261021_v1/checkpoint-10" ;;
 *) exit 2 ;;
esac
"$root/envs/copsd-vllm011-20261018/bin/python" - "$root" "$script_dir" "$model" <<'CHECK'
import hashlib,json,shutil,sys
from pathlib import Path
root,code=map(Path,sys.argv[1:3]);model=sys.argv[3]
spec=json.loads((code/'zh_saved10_development_frozen_20261021.PUBLIC.json').read_text())
assert shutil.disk_usage(root).free>=20*2**30
for name,sha in spec['code_sha256'].items():assert hashlib.sha256((code/name).read_bytes()).hexdigest()==sha,name
entry=spec['models'][model];cp=root/entry['checkpoint_relative']
assert json.loads((cp/'trainer_state.json').read_text())['global_step']==10
assert hashlib.sha256((cp/'adapter_model.safetensors').read_bytes()).hexdigest()==entry['adapter_sha256']
for run in entry['run_ids'].values():assert not (root/'logs/copsd_official_10lang/development_vllm8192'/run).exists()
print('Frozen saved ZH10 identity verified; no training or selection performed.',flush=True)
CHECK
if [[ "${COPSD_ZH10_PREPARE_ONLY:-0}" == 1 ]]; then
 for lang in zh en; do
  CUDA_VISIBLE_DEVICES='' "$root/envs/copsd-vllm011-20261018/bin/python" "$script_dir/evaluate_dev8192.py" --root "$root" --language "$lang" --run-id "${lang}_zh_saved10_${model}_dev100_8192_20261021_v1" --adapter "$checkpoint" --expected-step 10 --prepare-only
 done
 exit 0
fi
test "${GPU_QUEUE_GPU_COUNT:-}" = 1
[[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* && "$CUDA_VISIBLE_DEVICES" != *,* ]]
for lang in zh en; do
 bash "$script_dir/run_dev8192.sh" "$lang" "${lang}_zh_saved10_${model}_dev100_8192_20261021_v1" "$checkpoint" 10
done
