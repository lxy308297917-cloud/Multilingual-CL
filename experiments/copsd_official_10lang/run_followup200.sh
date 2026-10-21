#!/usr/bin/env bash
set -euo pipefail
case_id="${1:?frozen case required}"
root="${COPSD_PROJECT_ROOT:-$HOME}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="$root/envs/copsd-vllm011-20261018/bin/python"
export COPSD_VLLM_UUID_COMPAT=1
export PYTHONPATH="$script_dir/compat/vllm011_uuid"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export HF_HOME="$root/cache/copsd_official/huggingface" VLLM_CACHE_ROOT="$root/cache/copsd_official/vllm"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
"$python_bin" - "$root" "$script_dir" "$case_id" <<'CHECK'
import hashlib,json,shutil,sys
from pathlib import Path
root,code=map(Path,sys.argv[1:3]);case=sys.argv[3]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
lock=json.loads((root/'logs/copsd_official_10lang/followup200_runtime_20261021.CONFIG.json').read_text())
p=code/'followup200_frozen_20261021.PUBLIC.json'
assert sha(p)==lock['public_freeze_sha256']
frozen=json.loads(p.read_text());assert case in frozen['cases']
for n,h in frozen['runtime_code_sha256'].items():assert sha(code/n)==h,n
ex=frozen['future_training_exclusion'];assert sha(root/ex['relative'])==ex['sha256']
assert shutil.disk_usage(root).free>=20*2**30
print('Followup200 frozen protocol/source/exclusion and disk preflight passed.',flush=True)
CHECK
"$python_bin" "$script_dir/freeze_vllm_eval.py" verify --root "$root" \
 --lock "$root/logs/copsd_official_10lang/dev_vllm_runtime_20261020.CONFIG.json" \
 --input "$root/data/copsd_official_20261015/adapted_v2_ordered/project_dev_100_sw.json" \
 --source "$root/COPSD-official-20261015/polymath_eval/evaluate_math.py"
mapfile -t languages < <("$python_bin" - "$script_dir" "$case_id" <<'LANGS'
import json,sys
from pathlib import Path
f=json.loads((Path(sys.argv[1])/'followup200_frozen_20261021.PUBLIC.json').read_text())
print('\n'.join(f['cases'][sys.argv[2]]['evaluation_languages']))
LANGS
)
[[ "${#languages[@]}" -ge 2 ]]
for lang in "${languages[@]}"; do
 "$python_bin" "$script_dir/evaluate_followup200.py" --root "$root" --case "$case_id" --language "$lang" --prepare-only
done
if [[ "${COPSD_FOLLOWUP_PREPARE_ONLY:-0}" == 1 ]]; then exit 0; fi
test "${GPU_QUEUE_GPU_COUNT:-}" = 1
[[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* && "$CUDA_VISIBLE_DEVICES" != *,* ]]
telemetry="$root/logs/copsd_official_10lang/followup200_${case_id}_20261021_v1.gpu.csv"
test ! -e "$telemetry"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 1 > "$telemetry" &
telemetry_pid=$!
trap 'kill "$telemetry_pid" 2>/dev/null || true; wait "$telemetry_pid" 2>/dev/null || true' EXIT
for lang in "${languages[@]}"; do
 "$python_bin" "$script_dir/evaluate_followup200.py" --root "$root" --case "$case_id" --language "$lang"
done
