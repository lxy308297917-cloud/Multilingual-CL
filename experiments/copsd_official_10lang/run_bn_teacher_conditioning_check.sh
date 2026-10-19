#!/usr/bin/env bash
set -euo pipefail
test "${GPU_QUEUE_GPU_COUNT:-}" = 1
[[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* && "$CUDA_VISIBLE_DEVICES" != *,* ]]
root="${COPSD_PROJECT_ROOT:-$HOME}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="$root/envs/copsd-vllm011-20261018/bin/python"
prepared="$root/logs/copsd_official_10lang/teacher_context/bn_dev100_20261020_v1"
output="$root/logs/copsd_official_10lang/teacher_context/bn_conditioning_dev100_20261020_v1"
test ! -e "$output"
"$python_bin" - "$root" "$script_dir" "$prepared" <<'CHECK'
import hashlib,json,sys
from pathlib import Path
root,code,prepared=map(Path,sys.argv[1:])
lock=json.loads((root/'logs/copsd_official_10lang/bn_teacher_context_runtime_20261020.CONFIG.json').read_text())
for name,value in lock['code_sha256'].items():
    assert hashlib.sha256((code/name).read_bytes()).hexdigest()==value,name
for name,value in lock['prepared_sha256'].items():
    assert hashlib.sha256((prepared/name).read_bytes()).hexdigest()==value,name
print('Private conditioning inputs and diagnostic code identity verified.',flush=True)
CHECK
export COPSD_VLLM_UUID_COMPAT=1
export PYTHONPATH="$script_dir/compat/vllm011_uuid"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export HF_HOME="$root/cache/copsd_official/huggingface"
export VLLM_CACHE_ROOT="$root/cache/copsd_official/vllm"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
"$python_bin" "$script_dir/freeze_vllm_eval.py" verify --root "$root" \
  --lock "$root/logs/copsd_official_10lang/dev_vllm_runtime_20261020.CONFIG.json" \
  --input "$root/data/copsd_official_20261015/adapted_v2_ordered/project_dev_100_bn.json" \
  --source "$root/COPSD-official-20261015/polymath_eval/evaluate_math.py"
telemetry="$root/logs/copsd_official_10lang/bn_conditioning_dev100_20261020_v1.gpu.csv"
test ! -e "$telemetry"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 1 > "$telemetry" &
telemetry_pid=$!
trap 'kill "$telemetry_pid" 2>/dev/null || true; wait "$telemetry_pid" 2>/dev/null || true' EXIT
"$python_bin" "$script_dir/evaluate_teacher_conditioning.py" --prepared "$prepared" --output "$output"
