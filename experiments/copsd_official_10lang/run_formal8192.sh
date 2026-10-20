#!/usr/bin/env bash
set -euo pipefail
benchmark="${1:?benchmark}"
lang="${2:?language}"
run_id="${3:?new run ID}"
root="${COPSD_PROJECT_ROOT:-$HOME}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="$root/envs/copsd-vllm011-20261018/bin/python"
"$python_bin" - "$script_dir" <<'LOCK'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]);report=json.loads((p/'formal8192_CPU_preflight_20261020.PUBLIC.json').read_text())
assert hashlib.sha256((p/'evaluate_formal8192.py').read_bytes()).hexdigest()==report['evaluator_sha256']
assert report['protocol']=='project_formal_single_vllm8192_v1'
print('Prepared formal8192 entry identity verified.',flush=True)
LOCK
args=(--root "$root" --benchmark "$benchmark" --language "$lang" --run-id "$run_id")
if [[ -n "${4:-}" ]]; then args+=(--adapter "$4" --expected-step "${5:?checkpoint step}"); fi
"$python_bin" "$script_dir/evaluate_formal8192.py" "${args[@]}" --prepare-only
if [[ "${COPSD_FORMAL_PREPARE_ONLY:-0}" == 1 ]]; then exit 0; fi
test "${GPU_QUEUE_GPU_COUNT:-}" = 1
[[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* && "$CUDA_VISIBLE_DEVICES" != *,* ]]
export COPSD_VLLM_UUID_COMPAT=1
export PYTHONPATH="$script_dir/compat/vllm011_uuid"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export HF_HOME="$root/cache/copsd_official/huggingface"
export VLLM_CACHE_ROOT="$root/cache/copsd_official/vllm"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
telemetry="$root/logs/copsd_official_10lang/$run_id.gpu.csv"
test ! -e "$telemetry"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 1 > "$telemetry" &
telemetry_pid=$!
trap 'kill "$telemetry_pid" 2>/dev/null || true; wait "$telemetry_pid" 2>/dev/null || true' EXIT
"$python_bin" "$script_dir/evaluate_formal8192.py" "${args[@]}"
