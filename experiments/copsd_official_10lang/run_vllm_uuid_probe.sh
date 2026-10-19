#!/usr/bin/env bash
set -euo pipefail
batch="${1:?batch required}"
uuid="${2:?queue-allocated UUID required}"
run_id="${3:?new run ID required}"
case "$batch" in 2|4) ;; *) exit 2 ;; esac
[[ "$uuid" = GPU-* ]] || exit 2
[[ "$run_id" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
root=/data0/users/lixiaoyang
scripts="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output="$root/logs/copsd_official_10lang/handoff/th70_benchmark_resume_20261019_v1"
test ! -e "$output/$run_id.log"
"$root/envs/copsd-vllm011-20261018/bin/python" - "$scripts" "$uuid" "$output" <<'CHECK'
from pathlib import Path
import json,sys
sys.path.insert(0,sys.argv[1])
from benchmark_then_resume import occupied
manifest=json.loads((Path(sys.argv[3])/'MANIFEST.json').read_text())
assert sys.argv[2] in manifest['queue_gpu_uuids']
assert not occupied([sys.argv[2]]), 'GPU is no longer idle; refusing probe'
print('Verified idle UUID within the current four-GPU allocation:',sys.argv[2],flush=True)
CHECK
printf '%s\n' "$$" > "$output/$run_id.pid"
export CUDA_VISIBLE_DEVICES="$uuid"
export GPU_QUEUE_GPU_COUNT=1
export COPSD_VLLM_UUID_COMPAT=1
export PYTHONPATH="$scripts/compat/vllm011_uuid"
exec timeout --signal=TERM --kill-after=20s 12m bash "$scripts/run_rollout_benchmark.sh" \
  vllm "$batch" "$run_id" \
  "$root/models/checkpoint/copsd_official_17b_single_v1/sw_17b_project2900_seed42_20261015_v1/checkpoint-20" \
  > "$output/$run_id.log" 2>&1
