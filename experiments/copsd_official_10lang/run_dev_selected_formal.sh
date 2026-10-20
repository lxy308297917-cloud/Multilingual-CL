#!/usr/bin/env bash
set -euo pipefail
test "${GPU_QUEUE_GPU_COUNT:-}" = 1
[[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* && "$CUDA_VISIBLE_DEVICES" != *,* ]]
root="${COPSD_PROJECT_ROOT:-$HOME}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lock="${1:?frozen private selection manifest}"
python_bin="$root/envs/copsd-vllm011-20261018/bin/python"
"$python_bin" - "$lock" <<'CHECK'
import hashlib,json,sys
from pathlib import Path
x=json.loads(Path(sys.argv[1]).read_text())
assert x['new_training'] is False and len(x['jobs'])==9
for path,expected in x['sha256'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,path
print('Frozen development-only selection, checkpoint and evaluation identities verified.',flush=True)
CHECK
while IFS=$'\t' read -r benchmark lang run_id checkpoint; do
  bash "$script_dir/run_vllm_math_eval.sh" "$benchmark" "$lang" adapter "$run_id" "$checkpoint"
done < <("$python_bin" - "$lock" <<'JOBS'
import json,sys
for j in json.load(open(sys.argv[1]))['jobs']:
    print('\t'.join(j[k] for k in ['benchmark','language','run_id','checkpoint']))
JOBS
)
printf 'PIPELINE_PHASE dev_selected_formal_done\n'
