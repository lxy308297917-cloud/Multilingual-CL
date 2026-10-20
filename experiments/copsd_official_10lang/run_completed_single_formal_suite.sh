#!/usr/bin/env bash
set -euo pipefail
test "${GPU_QUEUE_GPU_COUNT:-}" = 1
[[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* && "$CUDA_VISIBLE_DEVICES" != *,* ]]
lang="${1:?language}"
tag="${2:?new report tag}"
checkpoint="${3:?completed fixed100 checkpoint}"
case "$lang" in sw|te|th|bn|zh|ja|ru|de|fr|es) ;; *) exit 2 ;; esac
[[ "$tag" =~ ^[a-zA-Z0-9_-]+$ ]]
root="${COPSD_PROJECT_ROOT:-$HOME}"
[[ "$checkpoint" == "$root/models/checkpoint/"* ]]
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$root/envs/copsd-vllm011-20261018/bin/python" - "$checkpoint" "$root" <<'CHECK'
import json,sys
from pathlib import Path
p,root=map(Path,sys.argv[1:])
assert json.loads((p/'trainer_state.json').read_text())['global_step']==100
assert all((p/n).stat().st_size>0 for n in ['adapter_model.safetensors','optimizer.pt','scheduler.pt'])
assert any(f.stat().st_size>0 for f in p.glob('rng_state*.pth'))
base=root/'logs/copsd_official_10lang/evaluation_vllm/en_17b_gsm8k_base_20261020_v1'
done=json.loads((base/'DONE.json').read_text())
assert done['items']==1319 and done['formal_full_test'] and done['n']==1
print('Fixed100 checkpoint and cached English Base ready. Formal report only.',flush=True)
CHECK
for benchmark in mgsm_rev2 polymath; do
  for mode in base ckpt100; do
    test ! -e "$root/logs/copsd_official_10lang/evaluation_vllm/${lang}_17b_${benchmark}_${mode}_${tag}"
  done
done
test ! -e "$root/logs/copsd_official_10lang/evaluation_vllm/en_17b_gsm8k_${lang}100_${tag}"
for benchmark in mgsm_rev2 polymath; do
  bash "$script_dir/run_vllm_math_eval.sh" "$benchmark" "$lang" base "${lang}_17b_${benchmark}_base_${tag}"
  bash "$script_dir/run_vllm_math_eval.sh" "$benchmark" "$lang" adapter "${lang}_17b_${benchmark}_ckpt100_${tag}" "$checkpoint"
done
bash "$script_dir/run_vllm_math_eval.sh" gsm8k en adapter "en_17b_gsm8k_${lang}100_${tag}" "$checkpoint"
printf 'PIPELINE_PHASE completed_single_formal_suite_done\n'
