#!/usr/bin/env bash
set -euo pipefail
test "${GPU_QUEUE_GPU_COUNT:-}" = 1
[[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* && "$CUDA_VISIBLE_DEVICES" != *,* ]]
lang=bn
mode="${1:?clip or unclip}"
[[ "$mode" == clip || "$mode" == unclip ]]
export COPSD_BN_CLIP_MODE="$mode"
clip_value=0.05
if [[ "$mode" == unclip ]]; then clip_value=0; fi
run_id="${2:?new run ID required}"
seed="${3:?training seed}"
stop_after="${4:?2 for smoke,20 for development repeat}"
steps=100
case "$lang" in sw|te|th|bn|zh|ja|ru|de|fr|es) ;; *) exit 2 ;; esac
[[ "$run_id" =~ ^[a-zA-Z0-9_-]+$ ]]
[[ "$seed" =~ ^[0-9]+$ ]]
[[ "$seed" == 42 && "$stop_after" == 20 ]]
root="${COPSD_PROJECT_ROOT:-$HOME}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
env_dir="$root/envs/copsd-vllm011-20261018"
track=copsd_bn_clip_control_v1
save_steps=5
logging_steps=2
if [[ "$stop_after" == 2 ]]; then
  track=copsd_prefix20_checks_v1
  save_steps=1
  logging_steps=1
fi
output="$root/models/checkpoint/$track/$run_id"
test ! -e "$output"
export COPSD_OFFICIAL_SOURCE="$root/COPSD-ddp-teacher-unwrapped-20261018"
export COPSD_VLLM_UUID_COMPAT=1
export PYTHONPATH="$script_dir/compat/vllm011_uuid"
export COPSD_CACHE_ROOT="$root/cache/copsd_official"
export WANDB_MODE=offline
export WANDB_DIR="$root/logs/copsd_official_10lang/wandb"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
"$env_dir/bin/python" - "$root" "$script_dir" "$lang" <<'CHECK'
import hashlib,json,sys
from pathlib import Path
root,script_dir=map(Path,sys.argv[1:3]);lang=sys.argv[3]
lock=json.loads((script_dir/'seeded_cost_runtime_20261020.CONFIG.json').read_text())
for name,expected in lock['code_sha256'].items():
    assert hashlib.sha256((script_dir/name).read_bytes()).hexdigest()==expected,name
parent=json.loads((root/'logs/copsd_official_10lang/th70_resume_uuid_20261019_v2.MANIFEST.json').read_text())
for name,expected in parent['core_source_sha256'].items():
    assert hashlib.sha256((root/'COPSD-ddp-teacher-unwrapped-20261018'/name).read_bytes()).hexdigest()==expected,name
assert hashlib.sha256(Path(parent['model_lock']['path']).read_bytes()).hexdigest()==parent['model_lock']['sha256']
data=root/f'data/copsd_official_20261015/adapted_v2_ordered/project_train_2900_{lang}.json'
assert hashlib.sha256(data.read_bytes()).hexdigest()==lock['data_sha256'][lang],lang
prefix=json.loads((script_dir/'prefix20_runtime_20261020.CONFIG.json').read_text())
for name,expected in prefix['code_sha256'].items():
    assert hashlib.sha256((script_dir/name).read_bytes()).hexdigest()==expected,name
print('Frozen runtime/core/model/data identities verified.',flush=True)
CHECK
"$env_dir/bin/python" - "$script_dir" <<'OWNLOCK'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]);lock=json.loads((p/'bn_clip_control_runtime_20261020.CONFIG.json').read_text())
for name,sha in lock['code_sha256'].items():
    assert hashlib.sha256((p/name).read_bytes()).hexdigest()==sha,name
print('BN matched-control launch, entry and evaluation identities verified.',flush=True)
OWNLOCK
export COPSD_EXPECT_INITIAL_SYNC_JSON="$root/models/checkpoint/copsd_prefix20_replicates_v1/sw_17b_prefix20_horizon100_seed42_20261020_v1/COLLOCATE_TRAIN_UPDATES_DONE.json"
test -s "$COPSD_EXPECT_INITIAL_SYNC_JSON"
if [[ "${COPSD_BN_PREPARE_ONLY:-0}" == 1 ]]; then
  printf 'BN control frozen source/data/output preflight passed; no GPU initialized.\n'
  exit 0
fi
export COPSD_STOP_AFTER_UPDATES="$stop_after"
mkdir -p "$(dirname "$output")" "$root/logs/copsd_official_10lang"
telemetry="$root/logs/copsd_official_10lang/$run_id.gpu.csv"
test ! -e "$telemetry"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 1 > "$telemetry" &
telemetry_pid=$!
trap 'kill "$telemetry_pid" 2>/dev/null || true; wait "$telemetry_pid" 2>/dev/null || true' EXIT
cd "$COPSD_OFFICIAL_SOURCE"
"$env_dir/bin/python" "$script_dir/bn_clip_control_entry.py" \
  --train_language "${lang^^}" \
  --translated_data_path "$root/data/copsd_official_20261015/adapted_v2_ordered/project_train_2900_${lang}.json" \
  --model_name_or_path "$root/models/Qwen3-1.7B" \
  --learning_rate 5e-6 --max_grad_norm 0.1 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 32 \
  --gradient_checkpointing --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --max_steps "$steps" --save_steps "$save_steps" --logging_steps "$logging_steps" \
  --output_dir "$output" --run_config "$run_id" \
  --max_completion_length 2048 --max_length 20000 \
  --attn_implementation sdpa --torch_dtype bfloat16 \
  --beta 0 --lmbda 1 --use_peft --lora_r 64 --lora_alpha 128 \
  --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --temperature 1.1 --top_p 0.95 --top_k 20 \
  --fixed_teacher --student_enable_thinking --jsd_token_clip "$clip_value" \
  --use_vllm --vllm_mode colocate --vllm_tensor_parallel_size 1 \
  --vllm_gpu_memory_utilization 0.4 --vllm_enable_sleep_mode true \
  --vllm_sync_frequency 1 --report_to tensorboard --seed "$seed" --data_seed "$seed" \
  --wandb_project COPSD-vllm-single

"$env_dir/bin/python" - "$output" "$mode" <<'VERIFY'
from pathlib import Path
import json,sys,torch
p=Path(sys.argv[1]);mode=sys.argv[2];cp=p/'checkpoint-20'
s=json.loads((cp/'trainer_state.json').read_text());assert s['global_step']==20 and s['max_steps']==100
a=torch.load(cp/'training_args.bin',map_location='cpu',weights_only=False)
assert a.seed==a.data_seed==42 and a.max_steps==100
for name in ['optimizer.pt','scheduler.pt','rng_state.pth','adapter_model.safetensors']:
    assert (cp/name).stat().st_size>0
assert torch.load(cp/'scheduler.pt',map_location='cpu',weights_only=False)['last_epoch']==20
assert {int(v['step']) for v in torch.load(cp/'optimizer.pt',map_location='cpu',weights_only=False)['state'].values()}=={20}
x=json.loads((p/'COLLOCATE_TRAIN_UPDATES_DONE.json').read_text());assert x['global_step']==20 and x['scheduler_horizon']==100 and x['micro_steps']==640 and x['base_unchanged_bitwise'] and len(x['syncs'])==21
assert json.loads((p/'RUNTIME_SHUTDOWN_DONE.json').read_text())['ordered_pool_release_completed']
x=json.loads((p/'ABLATION_ACTUAL.json').read_text());assert x['beta']==0 and x['actual_component_clip']==(.05 if mode=='clip' else None)
print('BN20 recovery state, budget, fixed Base and actual component-clip verified.',flush=True)
VERIFY
bash "$script_dir/run_dev8192.sh" bn "${run_id}_dev8192" "$output/checkpoint-20" 20
