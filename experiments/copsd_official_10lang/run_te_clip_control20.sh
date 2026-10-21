#!/usr/bin/env bash
set -euo pipefail
mode="${1:?clip or unclip}"
[[ "$mode" == clip || "$mode" == unclip ]]
root="${COPSD_PROJECT_ROOT:-$HOME}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
env_dir="$root/envs/copsd-vllm011-20261018"
run_id="te_17b_${mode}_control20_seed42_20261021_v1"
output="$root/models/checkpoint/copsd_te_clip_control_v1/$run_id"
export COPSD_TE_CLIP_MODE="$mode"
export COPSD_OFFICIAL_SOURCE="$root/COPSD-ddp-teacher-unwrapped-20261018"
export COPSD_VLLM_UUID_COMPAT=1
export PYTHONPATH="$script_dir/compat/vllm011_uuid"
export COPSD_CACHE_ROOT="$root/cache/copsd_official"
export WANDB_MODE=offline WANDB_DIR="$root/logs/copsd_official_10lang/wandb"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 COPSD_STOP_AFTER_UPDATES=20
export COPSD_EXPECT_INITIAL_SYNC_JSON="$root/models/checkpoint/copsd_prefix20_replicates_v1/sw_17b_prefix20_horizon100_seed42_20261020_v1/COLLOCATE_TRAIN_UPDATES_DONE.json"
"$env_dir/bin/python" - "$root" "$script_dir" "$mode" <<'CHECK'
import hashlib,json,shutil,sys
from pathlib import Path
root,code=map(Path,sys.argv[1:3]);mode=sys.argv[3]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
lock=json.loads((code/'te_clip_control20_frozen_20261021.PUBLIC.json').read_text())
assert shutil.disk_usage(root).free>=20*2**30
for n,h in lock['code_sha256'].items():assert sha(code/n)==h,n
runtime=json.loads((code/'seeded_cost_runtime_20261020.CONFIG.json').read_text())
for n,h in runtime['code_sha256'].items():assert sha(code/n)==h,n
source=json.loads((root/'logs/copsd_official_10lang/th70_resume_uuid_20261019_v2.MANIFEST.json').read_text())
for n,h in source['core_source_sha256'].items():assert sha(root/'COPSD-ddp-teacher-unwrapped-20261018'/n)==h,n
assert sha(source['model_lock']['path'])==source['model_lock']['sha256']
assert sha(root/lock['training_data_relative'])==lock['training_data_sha256']==runtime['data_sha256']['te']
compat=json.loads((code/'ja_nonefixed_runtime_20261020.CONFIG.json').read_text())
binary=root/'envs/copsd-vllm011-20261018/lib/python3.10/site-packages/vllm/cumem_allocator.abi3.so'
assert sha(binary)==compat['binary_sha256']
for n,h in lock['initial_reference_sha256'].items():assert sha(root/n)==h,n
stage=lock['conditions'][mode]
for path in [root/stage['checkpoint_output_relative'],*[root/'logs/copsd_official_10lang/development_vllm8192'/x for x in stage['evaluation_run_ids'].values()]]:assert not path.exists(),path
print('TE fresh20 source/data/model/initial-policy/binary/output preflight passed; no GPU initialized.',flush=True)
CHECK
if [[ "${COPSD_TE_CLIP_PREPARE_ONLY:-0}" == 1 ]]; then exit 0; fi
test "${GPU_QUEUE_GPU_COUNT:-}" = 1
[[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* && "$CUDA_VISIBLE_DEVICES" != *,* ]]
clip_value=0.05
if [[ "$mode" == unclip ]]; then clip_value=0; fi
telemetry="$root/logs/copsd_official_10lang/$run_id.gpu.csv"
test ! -e "$telemetry"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 1 > "$telemetry" &
telemetry_pid=$!
trap 'kill "$telemetry_pid" 2>/dev/null || true; wait "$telemetry_pid" 2>/dev/null || true' EXIT
cd "$COPSD_OFFICIAL_SOURCE"
"$env_dir/bin/python" "$script_dir/te_clip_control_entry.py" \
 --train_language TE --translated_data_path "$root/data/copsd_official_20261015/adapted_v2_ordered/project_train_2900_te.json" \
 --model_name_or_path "$root/models/Qwen3-1.7B" --learning_rate 5e-6 --max_grad_norm 0.1 \
 --per_device_train_batch_size 1 --gradient_accumulation_steps 32 \
 --gradient_checkpointing --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
 --max_steps 100 --save_steps 5 --logging_steps 2 --output_dir "$output" --run_config "$run_id" \
 --max_completion_length 2048 --max_length 20000 --attn_implementation sdpa --torch_dtype bfloat16 \
 --beta 0 --lmbda 1 --use_peft --lora_r 64 --lora_alpha 128 \
 --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
 --temperature 1.1 --top_p 0.95 --top_k 20 --fixed_teacher --student_enable_thinking --jsd_token_clip "$clip_value" \
 --use_vllm --vllm_mode colocate --vllm_tensor_parallel_size 1 --vllm_gpu_memory_utilization 0.4 \
 --vllm_enable_sleep_mode true --vllm_sync_frequency 1 --report_to tensorboard \
 --logging_dir "$root/logs/copsd_official_10lang/tensorboard/$run_id" --seed 42 --data_seed 42 --wandb_project COPSD-vllm-single
"$env_dir/bin/python" - "$output" "$mode" <<'VERIFY'
import json,sys,torch
from pathlib import Path
p=Path(sys.argv[1]);mode=sys.argv[2];cp=p/'checkpoint-20'
assert json.loads((cp/'trainer_state.json').read_text())['global_step']==20
a=torch.load(cp/'training_args.bin',map_location='cpu',weights_only=False)
assert a.seed==a.data_seed==42 and a.max_steps==100 and a.learning_rate==5e-6
assert torch.load(cp/'scheduler.pt',map_location='cpu',weights_only=False)['last_epoch']==20
assert {int(v['step']) for v in torch.load(cp/'optimizer.pt',map_location='cpu',weights_only=False)['state'].values()}=={20}
for n in ['rng_state.pth','adapter_model.safetensors']:assert (cp/n).stat().st_size>0
x=json.loads((p/'COLLOCATE_TRAIN_UPDATES_DONE.json').read_text())
assert x['global_step']==20 and x['micro_steps']==640 and x['base_unchanged_bitwise'] and len(x['syncs'])==21
assert json.loads((p/'RUNTIME_SHUTDOWN_DONE.json').read_text())['ordered_pool_release_completed']
x=json.loads((p/'ABLATION_ACTUAL.json').read_text());assert x['actual_component_clip']==(.05 if mode=='clip' else None)
print('TE20 saved state, fixed Base and actual component clip accepted.',flush=True)
VERIFY
for lang in te en; do
 bash "$script_dir/run_dev8192.sh" "$lang" "${run_id}_${lang}_dev8192" "$output/checkpoint-20" 20
done
