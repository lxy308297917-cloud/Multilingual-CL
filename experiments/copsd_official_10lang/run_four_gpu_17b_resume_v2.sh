#!/usr/bin/env bash
set -euo pipefail

# Full four-GPU project single OPD. Use a one-time queue with target_gpus=4;
# this entry resumes saved optimizer/scheduler/RNG state into a new output directory.
test "${GPU_QUEUE_GPU_COUNT:-}" = 4
test -n "${CUDA_VISIBLE_DEVICES:-}"
lang="${1:?usage: run_four_gpu_17b_resume_v2.sh th NEW_RUN_ID ABSOLUTE_CHECKPOINT}"
run_id="${2:?run ID required}"
checkpoint="${3:?complete checkpoint required}"
[[ "$checkpoint" = /* ]] || exit 2
case "$lang" in th) ;; *) exit 2 ;; esac
[[ "$run_id" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2

root=/data0/users/lixiaoyang
data="$root/data/copsd_official_20261015/adapted_v2_ordered/project_train_2900_${lang}.json"
output="$root/models/checkpoint/copsd_official_17b_single_4gpu_v1/$run_id"
test -f "$root/models/Qwen3-1.7B/MODEL_LOCK.json"
test -s "$data"
test ! -e "$output"
# CPU preflight checks original launch identity and all four rank RNG files.
"$root/envs/copsd-official-20261015/bin/python" - "$checkpoint" "$lang" <<'CHECK'
import hashlib,json,sys
from pathlib import Path
checkpoint=Path(sys.argv[1]).resolve()
root=Path('/data0/users/lixiaoyang')
manifest=json.loads((root/'logs/copsd_official_10lang/th_17b_project2900_4gpu_v2.MANIFEST.json').read_text())
assert sys.argv[2]=='th'
assert checkpoint.parent == Path(manifest['checkpoint_root']).resolve()
state=json.loads((checkpoint/'trainer_state.json').read_text())
assert 0 < state['global_step'] < 100
assert checkpoint.name == 'checkpoint-'+str(state['global_step'])
for name in ['adapter_model.safetensors','adapter_config.json','optimizer.pt','scheduler.pt','trainer_state.json']+[f'rng_state_{i}.pth' for i in range(4)]:
    assert (checkpoint/name).stat().st_size > 0, name
source=root/'COPSD-ddp-teacher-unwrapped-20261018'
for name,expected in manifest['source'].items():
    assert hashlib.sha256((source/name).read_bytes()).hexdigest()==expected,name
for key in ['model_lock','data']:
    item=manifest[key]
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256'],key
print('Resume preflight passed; actual distributed restore remains to be verified.',flush=True)
CHECK
export COPSD_CACHE_ROOT="$root/cache/copsd_official"
export WANDB_MODE=offline
export WANDB_DIR="$root/logs/copsd_official_10lang/wandb"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
mkdir -p "$COPSD_CACHE_ROOT" "$WANDB_DIR" "$(dirname "$output")"
cd "$root/COPSD-ddp-teacher-unwrapped-20261018"

# Four workers × micro batch 1 × accumulation 8 = effective batch 32.
# Frozen from successful v7 four-GPU one-update probe; fixed teacher runs
# unwrapped without gradients, student through DDP. This DDP/SDPA/Transformers
# backend is a project adaptation; official four-GPU launcher uses ZeRO-2, FA2, vLLM.
exec "$root/envs/copsd-official-20261015/bin/accelerate" launch \
  --multi_gpu --num_processes 4 --mixed_precision bf16 \
  --gradient_accumulation_steps 8 --main_process_port 12979 \
  multilingual_opsd_train.py \
  --train_language "${lang^^}" --translated_data_path "$data" \
  --model_name_or_path "$root/models/Qwen3-1.7B" \
  --learning_rate 5e-6 --max_grad_norm 0.1 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
  --gradient_checkpointing --gradient_checkpointing_kwargs '{"use_reentrant": false}' --ddp_find_unused_parameters false \
  --max_steps 100 --save_steps 5 --logging_steps 2 \
  --output_dir "$output" --run_config "$run_id" --resume_from_checkpoint "$checkpoint" \
  --max_completion_length 2048 --max_length 20000 \
  --attn_implementation sdpa --torch_dtype bfloat16 \
  --beta 0 --lmbda 1 \
  --use_peft --lora_r 64 --lora_alpha 128 \
  --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --temperature 1.1 --top_p 0.95 --top_k 20 \
  --fixed_teacher --student_enable_thinking --jsd_token_clip 0.05 \
  --report_to tensorboard --seed 42 --data_seed 42 \
  --wandb_project COPSD-four-gpu-project
