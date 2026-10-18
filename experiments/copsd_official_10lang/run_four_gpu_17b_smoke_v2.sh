#!/usr/bin/env bash
set -euo pipefail

# One-step throughput probe only. Use the queue with target_gpus=4 once four
# GPUs are free; never run this alongside the four active single-language jobs.
test "${GPU_QUEUE_GPU_COUNT:-}" = 4
test -n "${CUDA_VISIBLE_DEVICES:-}"
lang="${1:?usage: run_four_gpu_17b_smoke.sh LANG RUN_ID}"
run_id="${2:?run ID required}"
case "$lang" in sw|te|th|bn|zh|ja|ru|de|fr|es) ;; *) exit 2 ;; esac
[[ "$run_id" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2

root=/data0/users/lixiaoyang
data="$root/data/copsd_official_20261015/adapted_v2_ordered/project_train_2900_${lang}.json"
output="$root/models/checkpoint/copsd_four_gpu_17b_smoke/$run_id"
test -f "$root/models/Qwen3-1.7B/MODEL_LOCK.json"
test -s "$data"
test ! -e "$output"
export COPSD_CACHE_ROOT="$root/cache/copsd_official"
export WANDB_MODE=offline
export WANDB_DIR="$root/logs/copsd_official_10lang/wandb"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
mkdir -p "$COPSD_CACHE_ROOT" "$WANDB_DIR" "$(dirname "$output")"
cd "$root/COPSD-ddp-adapter-20261018"

# Four workers × micro batch 1 × accumulation 8 = effective batch 32.
# DDP/SDPA/Transformers is an isolated local speed adaptation: the published
# four-GPU launcher instead uses DeepSpeed ZeRO-2, FlashAttention 2 and vLLM.
exec "$root/envs/copsd-official-20261015/bin/accelerate" launch \
  --multi_gpu --num_processes 4 --mixed_precision bf16 \
  --gradient_accumulation_steps 8 --main_process_port 12971 \
  multilingual_opsd_train.py \
  --train_language "${lang^^}" --translated_data_path "$data" \
  --model_name_or_path "$root/models/Qwen3-1.7B" \
  --learning_rate 5e-6 --max_grad_norm 0.1 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
  --gradient_checkpointing --max_steps 1 --save_steps 1 --logging_steps 1 \
  --output_dir "$output" --run_config "$run_id" \
  --max_completion_length 2048 --max_length 20000 \
  --attn_implementation sdpa --torch_dtype bfloat16 \
  --beta 0 --lmbda 1 \
  --use_peft --lora_r 64 --lora_alpha 128 \
  --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --temperature 1.1 --top_p 0.95 --top_k 20 \
  --fixed_teacher --student_enable_thinking --jsd_token_clip 0.05 \
  --report_to tensorboard --seed 42 --data_seed 42 \
  --wandb_project COPSD-four-gpu-smoke
