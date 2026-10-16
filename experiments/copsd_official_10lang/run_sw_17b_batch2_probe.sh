#!/usr/bin/env bash
set -euo pipefail
test "${GPU_QUEUE_GPU_COUNT:-}" = 1
test -n "${CUDA_VISIBLE_DEVICES:-}"
root=/data0/users/lixiaoyang
test -f "$root/models/Qwen3-1.7B/MODEL_LOCK.json"
test -s "$root/data/copsd_official_20261015/adapted_v2_ordered/project_train_2900_sw.json"
output="$root/models/checkpoint/copsd_official_17b_batch2_probe_v1"
test ! -e "$output"
export COPSD_CACHE_ROOT="$root/cache/copsd_official"
export WANDB_MODE=offline
export WANDB_DIR="$root/logs/copsd_official_10lang/wandb"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
mkdir -p "$COPSD_CACHE_ROOT" "$WANDB_DIR"
cd "$root/COPSD-local-adapter-20261015"
exec "$root/envs/copsd-official-20261015/bin/python" multilingual_opsd_train.py \
  --train_language SW \
  --translated_data_path "$root/data/copsd_official_20261015/adapted_v2_ordered/project_train_2900_sw.json" \
  --model_name_or_path "$root/models/Qwen3-1.7B" \
  --learning_rate 5e-6 --max_grad_norm 0.1 \
  --per_device_train_batch_size 2 --gradient_accumulation_steps 1 \
  --gradient_checkpointing --max_steps 1 --save_steps 1 --logging_steps 1 \
  --output_dir "$output" --run_config sw_17b_batch2_probe_v1 \
  --max_completion_length 2048 --max_length 20000 \
  --attn_implementation sdpa --torch_dtype bfloat16 \
  --beta 0 --lmbda 1 \
  --use_peft --lora_r 64 --lora_alpha 128 \
  --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --temperature 1.1 --top_p 0.95 --top_k 20 \
  --fixed_teacher --student_enable_thinking --jsd_token_clip 0.05 \
  --report_to tensorboard --seed 42 --data_seed 42 \
  --wandb_project COPSD-official-local
