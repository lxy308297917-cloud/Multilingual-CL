#!/usr/bin/env bash
set -euo pipefail
test "${GPU_QUEUE_GPU_COUNT:-}" = 1
test -n "${CUDA_VISIBLE_DEVICES:-}"
lang="${1:?usage: run_single_17b_project.sh LANG RUN_ID [CHECKPOINT]}"
run_id="${2:?run ID required}"
resume="${3:-}"
case "$lang" in sw|te|th|bn|zh|ja|ru|de|fr|es) ;; *) echo "invalid language: $lang" >&2; exit 2 ;; esac
[[ "$run_id" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "invalid run ID" >&2; exit 2; }
root=/data0/users/lixiaoyang
test -f "$root/models/Qwen3-1.7B/MODEL_LOCK.json"
data="$root/data/copsd_official_20261015/adapted_v2_ordered/project_train_2900_${lang}.json"
test -s "$data"
output="$root/models/checkpoint/copsd_official_17b_single_v1/$run_id"
if [[ -z "$resume" ]]; then
  test ! -e "$output"
else
  test -d "$resume"
  test -d "$output"
fi
export COPSD_CACHE_ROOT="$root/cache/copsd_official"
export WANDB_MODE=offline
export WANDB_DIR="$root/logs/copsd_official_10lang/wandb"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
mkdir -p "$COPSD_CACHE_ROOT" "$WANDB_DIR"
args=()
if [[ -n "$resume" ]]; then args+=(--resume_from_checkpoint "$resume"); fi
cd "$root/COPSD-local-adapter-20261015"
exec "$root/envs/copsd-official-20261015/bin/python" multilingual_opsd_train.py \
  --train_language "${lang^^}" --translated_data_path "$data" \
  --model_name_or_path "$root/models/Qwen3-1.7B" \
  --learning_rate 5e-6 --max_grad_norm 0.1 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 32 \
  --gradient_checkpointing --max_steps 100 --save_steps 5 --logging_steps 2 \
  --output_dir "$output" --run_config "$run_id" \
  --max_completion_length 2048 --max_length 20000 \
  --attn_implementation sdpa --torch_dtype bfloat16 \
  --beta 0 --lmbda 1 \
  --use_peft --lora_r 64 --lora_alpha 128 \
  --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --temperature 1.1 --top_p 0.95 --top_k 20 \
  --fixed_teacher --student_enable_thinking --jsd_token_clip 0.05 \
  --report_to tensorboard --seed 42 --data_seed 42 \
  --wandb_project COPSD-official-local "${args[@]}"
