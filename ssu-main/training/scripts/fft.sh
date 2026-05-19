#!/bin/bash

# source /path/to/envs/ssu_train/bin/activate

# Configs
export TRANSFORMERS_VERBOSITY=info
export HF_HOME="./my_data/cache"
export HF_HUB_CACHE="./my_data/cache"
export HF_DATASETS_CACHE="./my_data/cache"
export HF_DATASETS_TRUST_REMOTE_CODE=true


# model_abbrev="OLMo-2-1124-7B-Instruct"
MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
MODEL_ABBREV="Qwen2.5-0.5B-Instruct"

LANG_CODE="$1"   # ig
#锁定项目根目录（运行 bash training/scripts/fft.sh 时的目录）
ROOT_DIR="$(pwd)"
# DATASET_DIR="${ROOT_DIR}/my_data/processed/${MODEL_ABBREV}_cpt_${LANG_CODE}"
# OUTPUT_DIR="${ROOT_DIR}/outputs/fft/${MODEL_ABBREV}-${LANG_CODE}"
# LOGGING_DIR="${ROOT_DIR}/logs/fft/${MODEL_ABBREV}-${LANG_CODE}"
DATASET_DIR="${ROOT_DIR}/my_data/processed/${MODEL_ABBREV}_cpt_${LANG_CODE}_fineweb2"
OUTPUT_DIR="${ROOT_DIR}/outputs/fft_fineweb2/${MODEL_ABBREV}-${LANG_CODE}"
LOGGING_DIR="${ROOT_DIR}/logs/fft_fineweb2/${MODEL_ABBREV}-${LANG_CODE}"




mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOGGING_DIR}"

# --gradient_accumulation_steps 1 \
# --per_device_train_batch_size 32 \
# --logging_steps 0.001 \
# --max_steps 12208 \
# --optim adamw_apex_fused \
# --bf16 \
# --gradient_checkpointing True
# --save_steps 0.5 \
#   --learning_rate 5e-5 \

cd training/src
python main.py \
    --dataset_path "${DATASET_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --logging_dir "${LOGGING_DIR}" \
    --model_name_or_path "${MODEL_NAME}" \
    --tokenizer_name_or_path "${MODEL_NAME}" \
    --cache_dir "../../my_data/cache" \
    --seed 42 \
    --do_train \
    --eval_strategy no \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --prediction_loss_only \
    --overwrite_output_dir \
    --lr_scheduler_type cosine \
    --disable_tqdm True \
    --label_names labels \
    --remove_unused_columns True \
    --save_strategy steps \
    --save_steps 50 \
    --max_steps 200 \
    --logging_steps 1 \
    --gradient_accumulation_steps 8 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-5 \
    --max_grad_norm 1.0 \
    # --fp16 False \
    # --bf16 False \
    # --logging_steps 10 \
    # --learning_rate 1e-5 \
   # --max_steps 200 \
    # --per_device_train_batch_size 1 \
    # --gradient_accumulation_steps 8 \
    # --max_grad_norm 1.0