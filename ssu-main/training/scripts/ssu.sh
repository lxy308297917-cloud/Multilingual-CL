#!/bin/bash

# source /path/to/envs/ssu_train/bin/activate

# Configs
export TRANSFORMERS_VERBOSITY=info
export HF_HOME="./my_data/cache"
export HF_HUB_CACHE="./my_data/cache"
export HF_DATASETS_CACHE="./my_data/cache"
export HF_DATASETS_TRUST_REMOTE_CODE=true
# model_abbrev="OLMo-2-1124-7B-Instruct"
# lang_code="$1"
MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
MODEL_ABBREV="Qwen2.5-0.5B-Instruct"
LANG_CODE="ig"

ROOT_DIR="$(pwd)"
DATASET_DIR="${ROOT_DIR}/my_data/processed/${MODEL_ABBREV}_cpt_${LANG_CODE}"
CALIB_DIR="${ROOT_DIR}/my_data/processed/${MODEL_ABBREV}_calib_tulu"
OUTPUT_DIR="${ROOT_DIR}/outputs/ssu/${MODEL_ABBREV}-${LANG_CODE}"
LOGGING_DIR="${ROOT_DIR}/logs/ssu/${MODEL_ABBREV}-${LANG_CODE}"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOGGING_DIR}"

cd training/src || exit 1

# --max_steps 12208 \
#  --save_steps 0.5 \
#  --logging_steps 0.001 \
# --learning_rate 5e-5 \
#     --weight_decay 0.01 \
#     --warmup_ratio 0.05 \
#     --max_steps 100 \
#     --per_device_train_batch_size 32 \
#     --gradient_accumulation_steps 1 \
# --optim adamw_apex_fused \

python main.py \
    --dataset_path "${DATASET_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --logging_dir "${LOGGING_DIR}" \
    --model_name_or_path "${MODEL_NAME}" \
    --tokenizer_name_or_path "${MODEL_NAME}" \
    --cache_dir "../../my_data/cache" \
    --optim adamw_torch \
    --seed 42 \
    --eval_strategy no \
    --logging_steps 10 \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --max_steps 100 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_grad_norm 1.0 \
    --prediction_loss_only \
    --overwrite_output_dir \
    --do_train \
    --lr_scheduler_type cosine \
    --disable_tqdm True \
    --label_names labels \
    --remove_unused_columns True \
    --save_strategy steps \
    --save_steps 50 \
    --do_hft \
    --freeze_ratio 0.5 \
    --freeze_strategy "ssu_based" \
    --skip_embeddings_and_head \
    --freeze_chat_template_tokens \
    --calibration_dataset_path "${CALIB_DIR}" \
    --num_calibration_samples 500 \
    --calibration_max_length 2048
    # --bf16 \
    # --gradient_checkpointing True \