#!/bin/bash
set -e

echo "===== ER training (Qwen2.5-0.5B, IG, Experience Replay) ====="

# =========================
# 0. 环境与缓存
# =========================
export TRANSFORMERS_VERBOSITY=info
export HF_HOME="./my_data/cache"
export HF_HUB_CACHE="./my_data/cache"
export HF_DATASETS_CACHE="./my_data/cache"
export HF_DATASETS_TRUST_REMOTE_CODE=true

export CUDA_VISIBLE_DEVICES=0

# =========================
# 1. 基本配置（对齐 ssu.sh）
# =========================
ROOT_DIR="$(pwd)"

MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
MODEL_ABBREV="Qwen2.5-0.5B-Instruct"
LANG_CODE="ig"

DATASET_DIR="${ROOT_DIR}/my_data/processed/${MODEL_ABBREV}_cpt_${LANG_CODE}"

OUTPUT_DIR="${ROOT_DIR}/outputs/er/${MODEL_ABBREV}-${LANG_CODE}"
LOGGING_DIR="${ROOT_DIR}/logs/er/${MODEL_ABBREV}-${LANG_CODE}"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOGGING_DIR}"

echo "Dataset: ${DATASET_DIR}"
echo "Output : ${OUTPUT_DIR}"
echo "Logs   : ${LOGGING_DIR}"

# =========================
# 2. 进入 training/src
# =========================
cd training/src || exit 1

# =========================
# 3. 启动 ER 训练
# =========================
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
    --learning_rate 2e-5 \
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
    --cl_method er \
    --replay_ratio 0.5 \
    --replay_buffer_size 2000

echo "===== ER training DONE ====="
