#!/usr/bin/env bash
set -e

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cl

export HF_ENDPOINT=https://hf-mirror.com

# ==========================================
# EWC Continual Learning Script
# Task1: ibo_Latn   -> 直接用 single，不走 ewc
# Task2: hau_Latn
# Task3: kir_Cyrl
# Task4: npi_Deva
# Task5: amh_Ethi
#
# 用法：
# bash training/scripts/ewc.sh <LANG> <PREV_MODEL>
# ==========================================

BASE_MODEL="/data/HwHiAiUser/cl_workspace/Qwen2.5-3B-Instruct"

LANG="$1"
PREV_MODEL="$2"

if [ -z "$LANG" ] || [ -z "$PREV_MODEL" ]; then
  echo "用法：bash training/scripts/ewc.sh <LANG> <PREV_MODEL>"
  exit 1
fi

DATA_ROOT="/data/HwHiAiUser/cl_workspace/data/fineweb2_cpt_ssu"

TRAIN_DIR="${DATA_ROOT}/${LANG}/train5k"
TEST_DIR="${DATA_ROOT}/${LANG}/test"

OUT_ROOT="/data/HwHiAiUser/cl_workspace/ckpt/ewc_seq_3b"
OUTPUT_DIR="${OUT_ROOT}/${LANG}"
LOG_DIR="${OUTPUT_DIR}/logs"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

# ===============================
# EWC 超参数
# ===============================
EWC_LAMBDA=10.0
EWC_FISHER_MAX_BATCHES=100

# 训练超参数
LEARNING_RATE=5e-6
GRAD_ACC=8
BATCH_SIZE=1
NUM_EPOCHS=1

echo "========================================"
echo "CL训练（FineWeb2）- EWC"
echo "LANG=${LANG}"
echo "TRAIN_DIR=${TRAIN_DIR}"
echo "TEST_DIR=${TEST_DIR}"
echo "MODEL_NAME_OR_PATH=${PREV_MODEL}"
echo "TOKENIZER_NAME_OR_PATH=${BASE_MODEL}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "EWC_LAMBDA=${EWC_LAMBDA}"
echo "EWC_FISHER_MAX_BATCHES=${EWC_FISHER_MAX_BATCHES}"
echo "LEARNING_RATE=${LEARNING_RATE}"
echo "GRAD_ACC=${GRAD_ACC}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "NUM_EPOCHS=${NUM_EPOCHS}"
echo "========================================"

cd training/src

python main.py \
  --dataset_path "${TRAIN_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --logging_dir "${LOG_DIR}" \
  --model_name_or_path "${PREV_MODEL}" \
  --tokenizer_name_or_path "${BASE_MODEL}" \
  --cache_dir "$HOME/cl_workspace/var/hf_cache" \
  --seed 42 \
  --do_train \
  --optim adamw_torch \
  --eval_strategy "no" \
  --prediction_loss_only \
  --overwrite_output_dir \
  --lr_scheduler_type cosine \
  --weight_decay 0.01 \
  --warmup_ratio 0.05 \
  --disable_tqdm True \
  --label_names labels \
  --remove_unused_columns True \
  --save_strategy epoch \
  --save_total_limit 1 \
  --num_train_epochs "${NUM_EPOCHS}" \
  --logging_steps 10 \
  --gradient_accumulation_steps "${GRAD_ACC}" \
  --per_device_train_batch_size "${BATCH_SIZE}" \
  --learning_rate "${LEARNING_RATE}" \
  --max_grad_norm 1.0 \
  --cl_method ewc \
  --ewc_lambda "${EWC_LAMBDA}" \
  --ewc_fisher_max_batches "${EWC_FISHER_MAX_BATCHES}" \
  --ewc_save_state

echo "✅ EWC 训练结束，开始评测 PPL ..."

cd ../../

python evaluation/src/eval_ppl_fineweb2.py \
  --model_name "${OUTPUT_DIR}" \
  --data_root "${DATA_ROOT}" \
  --langs "${LANG}" \
  --batch_size 8 \
  --max_test_blocks 5000

echo "🎉 ${LANG} EWC 训练 + 测试完成"