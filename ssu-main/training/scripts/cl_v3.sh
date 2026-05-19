#!/usr/bin/env bash
set -e

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cl
export HF_ENDPOINT=https://hf-mirror.com

BASE_MODEL="/home/HwHiAiUser/cl_workspace/var/hf_cache/Qwen2.5-1.5B-Instruct"

LANG="$1"
PREV_MODEL="$2"


if [ -z "$LANG" ]; then
  echo "用法：bash training/scripts/cl_v3.sh <lang> [prev_model] [run_name]"
  echo "示例1（single）: bash training/scripts/cl_v3.sh ibo_Latn"
  echo "示例2（CL）    : bash training/scripts/cl_v3.sh hau_Latn /data/.../cl_15b_v3/default/ibo_Latn"
  echo "示例3（自定义）: bash training/scripts/cl_v3.sh hau_Latn /data/.../some_prev_model myrun"
  exit 1
fi

if [ -n "$PREV_MODEL" ]; then
  MODEL_NAME="$PREV_MODEL"
  TRAIN_MODE="continual"
else
  MODEL_NAME="$BASE_MODEL"
  TRAIN_MODE="single"
fi

DATA_ROOT="$HOME/cl_workspace/data/fineweb2_cpt_ssu"
HF_CACHE="$HOME/cl_workspace/var/hf_cache"

# 新目录：不覆盖你已有的 single_15b / fft_ssu_seq_15b_v2 等
OUT_ROOT="/data/HwHiAiUser/cl_workspace/ckpt/single_15b_v3"

TRAIN_DIR="${DATA_ROOT}/${LANG}/train5k"
TEST_DIR="${DATA_ROOT}/${LANG}/test"

OUTPUT_DIR="${OUT_ROOT}/${LANG}"
LOG_DIR="${OUTPUT_DIR}/logs"

mkdir -p "${HF_CACHE}" "${OUTPUT_DIR}" "${LOG_DIR}"

echo "========================================"
echo "CL训练（FineWeb2, 1.5B, v3）"
echo "TRAIN_MODE=${TRAIN_MODE}"
echo "LANG=${LANG}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "BASE_MODEL=${BASE_MODEL}"
echo "TRAIN_DIR=${TRAIN_DIR}"
echo "TEST_DIR=${TEST_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "RUN_NAME=${RUN_NAME}"
echo "per_device_train_batch_size=1"
echo "gradient_accumulation_steps=2"
echo "learning_rate=1e-5"
echo "num_train_epochs=1"
echo "========================================"

cd /home/HwHiAiUser/cl_workspace/code/SSU/ssu-main/training/src

python main.py \
  --dataset_path "${TRAIN_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --logging_dir "${LOG_DIR}" \
  --model_name_or_path "${MODEL_NAME}" \
  --tokenizer_name_or_path "${BASE_MODEL}" \
  --cache_dir "${HF_CACHE}" \
  --seed 42 \
  --do_train \
  --evaluation_strategy "no" \
  --weight_decay 0.01 \
  --warmup_ratio 0.05 \
  --prediction_loss_only \
  --lr_scheduler_type cosine \
  --disable_tqdm True \
  --label_names labels \
  --remove_unused_columns True \
  --save_strategy epoch \
  --save_total_limit 1 \
  --num_train_epochs 1 \
  --logging_steps 10 \
  --gradient_accumulation_steps 2 \
  --per_rate 1e-5 \
  --max_grad_device_train_batch_size 1 \
  --learning_norm 1.0

echo "✅ 训练结束，开始评测 PPL ..."

cd /home/HwHiAiUser/cl_workspace/code/SSU/ssu-main

python evaluation/src/eval_ppl_fineweb2.py \
  --model_name "${OUTPUT_DIR}" \
  --data_root "${DATA_ROOT}" \
  --langs "${LANG}" \
  --batch_size 8

echo "🎉 ${LANG} cl_v3 训练 + 测试完成"