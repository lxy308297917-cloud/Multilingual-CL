#!/usr/bin/env bash
set -euo pipefail
export ASCEND_LAUNCH_BLOCKING=1

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cl
export HF_ENDPOINT=https://hf-mirror.com

# 用法：
# su - HwHiAiUser 
# cd /home/HwHiAiUser/cl_workspace/code/SSU/ssu-main
# bash training/scripts/lora.sh ibo_Latn train5k
# bash training/scripts/lora.sh hau_Latn train5k /data/HwHiAiUser/cl_workspace/ckpt/single_15b_lora/ibo_Latn_train5k

BASE_MODEL="/home/HwHiAiUser/cl_workspace/var/hf_cache/Qwen2.5-1.5B-Instruct"

LANG="${1:-}"
TRAIN_SPLIT_NAME="${2:-train5k}"
PREV_MODEL="${3:-}"

if [ -z "${LANG}" ]; then
  echo "用法: bash training/scripts/lora.sh <lang> [train_split_name] [prev_model]"
  echo "示例1: bash training/scripts/lora.sh ibo_Latn train5k"
  echo "示例2: bash training/scripts/lora.sh hau_Latn train5k /data/HwHiAiUser/cl_workspace/ckpt/single_15b_lora/ibo_Latn_train5k"
  exit 1
fi

if [ -n "${PREV_MODEL}" ]; then
  MODEL_NAME="${PREV_MODEL}"
else
  MODEL_NAME="${BASE_MODEL}"
fi

DATA_ROOT="/data/HwHiAiUser/cl_workspace/data/fineweb2_cpt_ssu"
HF_CACHE="/home/HwHiAiUser/cl_workspace/var/hf_cache"

OUT_ROOT="/data/HwHiAiUser/cl_workspace/ckpt/single_15b_lora"
RUN_LOG_ROOT="/home/HwHiAiUser/cl_workspace/logs/single_15b_lora"

TRAIN_DIR="${DATA_ROOT}/${LANG}/${TRAIN_SPLIT_NAME}"
TEST_DIR="${DATA_ROOT}/${LANG}/test"

OUTPUT_DIR="${OUT_ROOT}/${LANG}_${TRAIN_SPLIT_NAME}"
LOG_DIR="${OUTPUT_DIR}/trainer_logs"
RUN_LOG="${RUN_LOG_ROOT}/${LANG}_${TRAIN_SPLIT_NAME}.log"

mkdir -p "${HF_CACHE}" "${OUTPUT_DIR}" "${LOG_DIR}" "${RUN_LOG_ROOT}"

echo "========================================"
echo "Single LoRA训练（Qwen2.5-1.5B-Instruct）"
echo "LANG=${LANG}"
echo "TRAIN_SPLIT_NAME=${TRAIN_SPLIT_NAME}"
echo "BASE_MODEL=${BASE_MODEL}"
echo "PREV_MODEL=${PREV_MODEL:-None}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "TRAIN_DIR=${TRAIN_DIR}"
echo "TEST_DIR=${TEST_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "LOG_DIR=${LOG_DIR}"
echo "RUN_LOG=${RUN_LOG}"
echo "========================================"

if [ ! -d "${TRAIN_DIR}" ]; then
  echo "错误: TRAIN_DIR 不存在: ${TRAIN_DIR}"
  exit 1
fi

if [ ! -d "${TEST_DIR}" ]; then
  echo "错误: TEST_DIR 不存在: ${TEST_DIR}"
  exit 1
fi

echo "===== 查看数据目录 ====="
ls -lh "${DATA_ROOT}/${LANG}"
echo "===== 开始训练 ====="

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
  --evaluation_strategy no \
  --weight_decay 0.01 \
  --warmup_ratio 0.05 \
  --prediction_loss_only \
  --lr_scheduler_type cosine \
  --disable_tqdm True \
  --label_names labels \
  --remove_unused_columns True \
  --save_strategy no \
  --num_train_epochs 1 \
  --logging_steps 10 \
  --gradient_accumulation_steps 2 \
  --per_device_train_batch_size 1 \
  --learning_rate 5e-5 \
  --max_grad_norm 1.0 \
  --peft_method lora \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
  2>&1 | tee "${RUN_LOG}"

echo "✅ 训练结束，开始评测 PPL ..."

cd /home/HwHiAiUser/cl_workspace/code/SSU/ssu-main

python evaluation/src/eval_ppl_fineweb2.py \
  --model_name "${OUTPUT_DIR}" \
  --data_root "${DATA_ROOT}" \
  --langs "${LANG}" \
  --batch_size 8 \
  2>&1 | tee -a "${RUN_LOG}"

echo "🎉 ${LANG} ${TRAIN_SPLIT_NAME} LoRA训练 + 测试完成"