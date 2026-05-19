#!/usr/bin/env bash
set -e
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cl
export HF_ENDPOINT=https://hf-mirror.com

# Task1: ibo_Latn   (Igbo)
# Task2: hau_Latn   (Hausa)
# Task3: kir_Cyrl  (Kyrgyz)
# Task4: npi_Deva  (Nepali)
# Task5: amh_Ethi  (Amharic)



# 用法：
# su - HwHiAiUser 
# conda activate cl
# cd ~/cl_workspace/code/SSU/ssu-main
# bash training/scripts/cl.sh ibo_Latn D:/ckpt_single_fineweb2/swh_Latn


# BASE_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
BASE_MODEL="/home/HwHiAiUser/cl_workspace/var/hf_cache/Qwen2.5-1.5B-Instruct"
LANG="$1"
PREV_MODEL="$2"

if [ -n "$PREV_MODEL" ]; then
  MODEL_NAME="$PREV_MODEL"
else
  MODEL_NAME="$BASE_MODEL"
fi

if [ -z "$LANG" ]; then
  echo "用法：bash training/scripts/single_fineweb2.sh <lang>"
  exit 1
fi

# === FineWeb2 处理后数据根目录===
DATA_ROOT="$HOME/cl_workspace/data/fineweb2_cpt_ssu"   # 数据在 /cl_workspace/data 下
HF_CACHE="$HOME/cl_workspace/var/hf_cache"       
OUT_ROOT="/data/HwHiAiUser/cl_workspace/ckpt/cl_lowLR"

# 训练/测试目录
TRAIN_DIR="${DATA_ROOT}/${LANG}/train5k"
TEST_DIR="${DATA_ROOT}/${LANG}/test"

# === 输出目录（每个语言一个 checkpoint 目录）===
OUTPUT_DIR="${OUT_ROOT}/${LANG}"
LOG_DIR="${OUTPUT_DIR}/logs"

mkdir -p "${HF_CACHE}" "${OUTPUT_DIR}" "${LOG_DIR}"

echo "========================================"
echo "CL训练（FineWeb2, 1.5B，降低LR）"
echo "LANG=${LANG}"
echo "TRAIN_DIR=${TRAIN_DIR}"
echo "TEST_DIR=${TEST_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "========================================"

# 训练
cd training/src

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
  --save_strategy no \
  --num_train_epochs 1 \
  --logging_steps 10 \
  --gradient_accumulation_steps 2 \
  --per_device_train_batch_size 1 \
  --learning_rate 5e-6 \
  --max_grad_norm 1.0

echo "✅ 训练结束，开始评测 PPL ..."

# 跑 evaluation
cd /home/HwHiAiUser/cl_workspace/code/SSU/ssu-main

python evaluation/src/eval_ppl_fineweb2.py \
  --model_name "${OUTPUT_DIR}" \
  --data_root "${DATA_ROOT}" \
  --langs "${LANG}" \
  --batch_size 8 


echo "🎉 ${LANG} 单任务训练 + 测试完成"
