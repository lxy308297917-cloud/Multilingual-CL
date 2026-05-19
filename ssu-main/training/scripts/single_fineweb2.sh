#!/usr/bin/env bash
set -e

# 用法：
# bash training/scripts/single_fineweb2.sh tel_Telu

MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
LANG="$1"
if [ -z "$LANG" ]; then
  echo "用法：bash training/scripts/single_fineweb2.sh <lang>"
  exit 1
fi

# === FineWeb2 处理后数据根目录（按你的实际）===
DATA_ROOT="D:/fineweb2_cpt"

# 训练/测试目录
TRAIN_DIR="${DATA_ROOT}/${LANG}/train5k"
TEST_DIR="${DATA_ROOT}/${LANG}/test"

# === 输出目录（每个语言一个 checkpoint 目录）===
OUT_ROOT="D:/ckpt_single_fineweb2"
OUTPUT_DIR="${OUT_ROOT}/${LANG}"
LOG_DIR="${OUTPUT_DIR}/logs"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "========================================"
echo "单任务训练（FineWeb2）"
echo "LANG=${LANG}"
echo "TRAIN_DIR=${TRAIN_DIR}"
echo "TEST_DIR=${TEST_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "========================================"

# 进入训练入口
cd training/src

python main.py \
  --dataset_path "${TRAIN_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --logging_dir "${LOG_DIR}" \
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
  --save_steps 250 \
  --max_steps 500 \
  --logging_steps 1 \
  --gradient_accumulation_steps 1 \
  --per_device_train_batch_size 1 \
  --learning_rate 1e-5 \
  --max_grad_norm 1.0

echo "✅ 训练结束，开始评测 PPL ..."

# 回到项目根目录再跑 evaluation
cd ../../

python evaluation/src/eval_ppl_fineweb2.py \
  --model_name "${OUTPUT_DIR}" \
  --data_root "${DATA_ROOT}" \
  --langs "${LANG}" \
  --batch_size 8 \
  --max_test_blocks 5000

echo "🎉 ${LANG} 单任务训练 + 测试完成"
