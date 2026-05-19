#!/usr/bin/env bash
set -e
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cl
export HF_ENDPOINT=https://hf-mirror.com

BASE_MODEL="/home/HwHiAiUser/cl_workspace/var/hf_cache/Qwen2.5-1.5B-Instruct"
LANG="$1"

if [ -z "$LANG" ]; then
  echo "用法：bash training/scripts/single_v2.sh <lang>"
  exit 1
fi

DATA_ROOT="$HOME/cl_workspace/data/fineweb2_cpt_ssu"
HF_CACHE="$HOME/cl_workspace/var/hf_cache"

# 新目录，避免覆盖旧 single_15b
OUT_ROOT="/data/HwHiAiUser/cl_workspace/ckpt/single_15b_v2"

TRAIN_DIR="${DATA_ROOT}/${LANG}/train5k"
TEST_DIR="${DATA_ROOT}/${LANG}/test"

OUTPUT_DIR="${OUT_ROOT}/${LANG}"
LOG_DIR="${OUTPUT_DIR}/logs"

mkdir -p "${HF_CACHE}" "${OUTPUT_DIR}" "${LOG_DIR}"

echo "========================================"
echo "Single训练（FineWeb2, 1.5B, v2）"
echo "LANG=${LANG}"
echo "TRAIN_DIR=${TRAIN_DIR}"
echo "TEST_DIR=${TEST_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "per_device_train_batch_size=2"
echo "gradient_accumulation_steps=4"
echo "learning_rate=1e-5"
echo "num_train_epochs=1"
echo "========================================"

cd training/src

python main.py \
  --dataset_path "${TRAIN_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --logging_dir "${LOG_DIR}" \
  --model_name_or_path "${BASE_MODEL}" \
  --tokenizer_name_or_path "${BASE_MODEL}" \
  --cache_dir "${HF_CACHE}" \
  --seed 42 \
  --do_train \
  --evaluation_strategy "no" \
  --weight_decay 0.01 \
  --warmup_ratio 0.05 \
  --prediction_loss_only \
  --overwrite_output_dir \
  --lr_scheduler_type cosine \
  --disable_tqdm True \
  --label_names labels \
  --remove_unused_columns True \
  --save_strategy epoch \
  --save_total_limit 1 \
  --num_train_epochs 1 \
  --logging_steps 10 \
  --gradient_accumulation_steps 4 \
  --per_device_train_batch_size 2 \
  --learning_rate 1e-5 \
  --max_grad_norm 1.0

echo "✅ 训练结束，开始评测 PPL ..."

cd ../../

python evaluation/src/eval_ppl_fineweb2.py \
  --model_name "${OUTPUT_DIR}" \
  --data_root "${DATA_ROOT}" \
  --langs "${LANG}" \
  --batch_size 8

echo "🎉 ${LANG} single v2 训练 + 测试完成"