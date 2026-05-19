#!/usr/bin/env bash
set -euo pipefail

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cl

export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=""
export ASCEND_RT_VISIBLE_DEVICES=6

cd /home/HwHiAiUser/cl_workspace/code/SSU/ssu-main

BASE_MODEL="/data/HwHiAiUser/cl_workspace/Qwen2.5-3B-Instruct"
DATA_ROOT="/data/HwHiAiUser/cl_workspace/data/fineweb2_cpt_ssu"
HF_CACHE="$HOME/cl_workspace/var/hf_cache"

OUT_ROOT="/data/HwHiAiUser/cl_workspace/ckpt/replay_seq_3b"
RUN_LOG_DIR="/home/HwHiAiUser/cl_workspace/logs/replay_seq_3b_npu5"

mkdir -p "${HF_CACHE}" "${OUT_ROOT}" "${RUN_LOG_DIR}"

LANGS=(
  "ibo_Latn"
  "hau_Latn"
  "kir_Cyrl"
  "npi_Deva"
  "amh_Ethi"
)

PREV_MODEL=""
REPLAY_DATASETS=""

REPLAY_RATIO="0.1"
REPLAY_SEED="42"

for LANG in "${LANGS[@]}"; do
  if [ -n "${PREV_MODEL}" ]; then
    MODEL_NAME="${PREV_MODEL}"
  else
    MODEL_NAME="${BASE_MODEL}"
  fi

  TRAIN_DIR="${DATA_ROOT}/${LANG}/train5k"
  TEST_DIR="${DATA_ROOT}/${LANG}/test"
  OUTPUT_DIR="${OUT_ROOT}/${LANG}"
  LOG_DIR="${OUTPUT_DIR}/logs"
  RUN_LOG="${RUN_LOG_DIR}/${LANG}.log"

  mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

  {
    echo "========================================"
    echo "Replay-SEQ训练（FineWeb2, 3B）"
    echo "LANG=${LANG}"
    echo "NPU=${ASCEND_RT_VISIBLE_DEVICES}"
    echo "MODEL_NAME=${MODEL_NAME}"
    echo "BASE_MODEL=${BASE_MODEL}"
    echo "TRAIN_DIR=${TRAIN_DIR}"
    echo "TEST_DIR=${TEST_DIR}"
    echo "OUTPUT_DIR=${OUTPUT_DIR}"
    echo "LOG_DIR=${LOG_DIR}"
    echo "REPLAY_DATASETS=${REPLAY_DATASETS:-None}"
    echo "REPLAY_RATIO=${REPLAY_RATIO}"
    echo "per_device_train_batch_size=1"
    echo "gradient_accumulation_steps=8"
    echo "learning_rate=1e-5"
    echo "num_train_epochs=1"
    echo "========================================"

    cd /home/HwHiAiUser/cl_workspace/code/SSU/ssu-main/training/src

    if [ -z "${REPLAY_DATASETS}" ]; then
      echo "当前是 Task1，不使用 replay，等价于普通 FFT"

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
        --overwrite_output_dir \
        --lr_scheduler_type cosine \
        --disable_tqdm True \
        --label_names labels \
        --remove_unused_columns True \
        --save_strategy epoch \
        --save_total_limit 1 \
        --num_train_epochs 1 \
        --logging_steps 10 \
        --gradient_accumulation_steps 8 \
        --per_device_train_batch_size 1 \
        --learning_rate 1e-5 \
        --max_grad_norm 1.0

    else
      echo "当前是 Task2+，启用 replay"

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
        --overwrite_output_dir \
        --lr_scheduler_type cosine \
        --disable_tqdm True \
        --label_names labels \
        --remove_unused_columns True \
        --save_strategy epoch \
        --save_total_limit 1 \
        --num_train_epochs 1 \
        --logging_steps 10 \
        --gradient_accumulation_steps 8 \
        --per_device_train_batch_size 1 \
        --learning_rate 1e-5 \
        --max_grad_norm 1.0 \
        --cl_method replay \
        --replay_dataset_path "${REPLAY_DATASETS}" \
        --replay_ratio "${REPLAY_RATIO}" \
        --replay_seed "${REPLAY_SEED}"
    fi

    echo "✅ ${LANG} 训练结束，开始评测 PPL ..."

    cd /home/HwHiAiUser/cl_workspace/code/SSU/ssu-main

    python evaluation/src/eval_ppl_fineweb2.py \
      --model_name "${OUTPUT_DIR}" \
      --data_root "${DATA_ROOT}" \
      --langs "${LANG}" \
      --batch_size 4

    echo "🎉 ${LANG} Replay-SEQ 训练 + PPL测试完成"

  } > "${RUN_LOG}" 2>&1

  PREV_MODEL="${OUTPUT_DIR}"

  if [ -z "${REPLAY_DATASETS}" ]; then
    REPLAY_DATASETS="${TRAIN_DIR}"
  else
    REPLAY_DATASETS="${REPLAY_DATASETS},${TRAIN_DIR}"
  fi

done

echo "🎉 五语言 Replay-SEQ 3B 全部完成"
echo "模型保存目录：${OUT_ROOT}"
echo "日志目录：${RUN_LOG_DIR}"
