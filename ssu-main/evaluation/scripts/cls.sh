#!/bin/bash
set -e

echo "===== MMLU evaluation ====="

# =====================
# 0. 基础环境
# =====================
export TRANSFORMERS_VERBOSITY=info
export HF_HOME="$(pwd)/my_data/cache"
export HF_HUB_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"
export HF_DATASETS_TRUST_REMOTE_CODE=true

# 强制使用 GPU
export CUDA_VISIBLE_DEVICES=0

ROOT_DIR=$(pwd)
LOG_BASE_DIR="${ROOT_DIR}/evaluation/logs_cls"
mkdir -p "${LOG_BASE_DIR}"

# =====================
# 模型：FFT checkpoint-100
# =====================
# MODEL_PATH="${ROOT_DIR}/outputs/fft/Qwen2.5-0.5B-Instruct-ig/checkpoint-100"
# MODEL_TAG="fft_step100"

MODEL_PATH="${ROOT_DIR}/outputs/ssu/Qwen2.5-0.5B-Instruct-ig/checkpoint-100"
MODEL_TAG="ssu_step100"

# =====================
#  任务MMLU
# =====================
TASK_MMLU="leaderboard|mmlu:abstract_algebra|5|0,\
leaderboard|mmlu:computer_security|5|0,\
leaderboard|mmlu:high_school_mathematics|5|0"

OUT_DIR="${LOG_BASE_DIR}/${MODEL_TAG}/mmlu"
mkdir -p "${OUT_DIR}"


echo "Using model: ${MODEL_PATH}"
echo "Saving logs to: ${OUT_DIR}"

# =====================
#  运行
# =====================
lighteval accelerate \
  "model_name=${MODEL_PATH},batch_size=4,dtype=float16" \
  "${TASK_MMLU}" \
  --output-dir "${OUT_DIR}"

echo "===== DONE ====="

 ========= 同步结果文件到 logs 目录 =========
RESULT_JSON=$(ls "${MODEL_PATH}"/results_*.json | tail -n 1)

if [ -f "$RESULT_JSON" ]; then
  cp "$RESULT_JSON" "${OUT_DIR}/"
  echo "Copied result json to ${OUT_DIR}"
else
  echo "WARNING: result json not found in ${MODEL_PATH}"
fi