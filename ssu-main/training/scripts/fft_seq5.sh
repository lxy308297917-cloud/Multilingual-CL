#!/bin/bash
set -e

# =========================
# HF / datasets 缓存设置
# =========================
export TRANSFORMERS_VERBOSITY=info
export HF_HOME="./my_data/cache"
export HF_HUB_CACHE="./my_data/cache"
export HF_DATASETS_CACHE="./my_data/cache"
export HF_DATASETS_TRUST_REMOTE_CODE=true

# =========================
# 模型
# =========================
MODEL_ABBREV="Qwen2.5-0.5B-Instruct"
MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"

# =========================
# 语言 / 任务顺序
# =========================
LANGS=("ig" "ha" "ky" "ne" "am")

# =========================
# 项目根目录
# =========================
ROOT_DIR="$(pwd)"

# =========================
# 输出根目录（避免覆盖）
# =========================
OUT_ROOT="${ROOT_DIR}/outputs/fft_fineweb2_seq/${MODEL_ABBREV}"
LOG_ROOT="${ROOT_DIR}/logs/fft_fineweb2_seq/${MODEL_ABBREV}"
mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

# =========================
# 训练超参
# =========================
SEED=42
MAX_STEPS=200
LR=1e-5
BATCH=1
GRAD_ACC=8

# =========================
# 连续学习方法
# none / replay / lwf / ewc
# =========================
CL_METHOD="none"

# =========================
# 进入 training/src
# =========================
cd training/src

# =========================
# 顺序训练循环
# =========================
for i in "${!LANGS[@]}"; do
  LANG="${LANGS[$i]}"

  DATASET_DIR="${ROOT_DIR}/my_data/processed/${MODEL_ABBREV}_cpt_${LANG}_fineweb2"
  OUTPUT_DIR="${OUT_ROOT}/task_$((i+1))_${LANG}"
  LOGGING_DIR="${LOG_ROOT}/task_$((i+1))_${LANG}"

  mkdir -p "${OUTPUT_DIR}" "${LOGGING_DIR}"

  echo "========================================"
  echo "Task $((i+1)) / ${#LANGS[@]} | 语言=${LANG}"
  echo "模型输入：${MODEL_NAME}"
  echo "数据集：${DATASET_DIR}"
  echo "输出目录：${OUTPUT_DIR}"
  echo "========================================"

  python main.py \
    --dataset_path "${DATASET_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --logging_dir "${LOGGING_DIR}" \
    --model_name_or_path "${MODEL_NAME}" \
    --tokenizer_name_or_path "${MODEL_NAME}" \
    --cache_dir "../../my_data/cache" \
    --seed ${SEED} \
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
    --save_steps 50 \
    --max_steps ${MAX_STEPS} \
    --logging_steps 1 \
    --gradient_accumulation_steps ${GRAD_ACC} \
    --per_device_train_batch_size ${BATCH} \
    --learning_rate ${LR} \
    --max_grad_norm 1.0 \
    --cl_method "${CL_METHOD}"

  # =========================
  # 下一任务从当前任务的输出继续训练
  # =========================
  MODEL_NAME="${OUTPUT_DIR}"

done

echo " 顺序 5 语言 FFT 训练完成"
