#!/usr/bin/env bash
set -euo pipefail

# =========================
# AutoDL CUDA 环境
# =========================
export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com

# 减少 CUDA 显存碎片
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# conda 环境
source /root/miniconda3/etc/profile.d/conda.sh
conda activate cl

# =========================
# 参数
# =========================
LANG="${1:-}"
PREV_MODEL="${2:-}"

if [ -z "$LANG" ]; then
  echo "用法：bash training/scripts/cl_autodl_4090.sh <lang> [prev_model]"
  echo "例子：bash training/scripts/cl_autodl_4090.sh ibo_Latn"
  echo "例子：bash training/scripts/cl_autodl_4090.sh hau_Latn /root/autodl-tmp/ssu_workspace/ckpt/single_15b_newlangs/ibo_Latn"
  exit 1
fi

# =========================
# AutoDL 工作目录
# =========================
WORK_ROOT="/root/autodl-tmp/ssu_workspace"

# 模型路径：你可以改成本地模型路径，也可以直接用 HF 名称
BASE_MODEL="${WORK_ROOT}/models/Qwen2.5-1.5B-Instruct"

# 数据路径
DATA_ROOT="${WORK_ROOT}/data/fineweb2_cpt_ssu"

# HF 缓存
HF_CACHE="${WORK_ROOT}/hf_cache"

# 输出路径
OUT_ROOT="${WORK_ROOT}/ckpt/single_15b_newlangs"

# 代码路径
CODE_ROOT="/root/ssu-github/ssu-main"

# 如果传了上一个模型，就从上一个模型继续训练
if [ -n "$PREV_MODEL" ]; then
  MODEL_NAME="$PREV_MODEL"
else
  MODEL_NAME="$BASE_MODEL"
fi

TRAIN_DIR="${DATA_ROOT}/${LANG}/train5k"
TEST_DIR="${DATA_ROOT}/${LANG}/test"

OUTPUT_DIR="${OUT_ROOT}/${LANG}"
LOG_DIR="${OUTPUT_DIR}/logs"

mkdir -p "${HF_CACHE}" "${OUTPUT_DIR}" "${LOG_DIR}"

echo "========================================"
echo "AutoDL CUDA 单卡训练 FineWeb2 1.5B"
echo "LANG=${LANG}"
echo "PREV_MODEL=${PREV_MODEL:-None}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "BASE_MODEL=${BASE_MODEL}"
echo "TRAIN_DIR=${TRAIN_DIR}"
echo "TEST_DIR=${TEST_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "LOG_DIR=${LOG_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "========================================"

echo "===== GPU 检查 ====="
nvidia-smi || true

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
print("gpu name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")
PY

# =========================
# 训练
# =========================
cd "${CODE_ROOT}/training/src"

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
  --gradient_accumulation_steps 4 \
  --per_device_train_batch_size 1 \
  --learning_rate 1e-5 \
  --max_grad_norm 1.0 \
  --bf16 true \
  --gradient_checkpointing true \
  --optim adafactor \
  --report_to none

echo "✅ 训练结束，开始评测 PPL ..."

# =========================
# PPL 评测
# =========================
cd "${CODE_ROOT}"

python evaluation/src/eval_ppl_fineweb2.py \
  --model_name "${OUTPUT_DIR}" \
  --data_root "${DATA_ROOT}" \
  --langs "${LANG}" \
  --batch_size 1

echo "🎉 ${LANG} 单任务训练 + 测试完成"