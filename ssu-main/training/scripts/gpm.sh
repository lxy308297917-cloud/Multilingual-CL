#!/bin/bash
set -e

# source /path/to/envs/ssu_train/bin/activate




export TRANSFORMERS_VERBOSITY=info
export HF_HOME="./my_data/cache"
export HF_HUB_CACHE="./my_data/cache"
export HF_DATASETS_CACHE="./my_data/cache"
export HF_DATASETS_TRUST_REMOTE_CODE=true

MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
MODEL_ABBREV="Qwen2.5-0.5B-Instruct"

LANG_CODE="$1"         # 例如 swh_Latn
TASK_ID="${2:-0}"      # 第2个参数可选，默认 0


ROOT_DIR="$(pwd)"


DATASET_DIR="${ROOT_DIR}/my_data/processed/${MODEL_ABBREV}_cpt_${LANG_CODE}_fineweb2"


OUTPUT_DIR="${ROOT_DIR}/outputs/gpm_fineweb2/${MODEL_ABBREV}-${LANG_CODE}"
LOGGING_DIR="${ROOT_DIR}/logs/gpm_fineweb2/${MODEL_ABBREV}-${LANG_CODE}"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOGGING_DIR}"

# =========================
# GPM 关键点：是否继承上一个 task
# - task_id=0：从基础模型开始
# - task_id>0：必须从上一个 task 的输出目录开始（里面有 gpm_state.pt）
# =========================
PREV_LANG_CODE="${3:-}"   # 可选：第三个参数传“上一个语言名”，用于自动设置 model_name_or_path

if [ "${TASK_ID}" -eq 0 ]; then
  MODEL_PATH="${MODEL_NAME}"
else
  if [ -z "${PREV_LANG_CODE}" ]; then
    echo "[ERROR] task_id>0 时需要提供上一个语言名作为第3个参数，用于继承上一个 task 的模型与 gpm_state.pt"
    echo "示例：bash training/scripts/gpm.sh jav_Latn 1 ibo_Latn"
    exit 1
  fi
  MODEL_PATH="${ROOT_DIR}/outputs/gpm_fineweb2/${MODEL_ABBREV}-${PREV_LANG_CODE}"
fi

# =========================
# GPM 超参数
# =========================
GPM_THRESHOLD_BASE=0.97
GPM_THRESHOLD_INC=0.003
GPM_MAX_TOKENS_PER_LAYER=4096
GPM_KEYWORDS="mlp,o_proj,down_proj,up_proj,gate_proj"
GPM_UPDATE_MAX_BATCHES=20

# =========================
# 训练参数
# =========================
MAX_STEPS=200
SAVE_STEPS=50
LOGGING_STEPS=1
GRAD_ACCUM=8
PER_DEVICE_BS=1
LR=1e-5

cd training/src
python main.py \
  --dataset_path "${DATASET_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --logging_dir "${LOGGING_DIR}" \
  --model_name_or_path "${MODEL_PATH}" \
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
  --save_steps "${SAVE_STEPS}" \
  --max_steps "${MAX_STEPS}" \
  --logging_steps "${LOGGING_STEPS}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --per_device_train_batch_size "${PER_DEVICE_BS}" \
  --learning_rate "${LR}" \
  --max_grad_norm 1.0 \
  \
  --cl_method gpm \
  --task_id "${TASK_ID}" \
  --gpm_threshold_base "${GPM_THRESHOLD_BASE}" \
  --gpm_threshold_inc "${GPM_THRESHOLD_INC}" \
  --gpm_max_tokens_per_layer "${GPM_MAX_TOKENS_PER_LAYER}" \
  --gpm_keywords "${GPM_KEYWORDS}" \
  --gpm_update_max_batches "${GPM_UPDATE_MAX_BATCHES}"


# # task 0
# bash training/scripts/gpm.sh ibo_Latn 0

# # task 1（继承 task0 输出）
# bash training/scripts/gpm.sh jav_Latn 1 ibo_Latn

# # task 2
# bash training/scripts/gpm.sh kir_Cyrl 2 jav_Latn

# # task 3
# bash training/scripts/gpm.sh swh_Latn 3 kir_Cyrl

# # task 4
# bash training/scripts/gpm.sh tel_Telu 4 swh_Latn