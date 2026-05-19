#!/usr/bin/env bash
set -e


# su - HwHiAiUser 
# cd /home/HwHiAiUser/cl_workspace/code/SSU/ssu-main/evaluation/scripts

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cl

export HF_ENDPOINT=https://hf-mirror.com
# Replay 顺序（与 cl.sh 一致）:
# Task1: ibo_Latn   (Igbo)
# Task2: hau_Latn   (Hausa)
# Task3: kir_Cyrl  (Kyrgyz)
# Task4: npi_Deva  (Nepali)
# Task5: amh_Ethi  (Amharic)


# 用法：
# 1) Task1（不需要 replay，直接用 cl.sh 或 single）：
#    bash training/scripts/cl.sh swh_Latn
#
# 2) 从 Task2 开始用 replay：
#    bash training/scripts/replay.sh ibo_Latn D:/fineweb2_cpt/swh_Latn "D:/fineweb2_cpt/swh_Latn/train5k"
#
# 3) 多回放数据集（逗号分隔）：
#    bash training/scripts/replay.sh kir_Cyrl D:/ckpt_replay_seq/ibo_Latn "D:/fineweb2_cpt/swh_Latn/train5k,D:/fineweb2_cpt/ibo_Latn/train5k"
#
# 4) 你也可以在末尾继续追加 main.py 的其他参数（会透传）：
#    bash training/scripts/replay.sh tel_Telu D:/ckpt_replay_seq/kir_Cyrl "..." --max_steps 800 --learning_rate 2e-5

BASE_MODEL="/home/HwHiAiUser/cl_workspace/var/hf_cache/Qwen2.5-1.5B-Instruct"

LANG="$1"                 # 当前任务语言
PREV_MODEL="$2"            # 上一任务 checkpoint（为空则用 BASE_MODEL）
REPLAY_DATASETS="$3"       # 回放数据集路径（逗号分隔），例如 "D:/fineweb2_cpt/swh_Latn/train5k,D:/fineweb2_cpt/ibo_Latn/train5k"

# 额外参数透传给 main.py（从第4个参数开始）
shift 3 || true
EXTRA_ARGS=("$@")

if [ -z "$LANG" ]; then
  echo "用法：bash training/scripts/replay.sh <lang> <prev_model_or_empty> <replay_dataset_paths_csv> [extra_main_args...]"
  exit 1
fi

if [ -n "$PREV_MODEL" ]; then
  MODEL_NAME="$PREV_MODEL"
else
  MODEL_NAME="$BASE_MODEL"
fi

# === FineWeb2 处理后数据根目录（按你的实际）===
DATA_ROOT="$HOME/cl_workspace/data/fineweb2_cpt_ssu"

# 训练/测试目录
TRAIN_DIR="${DATA_ROOT}/${LANG}/train5k"
TEST_DIR="${DATA_ROOT}/${LANG}/test"

# === 输出目录（每个语言一个 checkpoint 目录）===
OUT_ROOT="/data/HwHiAiUser/cl_workspace/ckpt/replay_seq_15b"
OUTPUT_DIR="${OUT_ROOT}/${LANG}"

LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

# 回放比例
REPLAY_RATIO_DEFAULT="0.1"
REPLAY_SEED_DEFAULT="42"

echo "========================================"
echo "CL训练（FineWeb2）- Replay"
echo "LANG=${LANG}"
echo "TRAIN_DIR=${TRAIN_DIR}"
echo "TEST_DIR=${TEST_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "MODEL_NAME_OR_PATH=${MODEL_NAME}"
echo "TOKENIZER_NAME_OR_PATH=${BASE_MODEL}"
echo "REPLAY_DATASETS=${REPLAY_DATASETS}"
echo "REPLAY_RATIO(default)=${REPLAY_RATIO_DEFAULT}"
echo "REPLAY_SEED(default)=${REPLAY_SEED_DEFAULT}"
echo "EXTRA_ARGS=${EXTRA_ARGS[*]}"
echo "========================================"

# 基本校验：从 Task2 开始必须给回放路径
if [ -z "$REPLAY_DATASETS" ]; then
  echo "⚠️ 你没有提供 REPLAY_DATASETS（回放数据集路径）。"
  echo "   如果这是 Task1，请用 cl.sh；如果是 Task2+，请传入："
  echo "   例如：\"D:/fineweb2_cpt/swh_Latn/train5k\" 或多个用逗号分隔。"
  exit 1
fi

# 训练
cd training/src

python main.py \
  --dataset_path "${TRAIN_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --logging_dir "${LOG_DIR}" \
  --model_name_or_path "${MODEL_NAME}" \
  --tokenizer_name_or_path "${BASE_MODEL}" \
  --cache_dir "$HOME/cl_workspace/var/hf_cache" \
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
  --gradient_accumulation_steps 1 \
  --per_device_train_batch_size 1 \
  --learning_rate 1e-5 \
  --max_grad_norm 1.0 \
  \
  --cl_method replay \
  --replay_dataset_path "${REPLAY_DATASETS}" \
  --replay_ratio "${REPLAY_RATIO_DEFAULT}" \
  --replay_seed "${REPLAY_SEED_DEFAULT}" \
  \
  "${EXTRA_ARGS[@]}"

echo "✅ 训练结束，开始评测 PPL ..."

# 回到项目根目录跑 evaluation
cd ../../

python evaluation/src/eval_ppl_fineweb2.py \
  --model_name "${OUTPUT_DIR}" \
  --data_root "${DATA_ROOT}" \
  --langs "${LANG}" \
  --batch_size 8 \
  --max_test_blocks 5000

echo "🎉 ${LANG} Replay 训练 + 测试完成"
