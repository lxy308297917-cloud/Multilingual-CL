#!/usr/bin/env bash
set -e

# ===================== Cache / Env =====================
export HF_HOME="E:/hf_cache"
export HF_HUB_CACHE="E:/hf_cache/hub"
export HF_DATASETS_CACHE="E:/hf_cache/datasets"
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_HUB_DISABLE_XET=1

custom_task_script_dir="C:/Users/Administrator/Desktop/SSU/ssu-main/evaluation/src"
log_base_dir="C:/Users/Administrator/Desktop/SSU/ssu-main/eval_results/lighteval_mmlu_fast"
mkdir -p "${log_base_dir}"

# ===================== Args =====================
# 用法：bash evaluation/scripts/MMLU_only_fast.sh <lang_code> <model_path>
lang_code="$1"         # ig / ha / ky / ne / am
model_name="$2"        # 例如 E:/ckpt_fft_ssu_seq/amh_Ethi/checkpoint-500

if [[ -z "$lang_code" || -z "$model_name" ]]; then
  echo "Usage: $0 <lang_code: ig|ha|ky|ne|am> <model_path>"
  exit 1
fi

# SSU 用的映射：ne/am/ha/ig/ky -> iso639-3（GMMLU）
declare -A iso639_3_lang_code=(
  ["ne"]="npi"
  ["am"]="amh"
  ["ha"]="hau"
  ["ig"]="ibo"
  ["ky"]="kir"
)

# 让输出目录更直观
model_tag=$(basename "$(dirname "${model_name}")")_$(basename "${model_name}")
out_dir="${log_base_dir}/${model_tag}"
mkdir -p "${out_dir}"

echo "========================================"
echo "[FAST MMLU/GMMLU] lang_code=${lang_code}"
echo "[FAST MMLU/GMMLU] model=${model_name}"
echo "[FAST MMLU/GMMLU] out=${out_dir}"
echo "========================================"

# ===================== 只测少量 subjects（按你之前表格） =====================
SUBJECTS=(
  abstract_algebra
  computer_security
  high_school_mathematics
)

# 拼 GMMLU task（只 3 个）
task_gmmlu=""
for s in "${SUBJECTS[@]}"; do
  t="lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:${s}|5|0"
  if [[ -z "${task_gmmlu}" ]]; then
    task_gmmlu="${t}"
  else
    task_gmmlu="${task_gmmlu},${t}"
  fi
done

# （可选）是否同时跑 English MMLU：0=不跑，1=跑同样3个subject
RUN_EN_MMLU=1

task_mmlu=""
if [[ "${RUN_EN_MMLU}" == "1" ]]; then
  for s in "${SUBJECTS[@]}"; do
    t="leaderboard|mmlu:${s}|5|0"
    if [[ -z "${task_mmlu}" ]]; then
      task_mmlu="${t}"
    else
      task_mmlu="${task_mmlu},${t}"
    fi
  done
fi

# ===================== 推理参数（加速关键） =====================
# 3060 Laptop 通常 float16 更稳；batch_size 先 4，不行再改 2
MODEL_ARGS="model_name=${model_name},batch_size=4,dtype=float16"

# ===================== GMMLU =====================
echo "[RUN] GMMLU tasks: ${task_gmmlu}"
lighteval accelerate \
  "${MODEL_ARGS}" \
  "${task_gmmlu}" \
  --custom-tasks "${custom_task_script_dir}/gmmlu.py" \
  --output-dir "${out_dir}/gmmlu_${iso639_3_lang_code[${lang_code}]}" \

# ===================== English MMLU（可选） =====================
if [[ "${RUN_EN_MMLU}" == "1" ]]; then
  echo "[RUN] EN-MMLU tasks: ${task_mmlu}"
    lighteval accelerate \
    "${MODEL_ARGS}" \
    "${task_mmlu}" \
    --output-dir "${out_dir}/mmlu_en"
fi


echo "✅ Done. Results in: ${out_dir}"


# ===================== 把 tracker 的 results_*.json 也收拢到 out_dir =====================
MODEL_DIR="$2"   # e.g. E:/ckpt_fft_ssu_seq/amh_Ethi/checkpoint-500
LANG_CODE="$1"   # e.g. ig

TRAIN_LANG="$(basename "$(dirname "$MODEL_DIR")")"   # amh_Ethi
CKPT_NAME="$(basename "$MODEL_DIR")"                 # checkpoint-500

# LightEval 可能会把 results_*.json 写回 checkpoint 目录，这里把最新的那份复制到 out_dir 根目录
LATEST_JSON="$(ls -t "$MODEL_DIR"/results_*.json 2>/dev/null | head -n 1)"

if [ -z "$LATEST_JSON" ]; then
  echo "[WARN] 没在 $MODEL_DIR 找到 results_*.json（可能评测失败/中断，或版本不写回模型目录）。"
else
  # 统一放到图三这种目录：eval_results/lighteval_mmlu_fast/<train>_<ckpt>/
  DEST_DIR="${out_dir}"
  mkdir -p "$DEST_DIR"

  NEW_NAME="train_${TRAIN_LANG}__eval_${LANG_CODE}__${CKPT_NAME}.json"
  cp -f "$LATEST_JSON" "$DEST_DIR/$NEW_NAME"
  echo "[OK] 已收拢 tracker 结果到：$DEST_DIR/$NEW_NAME"
fi

