#!/usr/bin/env bash
set -euo pipefail

# ========= 你需要改/确认的路径 =========
# 训练模型根目录：每个语言一个目录，里面有 checkpoint-500
CKPT_ROOT="E:/ckpt_fft_ssu_seq"
CKPT_NAME="checkpoint-500"

# 统一汇总目录（最终你只看这里）
UNIFIED_ROOT="C:/Users/Administrator/Desktop/SSU/final_results_triangle"

# 评测脚本（你现在快版OK）
EVAL_SCRIPT="evaluation/scripts/MMLU_only_fast.sh"

# ========= 任务顺序（训练语言目录名）=========
TRAIN_LANGS=( "ibo_Latn" "hau_Latn" "kir_Cyrl" "npi_Deva" "amh_Ethi" )

# ========= GMMLU 的语言代码映射（传给 MMLU_only_fast.sh 的第1参数）=========
# ibo->ig, hau->ha, kir->ky, npi->ne, amh->am
GMMLU_CODES=( "ig" "ha" "ky" "ne" "am" )

mkdir -p "$UNIFIED_ROOT"

echo "=============================="
echo "[RUN] 下三角评测：GMMLU(3 subjects) + EN-MMLU(3 subjects)"
echo "[CKPT_ROOT] $CKPT_ROOT"
echo "[CKPT] $CKPT_NAME"
echo "[SAVE] $UNIFIED_ROOT"
echo "=============================="

# 外层：第 T 个任务训练后的模型
for ((t=0; t<${#TRAIN_LANGS[@]}; t++)); do
  TRAIN_LANG="${TRAIN_LANGS[$t]}"
  MODEL_DIR="$CKPT_ROOT/$TRAIN_LANG/$CKPT_NAME"

  if [ ! -d "$MODEL_DIR" ]; then
    echo "[WARN] 模型目录不存在：$MODEL_DIR  -> 跳过该训练任务"
    continue
  fi

  echo ""
  echo "========== [TRAIN TASK $((t+1))] $TRAIN_LANG =========="

  # 内层：只测前 T 个语言（下三角）
  for ((j=0; j<=t; j++)); do
    EVAL_LANG="${TRAIN_LANGS[$j]}"
    LANG_CODE="${GMMLU_CODES[$j]}"

    echo ""
    echo "[EVAL] train=$TRAIN_LANG  eval=$EVAL_LANG  code=$LANG_CODE  model=$MODEL_DIR"

    # 跑评测（脚本内部会同时跑 GMMLU(code) + EN-MMLU）
    bash "$EVAL_SCRIPT" "$LANG_CODE" "$MODEL_DIR" || {
      echo "[WARN] 评测失败：train=$TRAIN_LANG eval=$EVAL_LANG code=$LANG_CODE"
      continue
    }

    # LightEval 结果通常写回 $MODEL_DIR/results_*.json
    LATEST_JSON="$(ls -t "$MODEL_DIR"/results_*.json 2>/dev/null | head -n 1 || true)"
    if [ -z "$LATEST_JSON" ]; then
      echo "[WARN] 没找到 results_*.json（可能中断/失败）"
      continue
    fi

    # 收拢 + 重命名
    DEST_DIR="$UNIFIED_ROOT/train_${TRAIN_LANG}/${CKPT_NAME}"
    mkdir -p "$DEST_DIR"
    NEW_NAME="train_${TRAIN_LANG}__eval_${EVAL_LANG}__code_${LANG_CODE}__${CKPT_NAME}.json"
    cp -f "$LATEST_JSON" "$DEST_DIR/$NEW_NAME"
    echo "[OK] 保存：$DEST_DIR/$NEW_NAME"
  done
done

echo ""
echo "=============================="
echo "[DONE] 下三角全部评测结束"
echo "结果已统一放在：$UNIFIED_ROOT"
echo "=============================="
