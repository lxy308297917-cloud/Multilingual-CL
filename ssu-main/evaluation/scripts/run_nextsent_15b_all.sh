#!/usr/bin/env bash
set -e

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cl_eval

export CUDA_VISIBLE_DEVICES=""
export ASCEND_RT_VISIBLE_DEVICES=1

export HF_ENDPOINT=https://hf-mirror.com
export TRANSFORMERS_VERBOSITY=error
export HF_HOME="/home/HwHiAiUser/cl_workspace/var/hf_cache"
export HF_HUB_CACHE="/home/HwHiAiUser/cl_workspace/var/hf_cache"
export HF_DATASETS_CACHE="/home/HwHiAiUser/cl_workspace/var/hf_cache"
export HF_DATASETS_TRUST_REMOTE_CODE=true

SCRIPT="/home/HwHiAiUser/cl_workspace/code/SSU/ssu-main/evaluation/src/eval_fineweb2_nextsent_mcq.py"

DATA_ROOT="/home/HwHiAiUser/cl_workspace/data/fineweb2_eval/fineweb2_nextsent_overlap"
OUT_ROOT="/home/HwHiAiUser/cl_workspace/eval_logs/fineweb2_nextsent_15b"

BASE_MODEL="/home/HwHiAiUser/cl_workspace/var/hf_cache/Qwen2.5-1.5B-Instruct"
SINGLE_ROOT="/home/HwHiAiUser/cl_workspace/ckpt/single_15b"

LANGS=("ibo_Latn" "hau_Latn" "kir_Cyrl" "npi_Deva" "amh_Ethi")

echo "========================================"
echo "开始评测 1.5B BASE（5语言，全量）"
echo "========================================"

for L in "${LANGS[@]}"; do
  echo ">>> [BASE-15B] $L"
  python "$SCRIPT" \
    --model_name_or_path "$BASE_MODEL" \
    --dataset_path "$DATA_ROOT/$L/train_overlap.jsonl" \
    --output_dir "$OUT_ROOT/base/$L" \
    --cache_dir "/home/HwHiAiUser/cl_workspace/var/hf_cache" \
    --max_samples 1000
done

echo "========================================"
echo "开始评测 1.5B SINGLE（5语言，全量）"
echo "========================================"

for L in "${LANGS[@]}"; do
  echo ">>> [SINGLE-15B] $L"
  python "$SCRIPT" \
    --model_name_or_path "$SINGLE_ROOT/$L" \
    --dataset_path "$DATA_ROOT/$L/train_overlap.jsonl" \
    --output_dir "$OUT_ROOT/single/$L" \
    --cache_dir "/home/HwHiAiUser/cl_workspace/var/hf_cache" \
    --max_samples 1000
done

echo "========================================"
echo "全部评测完成"
echo "结果目录: $OUT_ROOT"
echo "========================================"