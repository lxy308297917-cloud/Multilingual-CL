#!/usr/bin/env bash
set -e

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cl_eval

export CUDA_VISIBLE_DEVICES=""
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"

export HF_ENDPOINT=https://hf-mirror.com
export TRANSFORMERS_VERBOSITY=error
export HF_HOME="/home/HwHiAiUser/cl_workspace/var/hf_cache"
export HF_HUB_CACHE="/home/HwHiAiUser/cl_workspace/var/hf_cache"
export HF_DATASETS_CACHE="/home/HwHiAiUser/cl_workspace/var/hf_cache"
export HF_DATASETS_TRUST_REMOTE_CODE=true

SCRIPT="/home/HwHiAiUser/cl_workspace/code/SSU/ssu-main/evaluation/src/eval_fineweb2_nextsent_mcq.py"

BASE_MODEL="/home/HwHiAiUser/cl_workspace/var/hf_cache/Qwen2.5-1.5B-Instruct"
SINGLE_MODEL="/home/HwHiAiUser/cl_workspace/ckpt/single_15b/ibo_Latn"
DATASET_PATH="/home/HwHiAiUser/cl_workspace/data/fineweb2_eval/fineweb2_nextsent_overlap/ibo_Latn/train_overlap.jsonl"
OUT_BASE="/home/HwHiAiUser/cl_workspace/eval_logs/fineweb2_nextsent_15b/ig"

python "$SCRIPT" \
  --model_name_or_path "$BASE_MODEL" \
  --dataset_path "$DATASET_PATH" \
  --output_dir "$OUT_BASE/base" \
  --cache_dir "/home/HwHiAiUser/cl_workspace/var/hf_cache" \
  --max_samples 1000

python "$SCRIPT" \
  --model_name_or_path "$SINGLE_MODEL" \
  --dataset_path "$DATASET_PATH" \
  --output_dir "$OUT_BASE/single" \
  --cache_dir "/home/HwHiAiUser/cl_workspace/var/hf_cache" \
  --max_samples 1000