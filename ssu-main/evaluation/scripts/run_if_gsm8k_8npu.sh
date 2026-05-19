#!/usr/bin/env bash
set -euo pipefail

cd /home/HwHiAiUser/cl_workspace/code/SSU/ssu-main

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cl_eval

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="/home/HwHiAiUser/cl_workspace/var/hf_cache"
export HF_HUB_CACHE="/home/HwHiAiUser/cl_workspace/var/hf_cache"
export HF_DATASETS_CACHE="/home/HwHiAiUser/cl_workspace/var/hf_cache"
export HF_DATASETS_TRUST_REMOTE_CODE=true

LOG_ROOT="/home/HwHiAiUser/cl_workspace/eval_logs/batch_if_gsm8k_logs"
mkdir -p "${LOG_ROOT}"

run_ifeval () {
  local NPU_ID="$1"
  local MODEL_PATH="$2"
  local OUT_DIR="$3"
  local LOG_FILE="$4"

  (
    export ASCEND_RT_VISIBLE_DEVICES="${NPU_ID}"
    python evaluation/src/ifeval.py \
      --model_name_or_path "${MODEL_PATH}" \
      --output_dir "${OUT_DIR}" \
      --cache_dir /home/HwHiAiUser/cl_workspace/var/hf_cache \
      --apply_chat_template
  ) > "${LOG_FILE}" 2>&1 &
}

run_gsm8k () {
  local NPU_ID="$1"
  local MODEL_PATH="$2"
  local OUT_DIR="$3"
  local LOG_FILE="$4"

  (
    export ASCEND_RT_VISIBLE_DEVICES="${NPU_ID}"
    python evaluation/src/gsm8k.py \
      --model_name_or_path "${MODEL_PATH}" \
      --output_dir "${OUT_DIR}" \
      --cache_dir /home/HwHiAiUser/cl_workspace/var/hf_cache \
      --mode gsm8k_cot \
      --apply_chat_template
  ) > "${LOG_FILE}" 2>&1 &
}

echo "================ 第一批：8 个作业并行 ================"

# Base
run_ifeval 0 \
  /home/HwHiAiUser/cl_workspace/var/hf_cache/Qwen2.5-1.5B-Instruct \
  /home/HwHiAiUser/cl_workspace/eval_logs/ifeval_full/Qwen2.5-1.5B-Instruct \
  ${LOG_ROOT}/base_ifeval.log

run_gsm8k 1 \
  /home/HwHiAiUser/cl_workspace/var/hf_cache/Qwen2.5-1.5B-Instruct \
  /home/HwHiAiUser/cl_workspace/eval_logs/gsm8k_cot_full/Qwen2.5-1.5B-Instruct \
  ${LOG_ROOT}/base_gsm8k.log

# single ig
run_ifeval 2 \
  /home/HwHiAiUser/cl_workspace/ckpt/single_15b/ibo_Latn \
  /home/HwHiAiUser/cl_workspace/eval_logs/ifeval_full/single_15b_ibo_Latn \
  ${LOG_ROOT}/single_ig_ifeval.log

run_gsm8k 3 \
  /home/HwHiAiUser/cl_workspace/ckpt/single_15b/ibo_Latn \
  /home/HwHiAiUser/cl_workspace/eval_logs/gsm8k_cot_full/single_15b_ibo_Latn \
  ${LOG_ROOT}/single_ig_gsm8k.log

# single ha
run_ifeval 4 \
  /home/HwHiAiUser/cl_workspace/ckpt/single_15b/hau_Latn \
  /home/HwHiAiUser/cl_workspace/eval_logs/ifeval_full/single_15b_hau_Latn \
  ${LOG_ROOT}/single_ha_ifeval.log

run_gsm8k 5 \
  /home/HwHiAiUser/cl_workspace/ckpt/single_15b/hau_Latn \
  /home/HwHiAiUser/cl_workspace/eval_logs/gsm8k_cot_full/single_15b_hau_Latn \
  ${LOG_ROOT}/single_ha_gsm8k.log

# single ky
run_ifeval 6 \
  /home/HwHiAiUser/cl_workspace/ckpt/single_15b/kir_Cyrl \
  /home/HwHiAiUser/cl_workspace/eval_logs/ifeval_full/single_15b_kir_Cyrl \
  ${LOG_ROOT}/single_ky_ifeval.log

run_gsm8k 7 \
  /home/HwHiAiUser/cl_workspace/ckpt/single_15b/kir_Cyrl \
  /home/HwHiAiUser/cl_workspace/eval_logs/gsm8k_cot_full/single_15b_kir_Cyrl \
  ${LOG_ROOT}/single_ky_gsm8k.log

wait

echo "================ 第二批：剩余 4 个作业 ================"

# single ne
run_ifeval 0 \
  /home/HwHiAiUser/cl_workspace/ckpt/single_15b/npi_Deva \
  /home/HwHiAiUser/cl_workspace/eval_logs/ifeval_full/single_15b_npi_Deva \
  ${LOG_ROOT}/single_ne_ifeval.log

run_gsm8k 1 \
  /home/HwHiAiUser/cl_workspace/ckpt/single_15b/npi_Deva \
  /home/HwHiAiUser/cl_workspace/eval_logs/gsm8k_cot_full/single_15b_npi_Deva \
  ${LOG_ROOT}/single_ne_gsm8k.log

# single amh
run_ifeval 2 \
  /home/HwHiAiUser/cl_workspace/ckpt/single_15b/amh_Ethi \
  /home/HwHiAiUser/cl_workspace/eval_logs/ifeval_full/single_15b_amh_Ethi \
  ${LOG_ROOT}/single_amh_ifeval.log

run_gsm8k 3 \
  /home/HwHiAiUser/cl_workspace/ckpt/single_15b/amh_Ethi \
  /home/HwHiAiUser/cl_workspace/eval_logs/gsm8k_cot_full/single_15b_amh_Ethi \
  ${LOG_ROOT}/single_amh_gsm8k.log

wait

echo "================ 全部完成 ================"
echo "日志目录: ${LOG_ROOT}"