#!/usr/bin/env bash
set -euo pipefail

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cl
export HF_ENDPOINT=https://hf-mirror.com

cd /home/HwHiAiUser/cl_workspace/code/SSU/ssu-main

DATA_ROOT="/data/HwHiAiUser/cl_workspace/data/fineweb2_cpt_ssu"
LOG_ROOT="/home/HwHiAiUser/cl_workspace/logs/ppl_triangle_lora_15b"
MODEL_ROOT="/data/HwHiAiUser/cl_workspace/ckpt/lora_15b_seq_merged"

mkdir -p "${LOG_ROOT}"

run_ppl () {
  local model_path="$1"
  local langs="$2"
  local tag="$3"
  local log_file="${LOG_ROOT}/${tag}.log"

  echo "========================================"
  echo "开始测试 PPL"
  echo "MODEL=${model_path}"
  echo "LANGS=${langs}"
  echo "LOG=${log_file}"
  echo "========================================"

  python evaluation/src/eval_ppl_fineweb2.py \
    --model_name "${model_path}" \
    --data_root "${DATA_ROOT}" \
    --langs ${langs} \
    --batch_size 8 \
    2>&1 | tee "${log_file}"
}

run_ppl "${MODEL_ROOT}/ibo_Latn_train5k" "ibo_Latn" "step1_ibo"
run_ppl "${MODEL_ROOT}/hau_Latn_train5k" "ibo_Latn hau_Latn" "step2_hau"
run_ppl "${MODEL_ROOT}/kir_Cyrl_train5k" "ibo_Latn hau_Latn kir_Cyrl" "step3_kir"
run_ppl "${MODEL_ROOT}/npi_Deva_train5k" "ibo_Latn hau_Latn kir_Cyrl npi_Deva" "step4_npi"
run_ppl "${MODEL_ROOT}/amh_Ethi_train5k" "ibo_Latn hau_Latn kir_Cyrl npi_Deva amh_Ethi" "step5_amh"

echo "🎉 LoRA 1.5B 下三角 PPL 测试完成"
echo "日志目录：${LOG_ROOT}"