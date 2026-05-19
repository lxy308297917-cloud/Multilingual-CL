#!/usr/bin/env bash
set -euo pipefail

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cl
export HF_ENDPOINT=https://hf-mirror.com

cd /home/HwHiAiUser/cl_workspace/code/SSU/ssu-main

DATA_ROOT="/data/HwHiAiUser/cl_workspace/data/fineweb2_cpt_ssu"
LOG_ROOT="/home/HwHiAiUser/cl_workspace/logs/ppl_triangle_fft_15b"
MODEL_ROOT="/data/HwHiAiUser/cl_workspace/ckpt/fft_ssu_seq_15b"

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


run_ppl "${MODEL_ROOT}/hau_Latn" "ibo_Latn hau_Latn" "step2_hau"
run_ppl "${MODEL_ROOT}/kir_Cyrl" "ibo_Latn hau_Latn kir_Cyrl" "step3_kir"
run_ppl "${MODEL_ROOT}/npi_Deva" "ibo_Latn hau_Latn kir_Cyrl npi_Deva" "step4_npi"
run_ppl "${MODEL_ROOT}/amh_Ethi" "ibo_Latn hau_Latn kir_Cyrl npi_Deva amh_Ethi" "step5_amh"

echo "🎉 FFT 1.5B 下三角 PPL 测试完成"
echo "日志目录：${LOG_ROOT}"