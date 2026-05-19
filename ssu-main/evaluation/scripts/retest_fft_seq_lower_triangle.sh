#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash evaluation/scripts/retest_fft_seq_lower_triangle.sh [out_dir]

OUT_DIR="${1:-/home/HwHiAiUser/cl_workspace/eval_logs/post/fft_seq_triangle_retest_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${OUT_DIR}/results_json"

source /root/anaconda3/etc/profile.d/conda.sh
conda activate /home/HwHiAiUser/anaconda3/envs/cl_eval
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export CUDA_VISIBLE_DEVICES=""
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"

export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

export TRANSFORMERS_VERBOSITY=error
export HF_HOME="/home/HwHiAiUser/cl_workspace/hf_cache"
export HF_HUB_CACHE="/home/HwHiAiUser/cl_workspace/hf_cache"
export HF_DATASETS_CACHE="/home/HwHiAiUser/cl_workspace/hf_cache"
export HF_DATASETS_TRUST_REMOTE_CODE=true

CUSTOM_TASK_DIR="/home/HwHiAiUser/cl_workspace/code/SSU/ssu-main/evaluation/src"

# Task order fixed by your table
TASK_LANGS=(ig ha ky ne am)

# Model checkpoints for FFT-SEQ lower triangle (Task1 starts from single ig)
MODEL_PATHS=(
  "/data/HwHiAiUser/cl_workspace/ckpt/single_15b/ibo_Latn"
  "/data/HwHiAiUser/cl_workspace/ckpt/fft_ssu_seq_15b/hau_Latn"
  "/data/HwHiAiUser/cl_workspace/ckpt/fft_ssu_seq_15b/kir_Cyrl"
  "/data/HwHiAiUser/cl_workspace/ckpt/fft_ssu_seq_15b/npi_Deva"
  "/data/HwHiAiUser/cl_workspace/ckpt/fft_ssu_seq_15b/amh_Ethi"
)

belebele_code() {
  case "$1" in
    ig) echo "ibo_Latn" ;;
    ha) echo "hau_Latn" ;;
    ky) echo "kir_Cyrl" ;;
    ne) echo "npi_Deva" ;;
    am) echo "amh_Ethi" ;;
    *) echo "unknown" ;;
  esac
}

gmmlu_code() {
  case "$1" in
    ig) echo "ibo" ;;
    ha) echo "hau" ;;
    ky) echo "kir" ;;
    ne) echo "npi" ;;
    am) echo "amh" ;;
    *) echo "unknown" ;;
  esac
}

GMMLU_SUBJECTS=(
  abstract_algebra anatomy astronomy business_ethics clinical_knowledge
  college_biology college_chemistry college_computer_science college_mathematics college_medicine
  college_physics computer_security conceptual_physics econometrics electrical_engineering
  elementary_mathematics formal_logic global_facts high_school_biology high_school_chemistry
  high_school_computer_science high_school_european_history high_school_geography high_school_government_and_politics high_school_macroeconomics
  high_school_mathematics high_school_microeconomics high_school_physics high_school_psychology high_school_statistics
  high_school_us_history high_school_world_history human_aging human_sexuality international_law
  jurisprudence logical_fallacies machine_learning management marketing
  medical_genetics miscellaneous moral_disputes moral_scenarios nutrition
  philosophy prehistory professional_accounting professional_law professional_medicine
  professional_psychology public_relations security_studies sociology us_foreign_policy
  virology world_religions
)

build_gmmlu_task() {
  local code="$1"
  local out=""
  local s
  for s in "${GMMLU_SUBJECTS[@]}"; do
    local t="lighteval|gmmlu_${code}_mcf:${s}|5|0"
    if [[ -z "${out}" ]]; then
      out="${t}"
    else
      out="${out},${t}"
    fi
  done
  echo "${out}"
}

GMMLU_CSV="${OUT_DIR}/gmmlu_triangle.csv"
BELEBELE_CSV="${OUT_DIR}/belebele_triangle.csv"
echo "model_task,test_task,model_lang,test_lang,score,result_json" > "${GMMLU_CSV}"
echo "model_task,test_task,model_lang,test_lang,score,result_json" > "${BELEBELE_CSV}"

run_and_capture() {
  local model_path="$1"
  local task_str="$2"
  local custom_task_py="$3"
  local tag="$4"
  local model_task="$5"
  local test_task="$6"
  local model_lang="$7"
  local test_lang="$8"
  local out_csv="$9"

  lighteval accelerate \
    "model_name=${model_path},batch_size=8,dtype=float16" \
    "${task_str}" \
    --custom-tasks "${custom_task_py}" \
    --use-chat-template > "${OUT_DIR}/${tag}_T${model_task}_to_T${test_task}.log" 2>&1

  local latest_json
  latest_json="$(ls -1t "${model_path}"/results_*.json 2>/dev/null | head -n 1 || true)"
  if [[ -z "${latest_json}" ]]; then
    echo "[ERROR] No results_*.json found in ${model_path}" >&2
    return 1
  fi

  local copied="${OUT_DIR}/results_json/${tag}_T${model_task}_to_T${test_task}_$(basename "${latest_json}")"
  cp "${latest_json}" "${copied}"

  local score
  score="$(jq -r '.results.all.acc // empty' "${latest_json}")"
  if [[ -z "${score}" || "${score}" == "null" ]]; then
    score="NaN"
  fi

  echo "${model_task},${test_task},${model_lang},${test_lang},${score},${copied}" >> "${out_csv}"
  echo "[DONE] ${tag} T${model_task}->T${test_task} ${model_lang}->${test_lang} score=${score}"
}

echo "[INFO] Output dir: ${OUT_DIR}"

for ((i=1; i<=5; i++)); do
  model_lang="${TASK_LANGS[$((i-1))]}"
  model_path="${MODEL_PATHS[$((i-1))]}"

  if [[ ! -d "${model_path}" ]]; then
    echo "[WARN] Skip missing model path: ${model_path}"
    continue
  fi

  for ((j=1; j<=i; j++)); do
    test_lang="${TASK_LANGS[$((j-1))]}"

    bb_code="$(belebele_code "${test_lang}")"
    gm_code="$(gmmlu_code "${test_lang}")"

    bb_task="lighteval|belebele_${bb_code}_mcf|3|0"
    gm_task="$(build_gmmlu_task "${gm_code}")"

    run_and_capture "${model_path}" "${bb_task}" "${CUSTOM_TASK_DIR}/belebele.py" "belebele" "${i}" "${j}" "${model_lang}" "${test_lang}" "${BELEBELE_CSV}"
    run_and_capture "${model_path}" "${gm_task}" "${CUSTOM_TASK_DIR}/gmmlu.py" "gmmlu" "${i}" "${j}" "${model_lang}" "${test_lang}" "${GMMLU_CSV}"
  done
done

echo "[INFO] Finished."
echo "[INFO] Belebele CSV: ${BELEBELE_CSV}"
echo "[INFO] GMMLU CSV:   ${GMMLU_CSV}"
