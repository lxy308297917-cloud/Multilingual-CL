#!/bin/bash
set -e

# ===== 1) 改成你自己的环境路径 =====
conda activate ssu_lighteval

# ===== 2) 基本配置 =====
export TRANSFORMERS_VERBOSITY=info
export HF_HOME="/home/HwHiAiUser/cl_workspace/hf_cache"
export HF_HUB_CACHE="/home/HwHiAiUser/cl_workspace/hf_cache"
export HF_DATASETS_CACHE="/home/HwHiAiUser/cl_workspace/hf_cache"
export HF_DATASETS_TRUST_REMOTE_CODE=true

# ===== 3) 项目路径 =====
PROJECT_ROOT="/home/HwHiAiUser/cl_workspace/code/SSU/ssu-main"
CUSTOM_TASK_SCRIPT_DIR="${PROJECT_ROOT}/evaluation/src"
LOG_BASE_DIR="${PROJECT_ROOT}/evaluation/logs/base_cls"
mkdir -p "${LOG_BASE_DIR}"

# ===== 4) base 模型 =====
MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
MODEL_ABBREV="Qwen2.5-0.5B-Instruct"

# 五种语言
LANG_CODES=(
    "am"
    "ha"
    "ig"
    "ky"
    "ne"
)

declare -A LANG_CODE_TO_BELEBELE_LANG_CODE=(
    ["ne"]="npi_Deva"
    ["am"]="amh_Ethi"
    ["ha"]="hau_Latn"
    ["ig"]="ibo_Latn"
    ["ky"]="kir_Cyrl"
)

declare -A ISO639_3_LANG_CODE=(
    ["ne"]="npi"
    ["am"]="amh"
    ["ha"]="hau"
    ["ig"]="ibo"
    ["ky"]="kir"
)

echo "========================================"
echo "开始评测 base model"
echo "MODEL_NAME=${MODEL_NAME}"
echo "LOG_BASE_DIR=${LOG_BASE_DIR}"
echo "========================================"

# ===== 5) 逐语言跑：Belebele + GMMLU =====
for lang_code in "${LANG_CODES[@]}"; do
    echo "----------------------------------------"
    echo "当前语言: ${lang_code}"
    echo "----------------------------------------"

    # 5.1 目标语言 Belebele
    task="lighteval|belebele_${LANG_CODE_TO_BELEBELE_LANG_CODE[${lang_code}]}_mcf|3|0"
    task_name=$(echo "${task}" | cut -d'|' -f2 | cut -d'_' -f1)

    lighteval accelerate \
        "model_name=${MODEL_NAME},batch_size=1,dtype=bfloat16" \
        "${task}" \
        --custom-tasks "${CUSTOM_TASK_SCRIPT_DIR}/belebele.py" \
        --output-dir "${LOG_BASE_DIR}/${MODEL_ABBREV}/${task_name}_${lang_code}" \
        --use-chat-template

    # 5.2 目标语言 GMMLU
    task="lighteval|gmmlu_${ISO639_3_LANG_CODE[${lang_code}]}_mcf:abstract_algebra|5|0,lighteval|gmmlu_${ISO639_3_LANG_CODE[${lang_code}]}_mcf:anatomy|5|0,lighteval|gmmlu_${ISO639_3_LANG_CODE[${lang_code}]}_mcf:astronomy|5|0,lighteval|gmmlu_${ISO639_3_LANG_CODE[${lang_code}]}_mcf:business_ethics|5|0,lighteval|gmmlu_${ISO639_3_LANG_CODE[${lang_code}]}_mcf:clinical_knowledge|5|0,lighteval|gmmlu_${ISO639_3_LANG_CODE[${lang_code}]}_mcf:college_biology|5|0,lighteval|gmmlu_${ISO639_3_LANG_CODE[${lang_code}]}_mcf:college_chemistry|5|0,lighteval|gmmlu_${ISO639_3_LANG_CODE[${lang_code}]}_mcf:college_computer_science|5|0,lighteval|gmmlu_${ISO639_3_LANG_CODE[${lang_code}]}_mcf:college_mathematics|5|0,lighteval|gmmlu_${ISO639_3_LANG_CODE[${lang_code}]}_mcf:college_medicine|5|0,lighteval|gmmlu_${ISO639_3_LANG_CODE[${lang_code}]}_mcf:college_physics|5|0,lighteval|gmmlu_${ISO639_3_LANG_CODE[${lang_code}]}_mcf:computer_security|5|0,lighteval|gmmlu_${ISO639_3_LANG_CODE[${lang_code}]}_mcf:conceptual_physics|5|0,lighteval|gmmlu_${ISO639_3_LANG_CODE[${lang_code}]}_mcf:econometrics|5|0,lighteval|gmmlu_${ISO639_3_LANG_CODE[${lang_code}]}_mcf:electrical_engineering|5|0,lighteval|gmmlu_${ISO639_3_LANG_CODE[${lang_code}]}_mcf:elementary_mathematics|5|0,lighteval|gmmlu_${ISO639_3_LANG_CODE[${lang_code}]}_mcf:formal_logic|5|0,lighteval|gmmlu_${ISO639_3_LANG_CODE[${lang_code}]}_mcf:global_facts|5|0,lighteval|gmmlu_${ISO639_3_LANG_CODE[${lang_code}]}_mcf:high_school_biology|5|0,lighteval|gmmlu_${ISO639_3_LANG_CODE[${lang_code}]}_mcf:high_school_chemistry|5|0,lighteval|gmmlu_${ISO639_3_LANG_CODE[${lang_code}]}_mcf:high_school_computer_science|5|0,lighteval|gmmlu_${ISO639_3_LANG_CODE[${lang_code}]}_mcf:high_school_european_history|5|0,lighteval|gmmlu_${ISO639_3_LANG_CODE[${lang_code}]}_mcf:high_school_geography|5|0,lighteval|gmmlu_${ISO639_3_LANG_CODE[${lang_code}]}_mcf:high_school_government_and_politics|5|0,lighteval|gmmlu_${ISO639_3_LANG_CODE[${lang_code}]}_mcf:high_school_macroeconomics|5|0,lighteval|gmmlu_${ISO639_3_LANG_CODE[${lang_code}]}_mcf:high_school_mathematics|5|0,lighteval|gmmlu_${ISO639_3_LANG_CODE[${lang_code}]}_mcf:high_school_microeconomics|5|0,lighteval|gmmlu_${ISO639_3_LANG_CODE[${lang_code}]}_mcf:high_school_physics|5|0,lighteval|gmmlu_${ISO639_3_LANG_CODE[${lang_code}]}_mcf:high_school_psychology|5|0,lighteval|gmmlu_${ISO639_3_LANG_CODE[${lang_code}]}_mcf:high_school_statistics|5|0,lighteval|gmmlu_${ISO639_3_LANG_CODE[${lang_code}]}_mcf:high_school_us_history|5|0,lighteval|gmmlu_${ISO639_3_LANG_CODE[${lang_code}]}_mcf:high_school_world_history|5|0,lighteval|gmmlu_${ISO639_3_LANG_CODE[${lang_code}]}_mcf:human_aging|5|0,lighteval|gmmlu_${ISO639_3_LANG_CODE[${lang_code}]}_mcf:human_sexuality|5|0,lighteval|gmmlu_${ISO639_3_LANG_CODE[${lang_code}]}_mcf:international_law|5|0,lighteval|gmmlu_${ISO639_3_LANG_CODE[${lang_code}]}_mcf:jurisprudence|5|0,lighteval|gmmlu_${ISO639_3_LANG_CODE[${lang_code}]}_mcf:logical_fallacies|5|0,lighteval|gmmlu_${ISO639_3_LANG_CODE[${lang_code}]}_mcf:machine_learning|5|0,lighteval|gmmlu_${ISO639_3_LANG_CODE[${lang_code}]}_mcf:management|5|0,lighteval|gmmlu_${ISO639_3_LANG_CODE[${lang_code}]}_mcf:marketing|5|0,lighteval|gmmlu_${ISO639_3_LANG_CODE[${lang_code}]}_mcf:medical_genetics|5|0,lighteval|gmmlu_${ISO639_3_LANG_CODE[${lang_code}]}_mcf:miscellaneous|5|0,lighteval|gmmlu_${ISO639_3_LANG_CODE[${lang_code}]}_mcf:moral_disputes|5|0,lighteval|gmmlu_${ISO639_3_LANG_CODE[${lang_code}]}_mcf:moral_scenarios|5|0,lighteval|gmmlu_${ISO639_3_LANG_CODE[${lang_code}]}_mcf:nutrition|5|0,lighteval|gmmlu_${ISO639_3_LANG_CODE[${lang_code}]}_mcf:philosophy|5|0,lighteval|gmmlu_${ISO639_3_LANG_CODE[${lang_code}]}_mcf:prehistory|5|0,lighteval|gmmlu_${ISO639_3_LANG_CODE[${lang_code}]}_mcf:professional_accounting|5|0,lighteval|gmmlu_${ISO639_3_LANG_CODE[${lang_code}]}_mcf:professional_law|5|0,lighteval|gmmlu_${ISO639_3_LANG_CODE[${lang_code}]}_mcf:professional_medicine|5|0,lighteval|gmmlu_${ISO639_3_LANG_CODE[${lang_code}]}_mcf:professional_psychology|5|0,lighteval|gmmlu_${ISO639_3_LANG_CODE[${lang_code}]}_mcf:public_relations|5|0,lighteval|gmmlu_${ISO639_3_LANG_CODE[${lang_code}]}_mcf:security_studies|5|0,lighteval|gmmlu_${ISO639_3_LANG_CODE[${lang_code}]}_mcf:sociology|5|0,lighteval|gmmlu_${ISO639_3_LANG_CODE[${lang_code}]}_mcf:us_foreign_policy|5|0,lighteval|gmmlu_${ISO639_3_LANG_CODE[${lang_code}]}_mcf:virology|5|0,lighteval|gmmlu_${ISO639_3_LANG_CODE[${lang_code}]}_mcf:world_religions|5|0"

    lighteval accelerate \
        "model_name=${MODEL_NAME},batch_size=1,dtype=bfloat16" \
        "${task}" \
        --custom-tasks "${CUSTOM_TASK_SCRIPT_DIR}/gmmlu.py" \
        --output-dir "${LOG_BASE_DIR}/${MODEL_ABBREV}/gmmlu_${lang_code}" \
        --use-chat-template
done

# ===== 6) 英文 Belebele =====
task="lighteval|belebele_eng_Latn_mcf|3|0"
task_name=$(echo "${task}" | cut -d'|' -f2 | cut -d'_' -f1)

lighteval accelerate \
    "model_name=${MODEL_NAME},batch_size=1,dtype=bfloat16" \
    "${task}" \
    --custom-tasks "${CUSTOM_TASK_SCRIPT_DIR}/belebele.py" \
    --output-dir "${LOG_BASE_DIR}/${MODEL_ABBREV}/${task_name}_en" \
    --use-chat-template

# ===== 7) 英文 MMLU =====
task="leaderboard|mmlu:abstract_algebra|5|0,leaderboard|mmlu:anatomy|5|0,leaderboard|mmlu:astronomy|5|0,leaderboard|mmlu:business_ethics|5|0,leaderboard|mmlu:clinical_knowledge|5|0,leaderboard|mmlu:college_biology|5|0,leaderboard|mmlu:college_chemistry|5|0,leaderboard|mmlu:college_computer_science|5|0,leaderboard|mmlu:college_mathematics|5|0,leaderboard|mmlu:college_medicine|5|0,leaderboard|mmlu:college_physics|5|0,leaderboard|mmlu:computer_security|5|0,leaderboard|mmlu:conceptual_physics|5|0,leaderboard|mmlu:econometrics|5|0,leaderboard|mmlu:electrical_engineering|5|0,leaderboard|mmlu:elementary_mathematics|5|0,leaderboard|mmlu:formal_logic|5|0,leaderboard|mmlu:global_facts|5|0,leaderboard|mmlu:high_school_biology|5|0,leaderboard|mmlu:high_school_chemistry|5|0,leaderboard|mmlu:high_school_computer_science|5|0,leaderboard|mmlu:high_school_european_history|5|0,leaderboard|mmlu:high_school_geography|5|0,leaderboard|mmlu:high_school_government_and_politics|5|0,leaderboard|mmlu:high_school_macroeconomics|5|0,leaderboard|mmlu:high_school_mathematics|5|0,leaderboard|mmlu:high_school_microeconomics|5|0,leaderboard|mmlu:high_school_physics|5|0,leaderboard|mmlu:high_school_psychology|5|0,leaderboard|mmlu:high_school_statistics|5|0,leaderboard|mmlu:high_school_us_history|5|0,leaderboard|mmlu:high_school_world_history|5|0,leaderboard|mmlu:human_aging|5|0,leaderboard|mmlu:human_sexuality|5|0,leaderboard|mmlu:international_law|5|0,leaderboard|mmlu:jurisprudence|5|0,leaderboard|mmlu:logical_fallacies|5|0,leaderboard|mmlu:machine_learning|5|0,leaderboard|mmlu:management|5|0,leaderboard|mmlu:marketing|5|0,leaderboard|mmlu:medical_genetics|5|0,leaderboard|mmlu:miscellaneous|5|0,leaderboard|mmlu:moral_disputes|5|0,leaderboard|mmlu:moral_scenarios|5|0,leaderboard|mmlu:nutrition|5|0,leaderboard|mmlu:philosophy|5|0,leaderboard|mmlu:prehistory|5|0,leaderboard|mmlu:professional_accounting|5|0,leaderboard|mmlu:professional_law|5|0,leaderboard|mmlu:professional_medicine|5|0,leaderboard|mmlu:professional_psychology|5|0,leaderboard|mmlu:public_relations|5|0,leaderboard|mmlu:security_studies|5|0,leaderboard|mmlu:sociology|5|0,leaderboard|mmlu:us_foreign_policy|5|0,leaderboard|mmlu:virology|5|0,leaderboard|mmlu:world_religions|5|0"

lighteval accelerate \
    "model_name=${MODEL_NAME},batch_size=1,dtype=bfloat16" \
    "${task}" \
    --output-dir "${LOG_BASE_DIR}/${MODEL_ABBREV}/mmlu_en" \
    --use-chat-template

echo "========================================"
echo "base 模型分类评测完成"
echo "结果目录: ${LOG_BASE_DIR}/${MODEL_ABBREV}"
echo "========================================"

conda deactivate