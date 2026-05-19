#!/usr/bin/env bash
set -e
# su - HwHiAiUser 
# cd /home/HwHiAiUser/cl_workspace/code/SSU/ssu-main/evaluation/scripts

# ===== NPU 环境 =====
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cl_eval

# ===== 设备控制 =====
export CUDA_VISIBLE_DEVICES=""
# 按需改成 0/1/2/3...，多开终端并行时非常重要
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"

# ===== HF =====
export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

export TRANSFORMERS_VERBOSITY=debug
export HF_HOME="/home/HwHiAiUser/cl_workspace/hf_cache"
export HF_HUB_CACHE="/home/HwHiAiUser/cl_workspace/hf_cache"
export HF_DATASETS_CACHE="/home/HwHiAiUser/cl_workspace/hf_cache"
export HF_DATASETS_TRUST_REMOTE_CODE=true

custom_task_script_dir="/home/HwHiAiUser/cl_workspace/code/SSU/ssu-main/evaluation/src"
log_base_dir="/home/HwHiAiUser/cl_workspace/eval_logs/post"

mkdir -p "${log_base_dir}"

# =========================
# 参数
# 用法:
#   bash evaluation/scripts/source_cls_single.sh <model_path> <langs>
#
# 例子:
#   bash evaluation/scripts/source_cls_single.sh \
#   /home/HwHiAiUser/cl_workspace/var/hf_cache/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775 \
#   ig,ha,ky,ne,am
#
#   bash evaluation/scripts/source_cls_single.sh \
#   /home/HwHiAiUser/cl_workspace/ckpt/single/ibo_Latn \
#   ig
# =========================
model_name="$1"
langs_csv="${2:-ig,ha,ky,ne,am}"

if [[ -z "${model_name}" ]]; then
    echo "Usage: $0 <model_path> [langs_csv]"
    echo "Example base:   $0 /path/to/base_model ig,ha,ky,ne,am"
    echo "Example single: $0 /home/HwHiAiUser/cl_workspace/ckpt/single/ibo_Latn ig"
    exit 1
fi

# 结果目录名：尽量稳定一点
if [[ "${model_name}" == *"/ckpt/"* ]]; then
    parent_name=$(basename "$(dirname "${model_name}")")
    current_name=$(basename "${model_name}")
    model_abbrev="${parent_name}_${current_name}"
else
    current_name=$(basename "${model_name}")
    # 如果是 snapshots/xxxx 这种路径，取上一级更有辨识度
    if [[ "${current_name}" == "snapshots" ]]; then
        model_abbrev=$(basename "$(dirname "${model_name}")")
    else
        model_abbrev="${current_name}"
    fi
fi

IFS=',' read -r -a lang_codes <<< "${langs_csv}"

declare -A lang_code_to_belebele_lang_code=(
    ["ne"]="npi_Deva"
    ["am"]="amh_Ethi"
    ["ha"]="hau_Latn"
    ["ig"]="ibo_Latn"
    ["ky"]="kir_Cyrl"
)

declare -A iso639_3_lang_code=(
    ["ne"]="npi"
    ["am"]="amh"
    ["ha"]="hau"
    ["ig"]="ibo"
    ["ky"]="kir"
)

for lang_code in "${lang_codes[@]}"; do
    echo "========================================"
    echo "Model : ${model_name}"
    echo "Lang  : ${lang_code}"
    echo "NPU   : ${ASCEND_RT_VISIBLE_DEVICES}"
    echo "========================================"

    # ---------- 1) 目标语言 Belebele ----------
    task="lighteval|belebele_${lang_code_to_belebele_lang_code[${lang_code}]}_mcf|3|0"
    task_name="belebele_${lang_code}"
    lighteval accelerate \
        "model_name=${model_name},batch_size=8,dtype=float16" \
        "${task}" \
        --custom-tasks "${custom_task_script_dir}/belebele.py" \
        --output-dir "${log_base_dir}/${model_abbrev}/${lang_code}/${task_name}" \
        --use-chat-template

    # ---------- 2) 目标语言 GMMLU ----------
    task="lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:abstract_algebra|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:anatomy|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:astronomy|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:business_ethics|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:clinical_knowledge|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:college_biology|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:college_chemistry|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:college_computer_science|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:college_mathematics|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:college_medicine|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:college_physics|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:computer_security|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:conceptual_physics|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:econometrics|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:electrical_engineering|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:elementary_mathematics|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:formal_logic|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:global_facts|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:high_school_biology|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:high_school_chemistry|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:high_school_computer_science|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:high_school_european_history|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:high_school_geography|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:high_school_government_and_politics|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:high_school_macroeconomics|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:high_school_mathematics|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:high_school_microeconomics|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:high_school_physics|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:high_school_psychology|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:high_school_statistics|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:high_school_us_history|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:high_school_world_history|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:human_aging|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:human_sexuality|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:international_law|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:jurisprudence|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:logical_fallacies|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:machine_learning|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:management|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:marketing|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:medical_genetics|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:miscellaneous|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:moral_disputes|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:moral_scenarios|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:nutrition|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:philosophy|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:prehistory|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:professional_accounting|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:professional_law|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:professional_medicine|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:professional_psychology|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:public_relations|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:security_studies|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:sociology|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:us_foreign_policy|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:virology|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:world_religions|5|0"
    task_name="gmmlu_${lang_code}"
    lighteval accelerate \
        "model_name=${model_name},batch_size=8,dtype=float16" \
        "${task}" \
        --custom-tasks "${custom_task_script_dir}/gmmlu.py" \
        --output-dir "${log_base_dir}/${model_abbrev}/${lang_code}/${task_name}" \
        --use-chat-template
done

# # ---------- 3) 英文 Belebele ----------
# task="lighteval|belebele_eng_Latn_mcf|3|0"
# task_name="belebele_en"
# lighteval accelerate \
#     "model_name=${model_name},batch_size=8,dtype=float16" \
#     "${task}" \
#     --custom-tasks "${custom_task_script_dir}/belebele.py" \
#     --output-dir "${log_base_dir}/${model_abbrev}/en/${task_name}" \
#     --use-chat-template

# # ---------- 4) 英文 MMLU ----------
# task="leaderboard|mmlu:abstract_algebra|5|0,leaderboard|mmlu:anatomy|5|0,leaderboard|mmlu:astronomy|5|0,leaderboard|mmlu:business_ethics|5|0,leaderboard|mmlu:clinical_knowledge|5|0,leaderboard|mmlu:college_biology|5|0,leaderboard|mmlu:college_chemistry|5|0,leaderboard|mmlu:college_computer_science|5|0,leaderboard|mmlu:college_mathematics|5|0,leaderboard|mmlu:college_medicine|5|0,leaderboard|mmlu:college_physics|5|0,leaderboard|mmlu:computer_security|5|0,leaderboard|mmlu:conceptual_physics|5|0,leaderboard|mmlu:econometrics|5|0,leaderboard|mmlu:electrical_engineering|5|0,leaderboard|mmlu:elementary_mathematics|5|0,leaderboard|mmlu:formal_logic|5|0,leaderboard|mmlu:global_facts|5|0,leaderboard|mmlu:high_school_biology|5|0,leaderboard|mmlu:high_school_chemistry|5|0,leaderboard|mmlu:high_school_computer_science|5|0,leaderboard|mmlu:high_school_european_history|5|0,leaderboard|mmlu:high_school_geography|5|0,leaderboard|mmlu:high_school_government_and_politics|5|0,leaderboard|mmlu:high_school_macroeconomics|5|0,leaderboard|mmlu:high_school_mathematics|5|0,leaderboard|mmlu:high_school_microeconomics|5|0,leaderboard|mmlu:high_school_physics|5|0,leaderboard|mmlu:high_school_psychology|5|0,leaderboard|mmlu:high_school_statistics|5|0,leaderboard|mmlu:high_school_us_history|5|0,leaderboard|mmlu:high_school_world_history|5|0,leaderboard|mmlu:human_aging|5|0,leaderboard|mmlu:human_sexuality|5|0,leaderboard|mmlu:international_law|5|0,leaderboard|mmlu:jurisprudence|5|0,leaderboard|mmlu:logical_fallacies|5|0,leaderboard|mmlu:machine_learning|5|0,leaderboard|mmlu:management|5|0,leaderboard|mmlu:marketing|5|0,leaderboard|mmlu:medical_genetics|5|0,leaderboard|mmlu:miscellaneous|5|0,leaderboard|mmlu:moral_disputes|5|0,leaderboard|mmlu:moral_scenarios|5|0,leaderboard|mmlu:nutrition|5|0,leaderboard|mmlu:philosophy|5|0,leaderboard|mmlu:prehistory|5|0,leaderboard|mmlu:professional_accounting|5|0,leaderboard|mmlu:professional_law|5|0,leaderboard|mmlu:professional_medicine|5|0,leaderboard|mmlu:professional_psychology|5|0,leaderboard|mmlu:public_relations|5|0,leaderboard|mmlu:security_studies|5|0,leaderboard|mmlu:sociology|5|0,leaderboard|mmlu:us_foreign_policy|5|0,leaderboard|mmlu:virology|5|0,leaderboard|mmlu:world_religions|5|0"
# task_name="mmlu_en"
# lighteval accelerate \
#     "model_name=${model_name},batch_size=8,dtype=float16" \
#     "${task}" \
#     --output-dir "${log_base_dir}/${model_abbrev}/en/${task_name}" \
#     --use-chat-template

echo "All done. Results saved under: ${log_base_dir}/${model_abbrev}"