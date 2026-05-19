#!/usr/bin/env bash
set -e

# ===== NPU 环境 =====
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cl_eval

# ===== 设备控制 =====
export CUDA_VISIBLE_DEVICES=""
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-1}"

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

# ===== 路径 =====
model_name="/home/HwHiAiUser/cl_workspace/var/hf_cache/Qwen2.5-3B-Instruct"
custom_task_script_dir="/home/HwHiAiUser/cl_workspace/code/SSU/ssu-main/evaluation/src"
log_base_dir="/home/HwHiAiUser/cl_workspace/eval_logs/base3b_full_gmmlu"

mkdir -p "${log_base_dir}"

# ===== 结果目录名 =====
model_abbrev="$(basename "${model_name}")"

# ===== 只跑缺的两个语言 =====
langs_csv="${1:-ky,ne}"
IFS=',' read -r -a lang_codes <<< "${langs_csv}"

declare -A iso639_3_lang_code=(
    ["ne"]="npi"
    ["am"]="amh"
    ["ha"]="hau"
    ["ig"]="ibo"
    ["ky"]="kir"
)

# ===== 全量 GMMLU 子任务 =====
subjects="abstract_algebra,anatomy,astronomy,business_ethics,clinical_knowledge,college_biology,college_chemistry,college_computer_science,college_mathematics,college_medicine,college_physics,computer_security,conceptual_physics,econometrics,electrical_engineering,elementary_mathematics,formal_logic,global_facts,high_school_biology,high_school_chemistry,high_school_computer_science,high_school_european_history,high_school_geography,high_school_government_and_politics,high_school_macroeconomics,high_school_mathematics,high_school_microeconomics,high_school_physics,high_school_psychology,high_school_statistics,high_school_us_history,high_school_world_history,human_aging,human_sexuality,international_law,jurisprudence,logical_fallacies,machine_learning,management,marketing,medical_genetics,miscellaneous,moral_disputes,moral_scenarios,nutrition,philosophy,prehistory,professional_accounting,professional_law,professional_medicine,professional_psychology,public_relations,security_studies,sociology,us_foreign_policy,virology,world_religions"

for lang_code in "${lang_codes[@]}"; do
    echo "========================================"
    echo "Model : ${model_name}"
    echo "Lang  : ${lang_code}"
    echo "NPU   : ${ASCEND_RT_VISIBLE_DEVICES}"
    echo "========================================"

    task="lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:${subjects}|5|0"
    task_name="gmmlu_${lang_code}"

    lighteval accelerate \
        "model_name=${model_name},batch_size=8,dtype=float16" \
        "${task}" \
        --custom-tasks "${custom_task_script_dir}/gmmlu.py" \
        --output-dir "${log_base_dir}/${model_abbrev}/${lang_code}/${task_name}" \
        --use-chat-template
done

echo "All done. Results saved under: ${log_base_dir}/${model_abbrev}"