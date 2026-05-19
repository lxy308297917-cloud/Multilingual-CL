# 名字叫做：eval_target.sh

#!/usr/bin/env bash
set -e

# su - HwHiAiUser 
# cd /home/HwHiAiUser/cl_workspace/code/SSU/ssu-main/evaluation/scripts

# =========================================================
# 用法:
#   bash evaluation/scripts/eval_target.sh <model_path> <lang_code>
#
# 例子:
#   bash evaluation/scripts/eval_target.sh \
#   /home/HwHiAiUser/cl_workspace/var/hf_cache/Qwen2.5-1.5B-Instruct ig
#
#   bash evaluation/scripts/eval_target.sh \
#   /home/HwHiAiUser/cl_workspace/ckpt/fft_ssu_seq/ibo_Latn ig
# =========================================================

# ===== NPU 环境 =====
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cl_eval

# ===== 设备控制 =====
export CUDA_VISIBLE_DEVICES=""
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"

# ===== HF =====
export HF_ENDPOINT=https://hf-mirror.com

# 这里默认和你当前能跑通的环境一致：不强制离线
unset HF_DATASETS_OFFLINE
unset TRANSFORMERS_OFFLINE
unset HF_HUB_OFFLINE

export TRANSFORMERS_VERBOSITY=debug
export HF_HOME="/home/HwHiAiUser/cl_workspace/var/hf_cache"
export HF_HUB_CACHE="/home/HwHiAiUser/cl_workspace/var/hf_cache"
export HF_DATASETS_CACHE="/home/HwHiAiUser/cl_workspace/var/hf_cache"
export HF_DATASETS_TRUST_REMOTE_CODE=true

# ===== 路径 =====
custom_task_script_dir="/home/HwHiAiUser/cl_workspace/code/SSU/ssu-main/evaluation/src"
log_base_dir="/home/HwHiAiUser/cl_workspace/eval_logs/unified_eval"
mkdir -p "${log_base_dir}"

# ===== 参数 =====
model_name="$1"
lang_code="$2"

if [[ -z "${model_name}" || -z "${lang_code}" ]]; then
    echo "Usage: $0 <model_path> <lang_code>"
    exit 1
fi

# ===== 模型缩写 =====
if [[ "${model_name}" == *"/ckpt/"* ]]; then
    parent_name=$(basename "$(dirname "${model_name}")")
    current_name=$(basename "${model_name}")
    model_abbrev="${parent_name}_${current_name}"
else
    current_name=$(basename "${model_name}")
    if [[ "${current_name}" == "snapshots" ]]; then
        model_abbrev=$(basename "$(dirname "${model_name}")")
    else
        model_abbrev="${current_name}"
    fi
fi

declare -A lang_code_to_belebele_lang_code=(
    ["ne"]="npi_Deva"
    ["am"]="amh_Ethi"
    ["ha"]="hau_Latn"
    ["ig"]="ibo_Latn"
    ["ky"]="kir_Cyrl"
    ["da"]="dan_Latn"
    ["is"]="isl_Latn"
    ["no"]="nob_Latn"
    ["fil"]="tgl_Latn"
    ["ro"]="ron_Latn"
    ["id"]="ind_Latn"
    ["bn"]="ben_Beng"
    ["el"]="ell_Grek"
    ["he"]="heb_Hebr"
    ["ko"]="kor_Hang"
    ["lt"]="lit_Latn"
    ["ms"]="zsm_Latn"
    ["uk"]="ukr_Cyrl"
    )
declare -A iso639_3_lang_code=(
    ["ne"]="npi"
    ["am"]="amh"
    ["ha"]="hau"
    ["ig"]="ibo"
    ["ky"]="kir"
    ["da"]="dan"
    ["is"]="isl"
    ["no"]="nob"
    ["fil"]="fil"
    ["ro"]="ron"
    ["id"]="ind"
    ["bn"]="ben"
    ["el"]="ell"
    ["he"]="heb"
    ["ko"]="kor"
    ["lt"]="lit"
    ["ms"]="zsm"
    ["uk"]="ukr"
    )
MODEL_OUT="${log_base_dir}/${model_abbrev}"
mkdir -p "${MODEL_OUT}"

echo "========================================"
echo "Unified evaluation start"
echo "MODEL=${model_name}"
echo "MODEL_ABBREV=${model_abbrev}"
echo "LANG=${lang_code}"
echo "NPU=${ASCEND_RT_VISIBLE_DEVICES}"
echo "OUT_DIR=${MODEL_OUT}"
echo "========================================"

# # =========================================================
# # 1) MT: en -> xx
# # =========================================================
# task="custom|mt:en2${lang_code}|3|0"
# task_name="mt_en2${lang_code}"

# lighteval accelerate \
#     "model_name=${model_name},batch_size=4,dtype=bfloat16" \
#     "${task}" \
#     --custom-tasks "${custom_task_script_dir}/mt.py" \
#     --save-details \
#     --output-dir "${MODEL_OUT}/${lang_code}/${task_name}" \
#     --use-chat-template

# # =========================================================
# # 2) MT: xx -> en
# # =========================================================
# task="custom|mt:${lang_code}2en|3|0"
# task_name="mt_${lang_code}2en"

# lighteval accelerate \
#     "model_name=${model_name},batch_size=4,dtype=bfloat16" \
#     "${task}" \
#     --custom-tasks "${custom_task_script_dir}/mt.py" \
#     --save-details \
#     --output-dir "${MODEL_OUT}/${lang_code}/${task_name}" \
#     --use-chat-template

# # =========================================================
# # 3) SUM: target language
# # =========================================================
# task="custom|sum:${lang_code}|0|1"
# task_name="sum_${lang_code}"

# lighteval accelerate \
#     "model_name=${model_name},batch_size=4,dtype=bfloat16" \
#     "${task}" \
#     --custom-tasks "${custom_task_script_dir}/sum.py" \
#     --save-details \
#     --output-dir "${MODEL_OUT}/${lang_code}/${task_name}" \
#     --use-chat-template

# # =========================================================
# # 4) SUM: English
# # =========================================================
# task="custom|sum:en|0|1"
# task_name="sum_en"

# lighteval accelerate \
#     "model_name=${model_name},batch_size=4,dtype=bfloat16" \
#     "${task}" \
#     --custom-tasks "${custom_task_script_dir}/sum.py" \
#     --save-details \
#     --output-dir "${MODEL_OUT}/en/${task_name}" \
#     --use-chat-template

# # =========================================================
# # 5) MT-Bench
# # =========================================================
# task="extended|mt_bench|0|0"
# task_name="mtbench"

# lighteval accelerate \
#     "model_name=${model_name},batch_size=2,dtype=bfloat16" \
#     "${task}" \
#     --custom-tasks "${custom_task_script_dir}/mtbench.py" \
#     --save-details \
#     --output-dir "${MODEL_OUT}/${task_name}" \
#     --use-chat-template

# =========================================================
# 6) 目标语言 Belebele（先注释）
# =========================================================
task="lighteval|belebele_${lang_code_to_belebele_lang_code[${lang_code}]}_mcf|3|0"
task_name="belebele_${lang_code}"
lighteval accelerate \
    "model_name=${model_name},batch_size=8,dtype=bfloat16" \
    "${task}" \
    --custom-tasks "${custom_task_script_dir}/belebele.py" \
    --output-dir "${MODEL_OUT}/${lang_code}/${task_name}" \
    --use-chat-template \
    --save-details

# =========================================================
# 7) 目标语言 GMMLU（先注释）
# =========================================================
task="lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:abstract_algebra|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:anatomy|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:astronomy|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:business_ethics|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:clinical_knowledge|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:college_biology|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:college_chemistry|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:college_computer_science|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:college_mathematics|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:college_medicine|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:college_physics|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:computer_security|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:conceptual_physics|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:econometrics|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:electrical_engineering|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:elementary_mathematics|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:formal_logic|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:global_facts|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:high_school_biology|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:high_school_chemistry|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:high_school_computer_science|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:high_school_european_history|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:high_school_geography|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:high_school_government_and_politics|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:high_school_macroeconomics|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:high_school_mathematics|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:high_school_microeconomics|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:high_school_physics|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:high_school_psychology|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:high_school_statistics|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:high_school_us_history|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:high_school_world_history|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:human_aging|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:human_sexuality|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:international_law|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:jurisprudence|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:logical_fallacies|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:machine_learning|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:management|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:marketing|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:medical_genetics|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:miscellaneous|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:moral_disputes|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:moral_scenarios|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:nutrition|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:philosophy|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:prehistory|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:professional_accounting|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:professional_law|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:professional_medicine|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:professional_psychology|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:public_relations|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:security_studies|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:sociology|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:us_foreign_policy|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:virology|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:world_religions|5|0"
task_name="gmmlu_${lang_code}"
lighteval accelerate \
    "model_name=${model_name},batch_size=8,dtype=bfloat16" \
    "${task}" \
    --custom-tasks "${custom_task_script_dir}/gmmlu.py" \
    --output-dir "${MODEL_OUT}/${lang_code}/${task_name}" \
    --use-chat-template \
    --save-details

# # =========================================================
# # 8) 英文 Belebele（先注释）
# # =========================================================
# task="lighteval|belebele_eng_Latn_mcf|3|0"
# task_name="belebele_en"
# lighteval accelerate \
#     "model_name=${model_name},batch_size=8,dtype=bfloat16" \
#     "${task}" \
#     --custom-tasks "${custom_task_script_dir}/belebele.py" \
#     --output-dir "${MODEL_OUT}/en/${task_name}" \
#     --use-chat-template

# # =========================================================
# # 9) 英文 MMLU（先注释）
# # =========================================================
# task="leaderboard|mmlu:abstract_algebra|5|0,leaderboard|mmlu:anatomy|5|0,leaderboard|mmlu:astronomy|5|0,leaderboard|mmlu:business_ethics|5|0,leaderboard|mmlu:clinical_knowledge|5|0,leaderboard|mmlu:college_biology|5|0,leaderboard|mmlu:college_chemistry|5|0,leaderboard|mmlu:college_computer_science|5|0,leaderboard|mmlu:college_mathematics|5|0,leaderboard|mmlu:college_medicine|5|0,leaderboard|mmlu:college_physics|5|0,leaderboard|mmlu:computer_security|5|0,leaderboard|mmlu:conceptual_physics|5|0,leaderboard|mmlu:econometrics|5|0,leaderboard|mmlu:electrical_engineering|5|0,leaderboard|mmlu:elementary_mathematics|5|0,leaderboard|mmlu:formal_logic|5|0,leaderboard|mmlu:global_facts|5|0,leaderboard|mmlu:high_school_biology|5|0,leaderboard|mmlu:high_school_chemistry|5|0,leaderboard|mmlu:high_school_computer_science|5|0,leaderboard|mmlu:high_school_european_history|5|0,leaderboard|mmlu:high_school_geography|5|0,leaderboard|mmlu:high_school_government_and_politics|5|0,leaderboard|mmlu:high_school_macroeconomics|5|0,leaderboard|mmlu:high_school_mathematics|5|0,leaderboard|mmlu:high_school_microeconomics|5|0,leaderboard|mmlu:high_school_physics|5|0,leaderboard|mmlu:high_school_psychology|5|0,leaderboard|mmlu:high_school_statistics|5|0,leaderboard|mmlu:high_school_us_history|5|0,leaderboard|mmlu:high_school_world_history|5|0,leaderboard|mmlu:human_aging|5|0,leaderboard|mmlu:human_sexuality|5|0,leaderboard|mmlu:international_law|5|0,leaderboard|mmlu:jurisprudence|5|0,leaderboard|mmlu:logical_fallacies|5|0,leaderboard|mmlu:machine_learning|5|0,leaderboard|mmlu:management|5|0,leaderboard|mmlu:marketing|5|0,leaderboard|mmlu:medical_genetics|5|0,leaderboard|mmlu:miscellaneous|5|0,leaderboard|mmlu:moral_disputes|5|0,leaderboard|mmlu:moral_scenarios|5|0,leaderboard|mmlu:nutrition|5|0,leaderboard|mmlu:philosophy|5|0,leaderboard|mmlu:prehistory|5|0,leaderboard|mmlu:professional_accounting|5|0,leaderboard|mmlu:professional_law|5|0,leaderboard|mmlu:professional_medicine|5|0,leaderboard|mmlu:professional_psychology|5|0,leaderboard|mmlu:public_relations|5|0,leaderboard|mmlu:security_studies|5|0,leaderboard|mmlu:sociology|5|0,leaderboard|mmlu:us_foreign_policy|5|0,leaderboard|mmlu:virology|5|0,leaderboard|mmlu:world_religions|5|0"
# task_name="mmlu_en"
# lighteval accelerate \
#     "model_name=${model_name},batch_size=8,dtype=bfloat16" \
#     "${task}" \
#     --output-dir "${MODEL_OUT}/en/${task_name}" \
#     --use-chat-template

# # =========================================================
# # 10) IFEval（用你的自定义脚本）
# # =========================================================
# python evaluation/src/ifeval.py \
#     --model_name_or_path "${model_name}" \
#     --output_dir "${MODEL_OUT}/ifeval" \
#     --cache_dir /home/HwHiAiUser/cl_workspace/var/hf_cache \
#     --apply_chat_template

# # =========================================================
# # 11) GSM8K（用你的自定义脚本）
# # =========================================================
# python evaluation/src/gsm8k.py \
#     --model_name_or_path "${model_name}" \
#     --output_dir "${MODEL_OUT}/gsm8k_cot" \
#     --cache_dir /home/HwHiAiUser/cl_workspace/var/hf_cache \
#     --mode gsm8k_cot \
#     --apply_chat_template

echo "========================================"
echo "ALL TASKS FINISHED."
echo "Results saved under: ${MODEL_OUT}"
echo "========================================"