#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# 名字建议：eval_qwen_autodl.sh
#
# 用法:
#   bash evaluation/scripts/eval_qwen_autodl.sh <model_path> <lang_code>
#
# 例子：测试 base 模型
#   bash evaluation/scripts/eval_qwen_autodl.sh \
#     /root/models/Qwen2.5-1.5B-Instruct ig
#
# 例子：测试训练后的模型
#   bash evaluation/scripts/eval_qwen_autodl.sh \
#     /root/autodl-tmp/ssu_workspace/ckpt/single_15b_newlangs/ibo_Latn ig
# =========================================================

# =========================================================
# 0) AutoDL CUDA 环境
# =========================================================

# AutoDL 当前只有 base 环境时，先用 base
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

# 单卡 4090D
export CUDA_VISIBLE_DEVICES=0

# 减少 CUDA 显存碎片
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# =========================================================
# 1) 路径配置：尽量放到数据盘 /root/autodl-tmp
# =========================================================

WORK_ROOT="/root/autodl-tmp/ssu_workspace"
CODE_ROOT="/root/ssu-github/ssu-main"

# HF 缓存全部放数据盘
export HF_HOME="${WORK_ROOT}/hf_cache"
export HF_HUB_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}"
export TRANSFORMERS_CACHE="${HF_HOME}"

# 评测日志也放数据盘
log_base_dir="${WORK_ROOT}/eval_logs/unified_eval"

# 临时目录也放数据盘，避免占系统盘
export TMPDIR="${WORK_ROOT}/tmp"

mkdir -p "${HF_HOME}" "${log_base_dir}" "${TMPDIR}"

# HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 不强制离线
unset HF_DATASETS_OFFLINE
unset TRANSFORMERS_OFFLINE
unset HF_HUB_OFFLINE

export TRANSFORMERS_VERBOSITY=debug
export HF_DATASETS_TRUST_REMOTE_CODE=true

# 自定义任务脚本路径
custom_task_script_dir="${CODE_ROOT}/evaluation/src"

# =========================================================
# 2) 参数
# =========================================================

model_name="${1:-}"
lang_code="${2:-}"

if [[ -z "${model_name}" || -z "${lang_code}" ]]; then
    echo "Usage: $0 <model_path> <lang_code>"
    echo "Example:"
    echo "  bash evaluation/scripts/eval_qwen_autodl.sh /root/models/Qwen2.5-1.5B-Instruct ig"
    exit 1
fi

# =========================================================
# 3) 环境检查
# =========================================================

echo "========================================"
echo "Environment check"
echo "CONDA_ENV=${CONDA_DEFAULT_ENV:-None}"
echo "python=$(which python)"
python -V
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "HF_HOME=${HF_HOME}"
echo "CODE_ROOT=${CODE_ROOT}"
echo "WORK_ROOT=${WORK_ROOT}"
echo "========================================"

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
print("gpu name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")
PY

nvidia-smi || echo "[WARN] 当前可能是无卡模式，nvidia-smi 不可用"

echo "========================================"
echo "Disk check"
df -h / /root/autodl-tmp /root/autodl-fs || true
echo "========================================"

# =========================================================
# 4) 模型缩写
# =========================================================

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
)

MODEL_OUT="${log_base_dir}/${model_abbrev}"
mkdir -p "${MODEL_OUT}"

echo "========================================"
echo "Unified evaluation start"
echo "MODEL=${model_name}"
echo "MODEL_ABBREV=${model_abbrev}"
echo "LANG=${lang_code}"
echo "OUT_DIR=${MODEL_OUT}"
echo "========================================"

cd "${CODE_ROOT}"

# =========================================================
# 说明：
# 4090D 24GB 先统一 batch_size=1，跑通后再改 2/4/8。
# 下面保留全部任务。
# 默认开启：MT、SUM、Belebele、IFEval、GSM8K。
# 默认注释：MT-Bench、GMMLU、MMLU，因为耗时长、容易先卡环境。
# =========================================================

# =========================================================
# 1) MT: en -> xx
# =========================================================
task="custom|mt:en2${lang_code}|3|0"
task_name="mt_en2${lang_code}"

lighteval accelerate \
    "model_name=${model_name},batch_size=1,dtype=bfloat16" \
    "${task}" \
    --custom-tasks "${custom_task_script_dir}/mt.py" \
    --save-details \
    --output-dir "${MODEL_OUT}/${lang_code}/${task_name}" \
    --use-chat-template

# =========================================================
# 2) MT: xx -> en
# =========================================================
task="custom|mt:${lang_code}2en|3|0"
task_name="mt_${lang_code}2en"

lighteval accelerate \
    "model_name=${model_name},batch_size=1,dtype=bfloat16" \
    "${task}" \
    --custom-tasks "${custom_task_script_dir}/mt.py" \
    --save-details \
    --output-dir "${MODEL_OUT}/${lang_code}/${task_name}" \
    --use-chat-template

# =========================================================
# 3) SUM: target language
# =========================================================
task="custom|sum:${lang_code}|0|1"
task_name="sum_${lang_code}"

lighteval accelerate \
    "model_name=${model_name},batch_size=1,dtype=bfloat16" \
    "${task}" \
    --custom-tasks "${custom_task_script_dir}/sum.py" \
    --save-details \
    --output-dir "${MODEL_OUT}/${lang_code}/${task_name}" \
    --use-chat-template

# =========================================================
# 4) SUM: English
# =========================================================
task="custom|sum:en|0|1"
task_name="sum_en"

lighteval accelerate \
    "model_name=${model_name},batch_size=1,dtype=bfloat16" \
    "${task}" \
    --custom-tasks "${custom_task_script_dir}/sum.py" \
    --save-details \
    --output-dir "${MODEL_OUT}/en/${task_name}" \
    --use-chat-template

# =========================================================
# 5) MT-Bench
# 暂时注释：这个任务比较慢，建议等前面任务跑通后再打开
# =========================================================
# task="extended|mt_bench|0|0"
# task_name="mtbench"
#
# lighteval accelerate \
#     "model_name=${model_name},batch_size=1,dtype=bfloat16" \
#     "${task}" \
#     --custom-tasks "${custom_task_script_dir}/mtbench.py" \
#     --save-details \
#     --output-dir "${MODEL_OUT}/${task_name}" \
#     --use-chat-template

# =========================================================
# 6) 目标语言 Belebele
# =========================================================
task="lighteval|belebele_${lang_code_to_belebele_lang_code[${lang_code}]}_mcf|3|0"
task_name="belebele_${lang_code}"

lighteval accelerate \
    "model_name=${model_name},batch_size=1,dtype=bfloat16" \
    "${task}" \
    --custom-tasks "${custom_task_script_dir}/belebele.py" \
    --output-dir "${MODEL_OUT}/${lang_code}/${task_name}" \
    --save-details \
    --use-chat-template

# =========================================================
# 7) 目标语言 GMMLU
# 暂时注释：任务很多，先不要一开始就跑，避免一次耗时太久。
# 要跑时取消下面整段注释。
# =========================================================
# task="lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:abstract_algebra|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:anatomy|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:astronomy|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:business_ethics|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:clinical_knowledge|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:college_biology|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:college_chemistry|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:college_computer_science|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:college_mathematics|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:college_medicine|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:college_physics|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:computer_security|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:conceptual_physics|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:econometrics|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:electrical_engineering|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:elementary_mathematics|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:formal_logic|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:global_facts|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:high_school_biology|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:high_school_chemistry|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:high_school_computer_science|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:high_school_european_history|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:high_school_geography|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:high_school_government_and_politics|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:high_school_macroeconomics|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:high_school_mathematics|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:high_school_microeconomics|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:high_school_physics|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:high_school_psychology|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:high_school_statistics|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:high_school_us_history|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:high_school_world_history|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:human_aging|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:human_sexuality|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:international_law|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:jurisprudence|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:logical_fallacies|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:machine_learning|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:management|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:marketing|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:medical_genetics|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:miscellaneous|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:moral_disputes|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:moral_scenarios|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:nutrition|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:philosophy|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:prehistory|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:professional_accounting|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:professional_law|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:professional_medicine|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:professional_psychology|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:public_relations|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:security_studies|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:sociology|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:us_foreign_policy|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:virology|5|0,lighteval|gmmlu_${iso639_3_lang_code[${lang_code}]}_mcf:world_religions|5|0"
# task_name="gmmlu_${lang_code}"
# lighteval accelerate \
#     "model_name=${model_name},batch_size=1,dtype=bfloat16" \
#     "${task}" \
#     --custom-tasks "${custom_task_script_dir}/gmmlu.py" \
#     --output-dir "${MODEL_OUT}/${lang_code}/${task_name}" \
#     --save-details \
#     --use-chat-template

# =========================================================
# 8) 英文 Belebele
# =========================================================
task="lighteval|belebele_eng_Latn_mcf|3|0"
task_name="belebele_en"

lighteval accelerate \
    "model_name=${model_name},batch_size=1,dtype=bfloat16" \
    "${task}" \
    --custom-tasks "${custom_task_script_dir}/belebele.py" \
    --output-dir "${MODEL_OUT}/en/${task_name}" \
    --save-details \
    --use-chat-template

# =========================================================
# 9) 英文 MMLU
# 暂时注释：任务较长，等前面任务正常后再跑。
# =========================================================
# task="leaderboard|mmlu:abstract_algebra|5|0,leaderboard|mmlu:anatomy|5|0,leaderboard|mmlu:astronomy|5|0,leaderboard|mmlu:business_ethics|5|0,leaderboard|mmlu:clinical_knowledge|5|0,leaderboard|mmlu:college_biology|5|0,leaderboard|mmlu:college_chemistry|5|0,leaderboard|mmlu:college_computer_science|5|0,leaderboard|mmlu:college_mathematics|5|0,leaderboard|mmlu:college_medicine|5|0,leaderboard|mmlu:college_physics|5|0,leaderboard|mmlu:computer_security|5|0,leaderboard|mmlu:conceptual_physics|5|0,leaderboard|mmlu:econometrics|5|0,leaderboard|mmlu:electrical_engineering|5|0,leaderboard|mmlu:elementary_mathematics|5|0,leaderboard|mmlu:formal_logic|5|0,leaderboard|mmlu:global_facts|5|0,leaderboard|mmlu:high_school_biology|5|0,leaderboard|mmlu:high_school_chemistry|5|0,leaderboard|mmlu:high_school_computer_science|5|0,leaderboard|mmlu:high_school_european_history|5|0,leaderboard|mmlu:high_school_geography|5|0,leaderboard|mmlu:high_school_government_and_politics|5|0,leaderboard|mmlu:high_school_macroeconomics|5|0,leaderboard|mmlu:high_school_mathematics|5|0,leaderboard|mmlu:high_school_microeconomics|5|0,leaderboard|mmlu:high_school_physics|5|0,leaderboard|mmlu:high_school_psychology|5|0,leaderboard|mmlu:high_school_statistics|5|0,leaderboard|mmlu:high_school_us_history|5|0,leaderboard|mmlu:high_school_world_history|5|0,leaderboard|mmlu:human_aging|5|0,leaderboard|mmlu:human_sexuality|5|0,leaderboard|mmlu:international_law|5|0,leaderboard|mmlu:jurisprudence|5|0,leaderboard|mmlu:logical_fallacies|5|0,leaderboard|mmlu:machine_learning|5|0,leaderboard|mmlu:management|5|0,leaderboard|mmlu:marketing|5|0,leaderboard|mmlu:medical_genetics|5|0,leaderboard|mmlu:miscellaneous|5|0,leaderboard|mmlu:moral_disputes|5|0,leaderboard|mmlu:moral_scenarios|5|0,leaderboard|mmlu:nutrition|5|0,leaderboard|mmlu:philosophy|5|0,leaderboard|mmlu:prehistory|5|0,leaderboard|mmlu:professional_accounting|5|0,leaderboard|mmlu:professional_law|5|0,leaderboard|mmlu:professional_medicine|5|0,leaderboard|mmlu:professional_psychology|5|0,leaderboard|mmlu:public_relations|5|0,leaderboard|mmlu:security_studies|5|0,leaderboard|mmlu:sociology|5|0,leaderboard|mmlu:us_foreign_policy|5|0,leaderboard|mmlu:virology|5|0,leaderboard|mmlu:world_religions|5|0"
# task_name="mmlu_en"
# lighteval accelerate \
#     "model_name=${model_name},batch_size=1,dtype=bfloat16" \
#     "${task}" \
#     --output-dir "${MODEL_OUT}/en/${task_name}" \
#     --save-details \
#     --use-chat-template

# =========================================================
# 10) IFEval
# =========================================================
python evaluation/src/ifeval.py \
    --model_name_or_path "${model_name}" \
    --output_dir "${MODEL_OUT}/ifeval" \
    --cache_dir "${HF_HOME}" \
    --apply_chat_template

# =========================================================
# 11) GSM8K
# =========================================================
python evaluation/src/gsm8k.py \
    --model_name_or_path "${model_name}" \
    --output_dir "${MODEL_OUT}/gsm8k_cot" \
    --cache_dir "${HF_HOME}" \
    --mode gsm8k_cot \
    --apply_chat_template

echo "========================================"
echo "ALL ENABLED TASKS FINISHED."
echo "Results saved under: ${MODEL_OUT}"
echo "========================================"