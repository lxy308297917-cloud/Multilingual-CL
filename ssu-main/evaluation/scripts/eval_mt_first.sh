#!/usr/bin/env bash
set -e

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
export HF_HOME="/home/HwHiAiUser/cl_workspace/var/hf_cache"
export HF_HUB_CACHE="/home/HwHiAiUser/cl_workspace/var/hf_cache"
export HF_DATASETS_CACHE="/home/HwHiAiUser/cl_workspace/var/hf_cache"
export HF_DATASETS_TRUST_REMOTE_CODE=true

custom_task_script_dir="/home/HwHiAiUser/cl_workspace/code/SSU/ssu-main/evaluation/src"
log_base_dir="/home/HwHiAiUser/cl_workspace/eval_logs/mt_first"
mkdir -p "${log_base_dir}"

# ===== 参数 =====
# 用法:
#   bash evaluation/scripts/eval_mt_first.sh <model_path> <lang_code>
#
# 例子:
#   bash evaluation/scripts/eval_mt_first.sh \
#   /home/HwHiAiUser/cl_workspace/var/hf_cache/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775 \
#   ig

model_name="$1"
lang_code="$2"

if [[ -z "${model_name}" || -z "${lang_code}" ]]; then
    echo "Usage: $0 <model_path> <lang_code>"
    exit 1
fi

# ===== 模型缩写，方便整理日志 =====
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

echo "========================================"
echo "MT 测试开始"
echo "MODEL=${model_name}"
echo "LANG=${lang_code}"
echo "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
echo "LOG_DIR=${log_base_dir}/${model_abbrev}/${lang_code}"
echo "========================================"

# ===== 1) English -> target =====
task="custom|mt:en2${lang_code}|3|0"
task_name="mt_en2${lang_code}"

lighteval accelerate \
    "model_name=${model_name},batch_size=4,dtype=float16" \
    "${task}" \
    --custom-tasks "${custom_task_script_dir}/mt.py" \
    --save-details \
    --output-dir "${log_base_dir}/${model_abbrev}/${lang_code}/${task_name}" \
    --use-chat-template

# ===== 2) target -> English =====
task="custom|mt:${lang_code}2en|3|0"
task_name="mt_${lang_code}2en"

lighteval accelerate \
    "model_name=${model_name},batch_size=4,dtype=float16" \
    "${task}" \
    --custom-tasks "${custom_task_script_dir}/mt.py" \
    --save-details \
    --output-dir "${log_base_dir}/${model_abbrev}/${lang_code}/${task_name}" \
    --use-chat-template

echo "========================================"
echo "MT 测试完成"
echo "结果目录：${log_base_dir}/${model_abbrev}/${lang_code}"
echo "========================================"