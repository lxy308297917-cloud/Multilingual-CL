#!/bin/bash

# source /path/to/envs/ssu_lighteval/bin/activate

# Configs
export OPENAI_API_KEY="your_openai_api_key"
export TRANSFORMERS_VERBOSITY=debug
export HF_HOME="./my_data/cache"
export HF_HUB_CACHE="./my_data/cache"
export HF_DATASETS_CACHE="./my_data/cache"
export HF_DATASETS_TRUST_REMOTE_CODE=true

MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
MODEL_ABBREV="Qwen2.5-0.5B-Instruct"
POSTFIX="source"

ROOT_DIR="$(pwd)"
OUTPUT_DIR="${ROOT_DIR}/evaluation/logs_ae2/source/${MODEL_ABBREV}"
mkdir -p "${OUTPUT_DIR}"

# postfix=$2
# if [ -z "$model_name" ] || [ -z "$postfix" ]; then
#     echo "Usage: $0 <model_name> <postfix>"
#     exit 1
# fi
# if [ "$model_name" == "allenai/OLMo-2-1124-7B-Instruct" ]; then
#     model_abbrev="OLMo-2-1124-7B-Instruct"
# elif [ "$model_name" == "allenai/OLMo-2-1124-13B-Instruct" ]; then
#     model_abbrev="OLMo-2-1124-13B-Instruct"
# else
#     echo "Unsupported model name: $model_name"
#     exit 1
# fi

# Run evaluation
cd evaluation/src || exit 1
python ae2.py \
    --model_name_or_path "${MODEL_NAME}" \
    --model_abbrev "${MODEL_ABBREV}" \
    --annotators_config "alpaca_eval_gpt4.1-nano.yml" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size 8 \
    --skip_inference \
    --do_eval \
    --postfix "${POSTFIX}"
