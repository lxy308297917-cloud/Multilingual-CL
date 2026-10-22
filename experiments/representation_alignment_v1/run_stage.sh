#!/usr/bin/env bash
set -euo pipefail
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NLTK_DATA=/data0/users/lixiaoyang/nltk_data
cd "$(dirname "$0")"
python launch.py "$@"
