#!/usr/bin/env bash
set -euo pipefail
source /data0/users/lixiaoyang/.local/miniforge3/etc/profile.d/conda.sh
conda activate /data0/users/lixiaoyang/ssu-migration/envs/cl
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
export NLTK_DATA=/data0/users/lixiaoyang/nltk_data
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")"
exec python -u gpu_worker.py "$@"
