#!/usr/bin/env bash
set -euo pipefail
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUBLAS_WORKSPACE_CONFIG=:4096:8
export NLTK_DATA=/data0/users/lixiaoyang/nltk_data
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python "$script_dir/series.py" --output /data0/users/lixiaoyang/models/checkpoint/paired_capability_sparse_20261021/comparison/matched_v1 --base-cache /data0/users/lixiaoyang/models/checkpoint/paired_capability_sparse_20261021/comparison/base_mastery_v1/cache_000000
