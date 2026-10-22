#!/usr/bin/env bash
set -euo pipefail
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")"
python diagnose.py
python diagnose.py --name ExistingFullDense --model /data0/users/lixiaoyang/models/checkpoint/sw_recipe_fft_20261010/aya80_mt20/final
