#!/usr/bin/env bash
set -euo pipefail
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/../.." && pwd)"
python "$script_dir/smoke_qwen.py" --audit "$repo_dir/docs/research/paired_capability_sparse/SOURCE_AUDIT.json" --resume-checkpoint /data0/users/lixiaoyang/models/checkpoint/paired_capability_sparse_20261021/technical_v1/resume.pt --output /data0/users/lixiaoyang/models/checkpoint/paired_capability_sparse_20261021/technical_recovery_v2
