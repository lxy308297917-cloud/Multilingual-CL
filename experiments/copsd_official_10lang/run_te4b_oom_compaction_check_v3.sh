#!/usr/bin/env bash
set -euo pipefail
root="${COPSD_PROJECT_ROOT:-$HOME}"
code="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec "$root/envs/copsd-vllm011-20261018/bin/python" "$code/check_te4b_oom_compaction_cuda_v3.py"
