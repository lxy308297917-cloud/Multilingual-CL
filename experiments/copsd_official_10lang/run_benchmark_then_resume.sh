#!/usr/bin/env bash
set -euo pipefail
test "${GPU_QUEUE_GPU_COUNT:-}" = 4
test -n "${CUDA_VISIBLE_DEVICES:-}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec /data0/users/lixiaoyang/envs/copsd-vllm011-20261018/bin/python \
  "$script_dir/benchmark_then_resume.py" "$@"
