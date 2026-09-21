#!/usr/bin/env bash
set -euo pipefail
exec /root/miniconda3/envs/cl/bin/python /root/autodl-fs/ssu-project/experiments/sw_fineweb2_aya_v1/scripts/evaluate_cl.py "$@"
