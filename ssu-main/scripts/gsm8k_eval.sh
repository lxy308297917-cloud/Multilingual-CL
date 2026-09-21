#!/usr/bin/env bash
set -euo pipefail
exec /root/autodl-tmp/envs/gsm8k_harness048/bin/python /root/autodl-fs/ssu-project/experiments/ig_gsm8k_harness048_v1/scripts/run.py "$@"
