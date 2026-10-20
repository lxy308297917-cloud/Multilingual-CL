#!/usr/bin/env bash
set -euo pipefail
root="${COPSD_PROJECT_ROOT:-$HOME}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
args=("${1:?ja80,th,te,ja20}" --root "$root")
if [[ "${COPSD_DISK_RECOVERY_PREPARE_ONLY:-0}" == 1 ]]; then args+=(--prepare-only); fi
exec "$root/envs/copsd-vllm011-20261018/bin/python" "$script_dir/disk_recovery_driver.py" "${args[@]}"
