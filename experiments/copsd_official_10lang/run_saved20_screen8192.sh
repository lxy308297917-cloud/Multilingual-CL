#!/usr/bin/env bash
set -euo pipefail
root="${COPSD_PROJECT_ROOT:-$HOME}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
args=(--root "$root" --language "${1:?language}")
if [[ "${COPSD_SCREEN_PREPARE_ONLY:-0}" == 1 ]]; then args+=(--prepare-only); fi
exec "$root/envs/copsd-vllm011-20261018/bin/python" "$script_dir/evaluate_saved20_screen8192.py" "${args[@]}"
