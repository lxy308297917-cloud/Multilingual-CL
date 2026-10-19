#!/usr/bin/env bash
set -euo pipefail
test "${GPU_QUEUE_GPU_COUNT:-}" = 1
root="${COPSD_PROJECT_ROOT:-$HOME}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# A separate frozen development protocol: never combine these with HF scores.
for lang in sw th te ja bn en; do
  bash "$script_dir/run_dev_vllm_17b.sh" "$lang" base "${lang}_17b_dev100_vllm_base_20261020_v1"
done
sw="$root/models/checkpoint/copsd_official_17b_single_v1/sw_17b_project2900_seed42_20261015_v1"
th="$root/models/checkpoint/copsd_official_17b_single_4gpu_v1/th_17b_project2900_seed42_4gpu_20261018_v2"
th_final="$root/models/checkpoint/copsd_official_17b_single_4gpu_v1/th_17b_project2900_seed42_4gpu_resume70_torch28_uuid_20261019_v2"
bash "$script_dir/run_dev_vllm_17b.sh" sw adapter sw_17b_dev100_vllm_ckpt20_20261020_v1 "$sw/checkpoint-20"
bash "$script_dir/run_dev_vllm_17b.sh" sw adapter sw_17b_dev100_vllm_ckpt100_20261020_v1 "$sw/checkpoint-100"
bash "$script_dir/run_dev_vllm_17b.sh" th adapter th_17b_dev100_vllm_ckpt20_20261020_v1 "$th/checkpoint-20"
bash "$script_dir/run_dev_vllm_17b.sh" th adapter th_17b_dev100_vllm_ckpt100_20261020_v1 "$th_final/checkpoint-100"
