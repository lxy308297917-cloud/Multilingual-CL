#!/usr/bin/env bash
set -euo pipefail

# One manually queued probe, no retries. Keep full generation and batch budget.
test "${GPU_QUEUE_GPU_COUNT:-}" = 1
test -n "${CUDA_VISIBLE_DEVICES:-}"
[[ "$CUDA_VISIBLE_DEVICES" == GPU-* && "$CUDA_VISIBLE_DEVICES" != *,* ]]
run_id="${1:?usage: run_vllm_colocate_probe.sh NEW_RUN_ID}"
[[ "$run_id" =~ ^[a-zA-Z0-9_-]+$ ]]
root="${COPSD_PROJECT_ROOT:-$HOME}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
env_dir="$root/envs/copsd-vllm011-20261018"
output="$root/models/checkpoint/copsd_vllm_colocate_probe/$run_id"
test ! -e "$output"
export COPSD_OFFICIAL_SOURCE="$root/COPSD-ddp-teacher-unwrapped-20261018"
export COPSD_VLLM_UUID_COMPAT=1
export PYTHONPATH="$script_dir/compat/vllm011_uuid"
export COPSD_CACHE_ROOT="$root/cache/copsd_official"
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
# Four exported-source hashes are frozen from the existing formal TH manifest.
"$env_dir/bin/python" - "$root" "$script_dir" <<'CHECK'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1])
manifest=json.loads((root/'logs/copsd_official_10lang/th70_resume_uuid_20261019_v2.MANIFEST.json').read_text())
for name,expected in manifest['core_source_sha256'].items():
    assert hashlib.sha256((root/'COPSD-ddp-teacher-unwrapped-20261018'/name).read_bytes()).hexdigest()==expected,name
for name in ['model_lock','data']:
    item=manifest[name]
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256'],name
print('Frozen source/model/parent TH data hashes verified; SW probe data locked separately.',flush=True)
script_dir=Path(sys.argv[2])
probe=json.loads((script_dir/'vllm_colocate_probe_20261019.CONFIG.json').read_text())
for relative,expected in probe['code_sha256'].items():
    assert hashlib.sha256((script_dir/relative).read_bytes()).hexdigest()==expected,relative
assert hashlib.sha256((root/probe['data_relative_path']).read_bytes()).hexdigest()==probe['data_sha256']
print('Probe adapter/entry/launcher and SW data hashes verified.',flush=True)
CHECK
mkdir -p "$(dirname "$output")"
cd "$COPSD_OFFICIAL_SOURCE"
# A separate engineering probe starts from Base, never from SW/TH formal weights.
# micro1 × accumulation32 preserves effective batch32; two updates only.
# Sleep mode frees inference weights/cache during full-vocabulary KD/backward.
exec "$env_dir/bin/python" "$script_dir/vllm_colocate_probe_entry.py" \
  --train_language SW \
  --translated_data_path "$root/data/copsd_official_20261015/adapted_v2_ordered/project_train_2900_sw.json" \
  --model_name_or_path "$root/models/Qwen3-1.7B" \
  --learning_rate 5e-6 --max_grad_norm 0.1 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 32 \
  --gradient_checkpointing --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --max_steps 2 --save_steps 1 --logging_steps 1 \
  --output_dir "$output" --run_config "$run_id" \
  --max_completion_length 2048 --max_length 20000 \
  --attn_implementation sdpa --torch_dtype bfloat16 \
  --beta 0 --lmbda 1 --use_peft --lora_r 64 --lora_alpha 128 \
  --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --temperature 1.1 --top_p 0.95 --top_k 20 \
  --fixed_teacher --student_enable_thinking --jsd_token_clip 0.05 \
  --use_vllm --vllm_mode colocate --vllm_tensor_parallel_size 1 \
  --vllm_gpu_memory_utilization 0.4 --vllm_enable_sleep_mode true \
  --vllm_sync_frequency 1 --report_to tensorboard --seed 42 --data_seed 42 \
  --wandb_project COPSD-vllm-colocate-probe
