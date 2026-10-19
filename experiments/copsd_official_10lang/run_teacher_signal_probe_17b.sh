#!/usr/bin/env bash
set -euo pipefail
test "${GPU_QUEUE_GPU_COUNT:-}" = 1
[[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* && "$CUDA_VISIBLE_DEVICES" != *,* ]]
lang="${1:?usage: run_teacher_signal_probe_17b.sh LANG NEW_RUN_ID 2|100 [CHECKPOINT]}"
run_id="${2:?new run ID required}"
steps="${3:?2 for runtime validation,100 for formal training}"
resume="${4:-}"
case "$lang" in sw|te|th|bn|zh|ja|ru|de|fr|es) ;; *) exit 2 ;; esac
[[ "$run_id" =~ ^[a-zA-Z0-9_-]+$ ]]
[[ "$steps" == 2 ]]
root="${COPSD_PROJECT_ROOT:-$HOME}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
env_dir="$root/envs/copsd-vllm011-20261018"
track=copsd_vllm_update_batch_single_v1
save_steps=5
logging_steps=2
if [[ "$steps" == 2 ]]; then
  track=copsd_vllm_teacher_signal_checks
  save_steps=1
  logging_steps=1
fi
output="$root/models/checkpoint/$track/$run_id"
test ! -e "$output"
export COPSD_OFFICIAL_SOURCE="$root/COPSD-ddp-teacher-unwrapped-20261018"
export COPSD_VLLM_UUID_COMPAT=1
export PYTHONPATH="$script_dir/compat/vllm011_uuid"
export COPSD_CACHE_ROOT="$root/cache/copsd_official"
export WANDB_MODE=offline
export WANDB_DIR="$root/logs/copsd_official_10lang/wandb"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
"$env_dir/bin/python" - "$root" "$script_dir" "$lang" <<'CHECK'
import hashlib,json,sys
from pathlib import Path
root,script_dir=map(Path,sys.argv[1:3]);lang=sys.argv[3]
lock=json.loads((script_dir/'teacher_signal_runtime_20261020.CONFIG.json').read_text())
for name,expected in lock['code_sha256'].items():
    assert hashlib.sha256((script_dir/name).read_bytes()).hexdigest()==expected,name
parent=json.loads((root/'logs/copsd_official_10lang/th70_resume_uuid_20261019_v2.MANIFEST.json').read_text())
for name,expected in parent['core_source_sha256'].items():
    assert hashlib.sha256((root/'COPSD-ddp-teacher-unwrapped-20261018'/name).read_bytes()).hexdigest()==expected,name
assert hashlib.sha256(Path(parent['model_lock']['path']).read_bytes()).hexdigest()==parent['model_lock']['sha256']
data=root/f'data/copsd_official_20261015/adapted_v2_ordered/project_train_2900_{lang}.json'
assert hashlib.sha256(data.read_bytes()).hexdigest()==lock['data_sha256'][lang],lang
print('Frozen runtime/core/model/data identities verified.',flush=True)
CHECK
args=()
if [[ -n "$resume" ]]; then
  test -s "$resume/trainer_state.json"
  test -s "$resume/optimizer.pt"
  test -s "$resume/scheduler.pt"
  test -s "$resume/rng_state.pth"
  args+=(--resume_from_checkpoint "$resume")
  reference="$(dirname "$resume")/COLLOCATE_PROBE_UPDATES_DONE.json"
  if [[ -f "$reference" ]]; then export COPSD_EXPECT_INITIAL_SYNC_JSON="$reference"; fi
fi
mkdir -p "$(dirname "$output")" "$root/logs/copsd_official_10lang"
telemetry="$root/logs/copsd_official_10lang/$run_id.gpu.csv"
test ! -e "$telemetry"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 1 > "$telemetry" &
telemetry_pid=$!
trap 'kill "$telemetry_pid" 2>/dev/null || true; wait "$telemetry_pid" 2>/dev/null || true' EXIT
cd "$COPSD_OFFICIAL_SOURCE"
"$env_dir/bin/python" "$script_dir/teacher_signal_probe_entry.py" \
  --train_language "${lang^^}" \
  --translated_data_path "$root/data/copsd_official_20261015/adapted_v2_ordered/project_train_2900_${lang}.json" \
  --model_name_or_path "$root/models/Qwen3-1.7B" \
  --learning_rate 5e-6 --max_grad_norm 0.1 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 32 \
  --gradient_checkpointing --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --max_steps "$steps" --save_steps "$save_steps" --logging_steps "$logging_steps" \
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
  --wandb_project COPSD-vllm-single "${args[@]}"
