#!/usr/bin/env bash
set -euo pipefail
test "${GPU_QUEUE_GPU_COUNT:-}" = 1
test -n "${CUDA_VISIBLE_DEVICES:-}"
benchmark="${1:?usage: run_english_math_17b.sh gsm8k|mgsm_rev2 base|adapter RUN_ID [CHECKPOINT]}"
mode="${2:?mode required}"
run_id="${3:?run ID required}"
checkpoint="${4:-}"
[[ "$run_id" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
root=/data0/users/lixiaoyang
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
test -f "$root/models/Qwen3-1.7B/MODEL_LOCK.json"
case "$benchmark" in
  gsm8k)
    input="$root/data/copsd_official_20261015/gsm8k_project_en_zeroshot_v1/gsm8k_en_test_1319.json"
    # Reuse the frozen generic id/question/answer JSON loader. Its historical
    # CLI name is polymath_project; the actual benchmark is declared in metadata.
    kind=polymath_project
    expected=1319
    ;;
  mgsm_rev2)
    input="$root/data/math_single_10lang_20261015/mgsm_rev2/MGSM-Rev2/mgsm_en.tsv"
    kind=mgsm_rev2
    expected=250
    ;;
  *) exit 2 ;;
esac
test -s "$input"
args=()
case "$mode" in
  base) test -z "$checkpoint" ;;
  adapter) test -s "$checkpoint/adapter_model.safetensors"; args+=(--adapter "$checkpoint") ;;
  *) exit 2 ;;
esac
output="$root/logs/copsd_official_10lang/evaluation/$run_id"
test ! -e "$output"
protocol="${output}.BENCHMARK_PROTOCOL.json"
test ! -e "$protocol"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
python_bin="$root/envs/copsd-official-20261015/bin/python"
"$python_bin" - "$benchmark" "$input" "$expected" "$protocol" "$kind" "$mode" "$checkpoint" "$script_dir/evaluate_transformers.py" <<'PY'
import hashlib, json, sys
from pathlib import Path
benchmark, input_path, expected, protocol, kind, mode, checkpoint, evaluator = sys.argv[1:]
def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()
path = Path(input_path)
if benchmark == "gsm8k":
    rows = json.loads(path.read_text())
    lock = json.loads((path.parent / "MANIFEST.json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != lock["output_sha256"]:
        raise ValueError("Frozen GSM8K input hash differs")
    if any(lock["normalized_exact_train_dev_overlap"].values()):
        raise ValueError("Unresolved training/development overlap")
else:
    rows = path.read_text().splitlines()
if len(rows) != int(expected):
    raise ValueError("Wrong benchmark size")
Path(protocol).parent.mkdir(parents=True, exist_ok=True)
Path(protocol).write_text(json.dumps({
    "benchmark": benchmark, "language": "en", "items": len(rows), "few_shot": 0,
    "input": str(path), "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    "project_evaluator_sha256": digest(evaluator), "mode": mode,
    "checkpoint": checkpoint if checkpoint else None,
    "adapter_sha256": digest(Path(checkpoint) / "adapter_model.safetensors") if checkpoint else None,
    "loader_schema_alias": kind, "max_new_tokens": 2048, "samples": 1,
    "purpose": "project English math retention; benchmark identity is independent of JSON loader alias",
    "selection": "formal test only after development-based checkpoint choice is frozen",
}, ensure_ascii=False, indent=2) + "\n")
PY
exec "$python_bin" "$script_dir/evaluate_transformers.py" \
  --official-eval-source "$root/COPSD-official-20261015/polymath_eval/evaluate_math.py" \
  --base-model "$root/models/Qwen3-1.7B" "${args[@]}" \
  --input "$input" --input-kind "$kind" --language en \
  --samples 1 --max-new-tokens 2048 --seed 42 --output-dir "$output"
