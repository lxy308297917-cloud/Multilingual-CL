#!/usr/bin/env bash
set -euo pipefail
METHOD=${1:?method required}
shift
case "$METHOD" in
  element_protected|element_unprotected|edge_projected_rows|edge_rows_no_projection|fft_fp32_matched) ;;
  *) echo "Unknown frozen method: $METHOD" >&2; exit 2 ;;
esac
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT=$(cd -- "$ROOT/../.." && pwd -P)
CONFIG="$ROOT/configs/$METHOD.json"
GPU=${GPU:-0}
MODE=both
RESUME=0
CHECK=0
while (($#)); do
  case "$1" in
    --gpu) GPU=${2:?GPU number required}; shift 2 ;;
    --resume) RESUME=1; shift ;;
    --train-only) MODE=train; shift ;;
    --eval-only) MODE=eval; shift ;;
    --check) CHECK=1; shift ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ "$GPU" =~ ^[0-9]+$ ]] || { echo "GPU must be a non-negative integer" >&2; exit 2; }
[[ -f "$CONFIG" && -f "$ROOT/TRAINING_RELEASE.json" ]] || { echo "Missing frozen config or release" >&2; exit 1; }
PYTHON=$(python3 - "$CONFIG" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['training_python'])
PY
)
[[ -x "$PYTHON" ]] || { echo "Missing frozen Python: $PYTHON" >&2; exit 1; }
[[ -f "$PROJECT/ssu-main/scripts/train_cl.py" && -f "$PROJECT/ssu-main/scripts/evaluate_cl.py" ]] || {
  echo "Place ssu-main and experiments side by side under the project root" >&2; exit 1;
}
"$PYTHON" - "$CONFIG" "$ROOT" "$METHOD" <<'PY'
import json,pathlib,sys
cfg=json.load(open(sys.argv[1])); root=pathlib.Path(sys.argv[2]); method=sys.argv[3]
assert cfg['method']==method and cfg['protocol_id']=='sw_d2_sparse_projection_seed42_v1'
assert cfg['seed']==42 and cfg['input_token_budget']==15360000 and cfg['max_length']==2048
release=json.load(open(root/'TRAINING_RELEASE.json'))
assert release['status']=='released' and method in release['methods']
required=[cfg['base_model'],cfg['manifest'],cfg['calibration_file'],cfg['basis_file'],cfg['recipe_selection']]
missing=[p for p in required if not pathlib.Path(p).exists()]
if missing: raise SystemExit('Missing frozen inputs: '+', '.join(missing))
print(f'Preflight OK: {method}; exact identities are verified again by train_cl.py')
PY
if ((CHECK)); then exit 0; fi
if [[ "$MODE" != eval ]]; then
  cmd=("$PYTHON" "$PROJECT/ssu-main/scripts/train_cl.py" --config "$CONFIG" --gpu "$GPU")
  if ((RESUME)); then cmd+=(--resume); fi
  "${cmd[@]}"
fi
if [[ "$MODE" != train ]]; then
  "$PYTHON" "$PROJECT/ssu-main/scripts/evaluate_cl.py" --config "$CONFIG" --gpu "$GPU"
fi
