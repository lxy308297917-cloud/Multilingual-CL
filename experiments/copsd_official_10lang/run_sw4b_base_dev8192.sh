#!/usr/bin/env bash
set -euo pipefail
root="${COPSD_PROJECT_ROOT:-$HOME}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="$root/envs/copsd-vllm011-20261018/bin/python"
for lang in sw en; do
 run_id="$("$python_bin" - "$script_dir" "$lang" <<'READ'
import json,sys
from pathlib import Path
print(json.loads((Path(sys.argv[1])/'sw4b_pilot20_frozen_20261022.PUBLIC.json').read_text())['base_evaluation_run_ids'][sys.argv[2]])
READ
 )"
 bash "$script_dir/run_sw4b_dev8192.sh" "$lang" "$run_id"
done
