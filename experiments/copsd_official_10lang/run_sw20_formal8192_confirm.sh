#!/usr/bin/env bash
set -euo pipefail
root="${COPSD_PROJECT_ROOT:-$HOME}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
model="${1:?base,seed42,seed43}"
exec "$root/envs/copsd-vllm011-20261018/bin/python" - "$root" "$script_dir" "$model" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

root, code = map(Path, sys.argv[1:3])
model = sys.argv[3]
assert model in ['base', 'seed42', 'seed43']
manifest_path = code / 'sw20_formal8192_frozen_20261021.PUBLIC.json'
manifest = json.loads(manifest_path.read_text())
prepare_only = os.environ.get('COPSD_SW_FORMAL_PREPARE_ONLY') == '1'
stages = [stage for stage in manifest['stages'] if stage['model'] == model]
assert len(stages) == 3 and sum(stage['items'] for stage in stages) == 2069
for name, sha in manifest['code_sha256'].items():
    assert hashlib.sha256((code / name).read_bytes()).hexdigest() == sha, name
assert shutil.disk_usage(root).free >= manifest['minimum_free_bytes']
if not prepare_only:
    assert os.environ.get('GPU_QUEUE_GPU_COUNT') == '1'
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    assert visible.startswith('GPU-') and ',' not in visible
for stage in stages:
    target = root / 'logs/copsd_official_10lang/evaluation_vllm8192' / stage['run_id']
    assert not target.exists(), target
    adapter = stage['adapter_relative']
    if adapter:
        cp = root / adapter
        assert json.loads((cp / 'trainer_state.json').read_text())['global_step'] == 20
        assert hashlib.sha256((cp / 'adapter_model.safetensors').read_bytes()).hexdigest() == stage['adapter_sha256']
if prepare_only:
    print(json.dumps({'CPU_identity_preflight': True, 'GPU_started': False,
                      'model': model, 'stages': len(stages),
                      'planned_responses': sum(stage['items'] for stage in stages)}))
    sys.exit(0)
for stage in stages:
    assert shutil.disk_usage(root).free >= manifest['minimum_free_bytes']
    command = ['bash', str(code / 'run_formal8192.sh'), stage['benchmark'], stage['language'], stage['run_id']]
    if stage['adapter_relative']:
        command += [str(root / stage['adapter_relative']), '20']
    print('FROZEN_FORMAL_STAGE ' + json.dumps(stage), flush=True)
    subprocess.run(command, check=True)
    output = root / 'logs/copsd_official_10lang/evaluation_vllm8192' / stage['run_id']
    config = json.loads((output / 'CONFIG.json').read_text())
    done = json.loads((output / 'DONE.json').read_text())
    assert done['formal_full_test'] and done['items'] == stage['items']
    assert done['protocol'] == config['protocol'] == manifest['protocol']
    assert config['adapter_sha256'] == stage.get('adapter_sha256')
    assert config['checkpoint_step'] == stage['step']
print('SW20_FORMAL_PIPELINE_DONE', flush=True)
PY
