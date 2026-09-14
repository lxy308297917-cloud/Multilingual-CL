#!/usr/bin/env python3
"""Write a separate machine-local config; retain the frozen evaluation settings."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root', required=True, type=Path)
    ap.add_argument('--base-model', required=True, type=Path)
    ap.add_argument('--training-python', required=True, type=Path)
    ap.add_argument('--evaluation-python', required=True, type=Path)
    ap.add_argument('--output-root', required=True, type=Path)
    ap.add_argument('--config-out', required=True, type=Path)
    ap.add_argument('--gpus', default='0', help='Local GPU indices, e.g. 0 or 0,1')
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    data = a.data_root.resolve()
    receipt = data / 'DATA_VERIFIED.json'
    manifest = json.loads(receipt.read_text())
    public_spec = root / 'configs/ig_baseline_data_v1.json'
    if manifest['spec_sha256'] != hashlib.sha256(public_spec.read_bytes()).hexdigest():
        raise ValueError('Data was not verified against this checkout specification')
    for path, expected in manifest['evaluation_inputs_inherited'].items():
        p = Path(path)
        if not p.is_relative_to(data) or hashlib.sha256(p.read_bytes()).hexdigest() != expected:
            raise ValueError('Invalid or changed rebuilt file: ' + path)
    for p in [a.training_python, a.evaluation_python, a.base_model / 'config.json']:
        if not p.is_file():
            raise FileNotFoundError(p)
    if a.output_root.exists() and any(a.output_root.iterdir()):
        raise ValueError('Use an empty, separate evaluation output directory')
    if a.config_out.exists():
        raise FileExistsError(a.config_out)
    c = json.loads((root / 'configs/ig_baseline_eval_v3.json').read_text())
    c.update(base_model=str(a.base_model.resolve()),
             training_python=str(a.training_python.resolve()),
             evaluation_python=str(a.evaluation_python.resolve()),
             output_root=str(a.output_root.resolve()),
             evaluation_data_root=str(data), evaluation_data_manifest=str(receipt),
             lighteval_source=str(root.parent / 'lighteval_latest/src'),
             hf_cache=str(data / 'runtime_cache'), ppl_data_root=str(data / 'sum_ssu'),
             ifeval_data=str(data / 'ifeval/ifeval_input_data.jsonl'),
             gsm8k_data=str(data / 'gsm8k/gsm8k-test.arrow'),
             evaluation_gpus=[int(g) for g in a.gpus.split(',')])
    c['gpu'] = c['evaluation_gpus'][0]
    # The local config/code/environment identity creates a new protocol lock.
    a.config_out.parent.mkdir(parents=True, exist_ok=True)
    a.config_out.write_text(json.dumps(c, ensure_ascii=False, indent=2) + '\n')
    print(a.config_out.resolve())
    print('No evaluation started. Use this config with baseline_eval.sh.')


if __name__ == '__main__':
    main()
