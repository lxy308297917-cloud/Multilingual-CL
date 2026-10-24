"""Prepare unique manual queues for one frozen 4B stage or seven-case plan.

Validates science/data first; never starts monitors, holds GPUs or submits jobs.
The caller must inspect account-wide GPU ownership before manually starting.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from prepare_sequential_pools import read, sha


def prepare(root, freeze, kind, prefix, output):
    root, freeze, output = Path(root), Path(freeze), Path(output)
    code = Path(__file__).parent
    assert kind in ['training', 'evaluation']
    assert re.fullmatch(r'[a-z0-9][a-z0-9-]{0,90}', prefix)
    assert not output.exists()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    if kind == 'training':
        from sequential4b_context import prepare as preflight
        preflight(root, freeze)
        units = {'training': [str(freeze.resolve())]}
        script = code / 'run_sequential4b_stage.sh'
        count = 2
        template = root / 'gpu-queue/queues/copsd-sequential4b-engineering2-v1/config.json'
    else:
        from formal4b_contract import prepare as preflight
        frozen, cases = preflight(root, freeze)
        assert frozen['engineering_preflight_only'] is False
        # Reuse the stronger production selection gate before creating queues.
        from sequential4b_selection import selection, local
        selection(root, local(root, frozen['selection_record_relative']), frozen['selection_record_sha256'])
        cached = frozen.get('cached_cases', {})
        assert set(cached) <= set(cases)
        if cached:
            assert frozen['model_id'] == 'Base'
            assert sha(local(root, frozen['Base_cache_audit_relative'])) == frozen['Base_cache_audit_sha256']
            for case in cached.values():
                folder = local(root, case['output_relative'])
                for name, digest in case['file_sha256'].items():
                    assert sha(folder / name) == digest
        units = {name: [str(freeze.resolve()), name] for name in cases if name not in cached}
        assert all(not cases[name]['output'].exists() for name in units)
        script = code / 'run_formal4b_case.sh'
        count = 1
        template = root / 'gpu-queue/queues/copsd-es4b-eval200-v1/config.json'
    names = {name: prefix + '-' + name.replace('_', '-') for name in units}
    assert len(names) == len(set(names.values()))
    queue_root = root / 'gpu-queue/queues'
    assert all(not (queue_root / name).exists() for name in names.values())
    template_config = read(template)
    assert template_config['mode'] == 'job'
    for unit, name in names.items():
        config = copy.deepcopy(template_config)
        config.update(interval_minutes=5., target_gpus=count, max_user_gpus=4, mode='job')
        config['job'].update(script=str(script), cwd=str(code.parent.parent),
                             args=units[unit], log=str(root / 'logs/copsd_official_10lang' / (name + '.run.log')))
        folder = queue_root / name
        folder.mkdir(exist_ok=False)
        with (folder / 'config.json').open('x') as stream:
            json.dump(config, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        subprocess.run([sys.executable, str(root / 'gpu-queue/gpu_queue.py'),
                        '--queue', name, 'validate'], env=env, check=True)
    with output.open('x') as stream:
        json.dump(names, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--freeze', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--training', action='store_true')
    mode.add_argument('--evaluation', action='store_true')
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    queues = prepare(args.root, args.freeze, 'training' if args.training else 'evaluation', args.prefix, args.output)
    print(json.dumps({'queues_prepared': queues, 'GPU_acquired': False, 'monitors_started': False}))


if __name__ == '__main__':
    main()
