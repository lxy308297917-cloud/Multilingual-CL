"""Audit one complete frozen SW20 formal pipeline after its parent exits0."""
import argparse
import datetime
import json
from pathlib import Path

from sequential_evaluation import audit_benchmark, digest, read, write_new


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--model', choices=['base', 'seed42', 'seed43'], required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    code = Path(__file__).parent
    frozen = read(code / 'sw20_formal8192_frozen_20261021.PUBLIC.json')
    for name, sha in frozen['code_sha256'].items():
        assert digest(code / name) == sha
    queue = {'base': 'copsd-sw20-formal8192-confirm-v1',
        'seed42': 'copsd-sw20-formal8192-seed42-v1', 'seed43': 'copsd-sw20-formal8192-seed43-v1'}[a.model]
    state = read(a.root / 'gpu-queue/queues' / queue / 'state.json')
    assert state['phase'] == 'completed' and state['job']['exit_code'] == 0
    job = state['job']
    assert job['config']['job']['args'] == [a.model]
    assert Path(job['script']).resolve() == code / 'run_sw20_formal8192_confirm.sh'
    assert digest(job['script']) == job['script_sha256']
    stages = [s for s in frozen['stages'] if s['model'] == a.model]
    assert {s['benchmark'] for s in stages} == {'mgsm_rev2', 'polymath', 'gsm8k'} and len(stages) == 3
    completed = []
    for s in stages:
        adapter = a.root / s['adapter_relative'] if s['adapter_relative'] else None
        if adapter:
            assert digest(adapter / 'adapter_model.safetensors') == s['adapter_sha256']
        case = {'benchmark': s['benchmark'], 'language': s['language'], 'run_id': s['run_id'],
            'protocol': frozen['protocol']}
        folder = a.root / 'logs/copsd_official_10lang/evaluation_vllm8192' / s['run_id']
        case['input_sha256'] = read(folder / 'CONFIG.json')['input_sha256']
        result = audit_benchmark(a.root, code, case, adapter)
        assert result['items'] == s['items']
        completed.append(result)
    wall = (datetime.datetime.fromisoformat(job['finished_at']) - datetime.datetime.fromisoformat(job['started_at'])).total_seconds()
    write_new(a.output, {'observed_at': datetime.datetime.now().isoformat(), 'model': a.model,
        'protocol': frozen['protocol'], 'queue': queue, 'parent_completed_exit0': True,
        'results': completed, 'cost': {'responses': sum(s['items'] for s in completed),
            'generated_tokens': sum(s['metrics']['generated_tokens'] for s in completed),
            'generation_seconds': sum(s['metrics']['generation_seconds'] for s in completed),
            'allocated_GPU_hours_parent_counted_once': wall / 3600},
        'scope': 'One complete SW MGSM/PolyMath plus English pipeline, project n1/8192, not official pass12. Other model pipelines may still be running. This cost overlaps benchmark-stage reports; never sum both. No training or generation performed by this analysis.'})
    print(json.dumps({r['benchmark']: {'correct': r['correct'], 'items': r['items']} for r in completed}))


if __name__ == '__main__':
    main()
