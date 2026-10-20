"""One explicitly started recovery job; never retries failed subprocesses."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from summarize_saved_candidates8192 import pair


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_stage(spec, root, code):
    directory = Path(spec['directory'])
    c, m = read(directory / 'CONFIG.json'), read(directory / 'DONE.json')
    rows = [json.loads(x) for x in (directory / 'ITEMS.jsonl').read_text().splitlines()]
    assert c['language'] == spec['language'] and c['adapter'] == spec['adapter']
    assert c.get('checkpoint_step') == spec.get('step')
    assert c['protocol'] == m['protocol'] == spec['protocol']
    assert c['samples'] == m['n'] == 1 and c['max_new_tokens'] == spec['budget']
    assert c['model'] == str(root / 'models/Qwen3-1.7B')
    assert c['evaluator_sha256'] == digest(code / spec['evaluator'])
    assert c['input_sha256'] == digest(c['input']) and c['official_source_sha256'] == digest(c['source'])
    if spec['budget'] == 8192:
        assert c['max_num_seqs'] == m['actual_max_num_seqs'] == 16
    assert len(rows) == len({r['id'] for r in rows}) == m['items'] == 100 and not m['formal_test']
    assert all(len(r['samples']) == 1 for r in rows)
    assert sum(r['samples'][0]['correct'] for r in rows) == m['single_correct']
    assert sum(len(r['samples'][0]['token_ids']) for r in rows) == m['generated_tokens']
    source = read(c['input'])
    assert len(source) == 100
    for row, item in zip(rows, source):
        assert row['id'] == item['problem_id'] and row['answer'] == item['Answer']
        field = 'problem' if spec['language'] == 'en' else 'problem_' + spec['language']
        assert row['question'] == item[field]
        sample = row['samples'][0]
        assert sample['tokens'] == len(sample['token_ids']) <= spec['budget']
        seed = int.from_bytes(hashlib.sha256(f"42:{spec['language']}:{row['id']}:0".encode()).digest()[:8], 'big') % 2**32
        assert sample['seed'] == seed
    return c, m, rows, 0.


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation', choices=['ja80', 'th', 'te', 'ja20'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args()
    root, code = a.root, Path(__file__).parent
    lock = read(code / 'disk_recovery_runtime_20261020.CONFIG.json')
    for name, expected in lock['code_sha256'].items():
        assert digest(code / name) == expected, name
    assert shutil.disk_usage(root).free >= lock['minimum_free_bytes'], 'Storage reserve required before any recovery'
    op = lock['operations'][a.operation]
    for filename, sha in op.get('frozen_file_sha256', {}).items():
        assert digest(filename) == sha, filename
    for spec in op.get('cached_stages', []):
        audit_stage(spec, root, code)
    for path in op['new_output_directories']:
        assert not Path(path).exists(), path
    env = dict(os.environ)
    env.update(COPSD_VLLM_UUID_COMPAT='1', PYTHONPATH=str(code / 'compat/vllm011_uuid'),
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', VLLM_NO_USAGE_STATS='1', DO_NOT_TRACK='1',
               HF_HOME=str(root / 'cache/copsd_official/huggingface'),
               VLLM_CACHE_ROOT=str(root / 'cache/copsd_official/vllm'), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    if a.prepare_only:
        for command in op['commands']:
            name = Path(command[1]).name
            if name == 'run_ja_resume80_nonefixed_17b.sh':
                # The script checks UUID shape before its CPU-only early return.
                cpu = dict(env, GPU_QUEUE_GPU_COUNT='1', CUDA_VISIBLE_DEVICES='GPU-CPU-PREFLIGHT', COPSD_JA_PREPARE_ONLY='1')
                subprocess.run(command, env=cpu, check=True)
            elif name == 'run_ja_saved20_screen8192.sh':
                subprocess.run(command, env=dict(env, COPSD_SCREEN_PREPARE_ONLY='1'), check=True)
            elif name == 'resume_te_base8192.py':
                subprocess.run(command + ['--prepare-only'], env=env, check=True)
            elif name == 'run_dev8192.sh':
                lang, run_id = command[2:4]
                prepared = [str(root / 'envs/copsd-vllm011-20261018/bin/python'), str(code / 'evaluate_dev8192.py'),
                            '--root', str(root), '--language', lang, '--run-id', run_id, '--prepare-only']
                if len(command) > 4:
                    prepared += ['--adapter', command[4], '--expected-step', command[5]]
                subprocess.run(prepared, env=env, check=True)
            else:
                assert a.operation == 'ja80' and name == 'run_dev_vllm_17b.sh'
                # This future checkpoint100 does not exist before recovery.
                assert command[4] == op['new_final_dev_id']
        print(json.dumps({'operation': a.operation, 'CPU_preflight_passed': True, 'GPU_started': False,
                          'space_reserve_GiB': lock['minimum_free_bytes'] / 2**30}))
        return
    assert env.get('GPU_QUEUE_GPU_COUNT') == '1'
    visible = env.get('CUDA_VISIBLE_DEVICES', '')
    assert visible.startswith('GPU-') and ',' not in visible
    for i, command in enumerate(op['commands']):
        subprocess.run(command, env=env, check=True)
        if a.operation == 'ja80' and i == 0:
            model = Path(op['new_model_directory'])
            state = read(model / 'checkpoint-100/trainer_state.json')
            progress = read(model / 'COST_SEGMENT_DONE.json')
            proof = read(model / 'COLLOCATE_TRAIN_UPDATES_DONE.json')
            assert state['global_step'] == 100 and state['max_steps'] == 100
            assert progress['start_step'] == proof['start_step'] == 80
            assert progress['loss_calls'] == proof['micro_steps'] == 628
            assert proof['global_step'] == 100 and proof['base_unchanged_bitwise'] and len(proof['syncs']) == 21
            assert read(model / 'RUNTIME_SHUTDOWN_DONE.json')['ordered_pool_release_completed']
            for name in ['adapter_model.safetensors', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth']:
                assert (model / 'checkpoint-100' / name).stat().st_size > 0
    stages = {name: audit_stage(spec, root, code) for name, spec in op['result_stages'].items()}
    pairs = {'target': pair(stages['base'], stages['target'])}
    if 'English' in stages:
        pairs['English'] = pair(stages['English_Base'], stages['English'])
    if a.operation == 'te':
        provenance = read(Path(op['result_stages']['base']['directory']) / 'RECOVERY.PROVENANCE.json')
        assert provenance['reused_items'] == 73 and provenance['new_generated_items'] == 27
        assert provenance['cached_records_preserved_exactly']
    marker = Path(op['completion_marker'])
    assert not marker.exists()
    marker.write_text(json.dumps({'paired_scores': pairs, 'scope': 'Stage outputs verified; parent queue exit0 and full cost audit still required.'}, indent=2) + '\n')
    print(json.dumps(pairs))


if __name__ == '__main__':
    main()
