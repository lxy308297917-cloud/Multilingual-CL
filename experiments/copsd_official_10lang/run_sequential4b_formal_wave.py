"""One manually launched formal evaluation wave after both stage3 jobs finish.

Uses the existing frozen seven-case plans and queue configurations. Maximum four
single-GPU queue monitors; account-wide admission remains enforced by gpu-queue.
No retries, cancelled-task revival, training changes or boot scheduling.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    root, code = a.root.resolve(), Path(__file__).parent.resolve()
    logs = root / 'logs/copsd_official_10lang'
    assert not a.output.exists()
    state = dict(pid=os.getpid(), started_at=datetime.datetime.now().isoformat(),
                 max_single_GPU_monitors=4, automatic_retries=False, submitted=[])

    def record(phase, **kwargs):
        state.update(phase=phase, updated_at=datetime.datetime.now().isoformat(), **kwargs)
        tmp = a.output.with_name(a.output.name + '.tmp')
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
        tmp.replace(a.output)
        print(json.dumps({'phase': phase, **kwargs}), flush=True)

    def read(path):
        return json.loads(path.read_text())

    def run(name, *args):
        subprocess.run([sys.executable, str(code / name), *map(str, args)], check=True)

    def queue_status(queue, just_started=False):
        path = root / 'gpu-queue/queues' / queue / 'state.json'
        for _ in range(60 if just_started else 1):
            if path.exists():
                s = read(path)
                if s['phase'] == 'completed':
                    assert s['job']['exit_code'] == 0, queue
                    return 'completed'
                assert s['phase'] not in ['failed', 'error', 'stopped', 'cancelled', 'canceled'], (queue, s)
                candidates = [s.get('monitor_pid'), (s.get('job') or {}).get('pid')]
                if any(pid and Path('/proc', str(pid)).exists() for pid in candidates):
                    return 'active'
            if just_started:
                time.sleep(1)
        raise RuntimeError('No live tracked monitor/parent for submitted queue: ' + queue)

    try:
        record('waiting_both_stage3_releases')
        while True:
            done = []
            for direction in ['forward', 'reverse']:
                s = read(logs / f'sequential4b_{direction}_continuation_20261024.PRIVATE.json')
                if s['phase'] == 'completed_training_and_prepared_formal_queues':
                    done.append(direction)
                else:
                    assert s['phase'] != 'failed', s
                    assert Path('/proc', str(s['pid'])).exists(), s
            if len(done) == 2:
                break
            time.sleep(30)
        models = ['Base', 'SWmatched', 'TEmatched', 'ZHmatched',
                  'forward_s2', 'reverse_s2', 'forward_s3', 'reverse_s3']
        plans = {}
        queues = []
        for model in models:
            freeze = code / f'sequential4b_{model}_formal_frozen_20261024_v1.LOCAL.json'
            mapping = logs / f'sequential4b_{model}_formal_queues_20261024.PRIVATE.json'
            f, q = read(freeze), read(mapping)
            assert set(q) == set(f['cases']) - set(f.get('cached_cases', {}))
            plans[model] = (freeze, mapping)
            for name, queue in q.items():
                config = read(root / 'gpu-queue/queues' / queue / 'config.json')
                assert config['max_user_gpus'] == 4 and config['target_gpus'] == 1
                assert config['job']['args'] == [str(freeze), name]
                queues.append(queue)
        assert len(queues) == len(set(queues))
        pending, active, completed = [], [], []
        for queue in queues:
            path = root / 'gpu-queue/queues' / queue / 'state.json'
            if not path.exists():
                pending.append(queue)
            elif queue_status(queue) == 'completed':
                completed.append(queue)
            else:
                active.append(queue)
        assert len(active) <= 4
        while pending or active:
            for queue in list(active):
                if queue_status(queue) == 'completed':
                    active.remove(queue)
                    completed.append(queue)
            while pending and len(active) < 4:
                queue = pending.pop(0)
                subprocess.run([sys.executable, str(root / 'gpu-queue/gpu_queue.py'),
                                '--queue', queue, 'start'], check=True)
                state['submitted'].append(queue)
                if queue_status(queue, just_started=True) == 'completed':
                    completed.append(queue)
                else:
                    active.append(queue)
            record('formal_wave', active=list(active), pending=len(pending), completed=len(completed))
            if pending or active:
                time.sleep(30)
        record('collecting_complete_seven_case_markers')
        for model, (freeze, mapping) in plans.items():
            marker = logs / f'selected4b_{model}_FORMAL_COMPLETED_20261024.PRIVATE.json'
            run('collect_sequential4b_evaluation.py', '--root', root, '--freeze', freeze,
                '--queue-map', mapping, '--output', marker)
            run('record_sequential4b_evaluation_tensorboard.py', '--root', root,
                '--evaluation', marker,
                '--directory', logs / 'tensorboard/selected4b_formal_complete' / model,
                '--receipt', logs / f'selected4b_{model}_FORMAL_TB_20261024.PRIVATE.json')
        record('completed_evaluation_and_TensorBoard',
               remaining='CPU language analysis, matched comparisons, forgetting matrices, ten-primary formal tests and reporting remain required.')
    except BaseException as error:
        record('failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
