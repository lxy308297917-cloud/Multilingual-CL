"""Manually started one-shot wave of missing primary ten-language formal tests.

Waits for the selected-language stage wave, then submits already validated queues.
Does not train, alter protocols, retry failures, or install persistent scheduling.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    root = a.root.resolve()
    logs = root / 'logs/copsd_official_10lang'
    assert not a.output.exists()
    status = dict(pid=os.getpid(), started_at=datetime.datetime.now().isoformat(),
                  automatic_retries=False, max_single_GPU_monitors=4, submitted=[])

    def read(path):
        return json.loads(path.read_text())

    def record(phase, **kwargs):
        status.update(phase=phase, updated_at=datetime.datetime.now().isoformat(), **kwargs)
        tmp = a.output.with_name(a.output.name + '.tmp')
        tmp.write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')
        tmp.replace(a.output)
        print(json.dumps({'phase': phase, **kwargs}), flush=True)

    def queue_status(queue, fresh=False):
        path = root / 'gpu-queue/queues' / queue / 'state.json'
        for _ in range(60 if fresh else 1):
            if path.exists():
                s = read(path)
                if s['phase'] == 'completed':
                    assert s['job']['exit_code'] == 0, queue
                    return 'completed'
                assert s['phase'] not in ['failed', 'error', 'stopped', 'cancelled', 'canceled'], (queue, s)
                pids = [s.get('monitor_pid'), (s.get('job') or {}).get('pid')]
                if any(pid and Path('/proc', str(pid)).exists() for pid in pids):
                    return 'active'
            if fresh:
                time.sleep(1)
        raise RuntimeError('Submitted queue has no tracked live process: ' + queue)

    try:
        maps = [logs / 'primary4b_Base_additional14_formal_queues_20261024.PRIVATE.json']
        maps += [logs / f'primary4b_{lang}20_formal_queues_20261024.PRIVATE.json'
                 for lang in ['SW', 'TE', 'TH', 'BN', 'ZH', 'JA', 'RU', 'DE', 'FR', 'ES']]
        queues = [queue for path in maps for queue in read(path).values()]
        assert len(queues) == len(set(queues)) == 42
        for queue in queues:
            config = read(root / 'gpu-queue/queues' / queue / 'config.json')
            assert config['max_user_gpus'] == 4 and config['target_gpus'] == 1
            assert Path(config['job']['script']).name == 'run_formal4b_case.sh'
            freeze, case = config['job']['args']
            f = read(Path(freeze))
            assert case in f['cases'] and not f['engineering_preflight_only']
            assert f['max_new_tokens'] == 8192 and f['max_num_seqs'] == 16
        record('waiting_selected_formal_wave', configured_cases=len(queues))
        while True:
            preceding = read(logs / 'sequential4b_formal_wave_20261024.PRIVATE.json')
            if preceding['phase'] == 'completed_evaluation_and_TensorBoard':
                break
            assert preceding['phase'] != 'failed', preceding
            assert Path('/proc', str(preceding['pid'])).exists(), preceding
            time.sleep(30)
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
                # Recheck both real filesystems at submission time; model storage
                # may be a symlink to data1 and must not be inferred from data0.
                checkpoints = (root / 'models/checkpoint/copsd_selected4b_sequential_v1').resolve(strict=True)
                assert shutil.disk_usage(root).free > 2 * 2**30
                assert shutil.disk_usage(checkpoints).free > 2 * 2**30
                queue = pending.pop(0)
                subprocess.run([sys.executable, str(root / 'gpu-queue/gpu_queue.py'),
                                '--queue', queue, 'start'], check=True)
                status['submitted'].append(queue)
                if queue_status(queue, fresh=True) == 'completed':
                    completed.append(queue)
                else:
                    active.append(queue)
            record('formal_wave', active=list(active), pending=len(pending), completed=len(completed))
            if pending or active:
                time.sleep(30)
        record('completed_formal_queue_jobs',
               remaining='Full paired audits, language analysis, unique cost aggregation and reporting remain required; queue completion alone is not effect evidence.')
    except BaseException as error:
        record('failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
