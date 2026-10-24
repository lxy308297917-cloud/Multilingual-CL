"""Observe one manually started engineering queue, then collect once on CPU."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

from single4b_compact_entry import read, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--freeze', type=Path, required=True)
    p.add_argument('--monitor', type=int, required=True)
    a = p.parse_args()
    code = Path(__file__).parent
    queue = a.root / 'gpu-queue/queues/copsd-sequential4b-engineering2-v1'
    pins = {a.freeze: sha(a.freeze), queue / 'config.json': sha(queue / 'config.json'),
            Path(__file__): sha(Path(__file__)),
            code / 'collect_sequential4b_engineering.py': sha(code / 'collect_sequential4b_engineering.py')}
    status = {'observer_PID': os.getpid(), 'queue_monitor_PID': a.monitor,
              'automatic_retries': False, 'jobs_cancelled_or_restarted': False,
              'started_at': datetime.datetime.now().isoformat(),
              'bindings': {str(path): digest for path, digest in pins.items()}}
    out = a.root / 'logs/copsd_official_10lang/sequential4b_engineering2_20261024.series.PRIVATE.json'
    with out.open('x') as stream:
        stream.write(json.dumps(status, indent=2) + '\n')

    def save(phase):
        status.update(phase=phase, updated_at=datetime.datetime.now().isoformat())
        tmp = out.with_suffix('.tmp')
        tmp.write_text(json.dumps(status, indent=2) + '\n')
        tmp.replace(out)
        print(phase, flush=True)

    try:
        save('waiting_for_engineering_queue')
        deadline = time.monotonic() + 8 * 3600
        while True:
            for path, digest in pins.items():
                assert sha(path) == digest, str(path)
            path = queue / 'state.json'
            if not path.exists():
                os.kill(a.monitor, 0)
            else:
                state = read(path)
                if state['phase'] == 'completed' and state['job']['exit_code'] == 0:
                    break
                if state['phase'] in ['failed', 'cancelled', 'stopped']:
                    raise RuntimeError(state['phase'])
                assert state['monitor_pid'] == a.monitor
                try:
                    os.kill(a.monitor, 0)
                except ProcessLookupError:
                    time.sleep(2)
                    state = read(path)
                    if state['phase'] == 'completed' and state['job']['exit_code'] == 0:
                        break
                    raise RuntimeError('Original monitor missing; inspect actual job, no restart')
            if time.monotonic() > deadline:
                raise TimeoutError('Observation deadline only; original experiment unchanged')
            time.sleep(30)
        save('collecting_engineering_evidence')
        subprocess.run([str(a.root / 'envs/copsd-vllm011-20261018/bin/python'),
                        str(code / 'collect_sequential4b_engineering.py'),
                        '--root', str(a.root), '--freeze', str(a.freeze),
                        '--output', str(code / 'SEQUENTIAL4B_ENGINEERING2_COMPLETED_SCIENCE_20261024.PUBLIC.json')],
                       env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, check=True)
        save('done')
    except BaseException as error:
        status['error'] = repr(error)
        save('observer_or_postprocessing_error_inspect_original_job')
        raise


if __name__ == '__main__':
    main()
