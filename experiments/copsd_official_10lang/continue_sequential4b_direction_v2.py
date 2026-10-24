"""Recover one-shot supervision without restarting existing GPU jobs.

Wait for the tracked stage2 queue, accept its checkpoint, prepare formal queues,
then start stage3 from that checkpoint. Accept stage3 and prepare its formal
queues too. No retries, reboot hooks, cancellation or evaluation submission.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from prepare_sequential_pools import read, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--direction', choices=['forward', 'reverse'], required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    root, code = a.root.resolve(), Path(__file__).parent.resolve()
    assert not a.output.exists()
    status = {'pid': os.getpid(), 'started_at': datetime.datetime.now().isoformat(),
              'direction': a.direction, 'automatic_retries': False,
              'evaluation_queues_started': False, 'events': []}

    def record(phase, **kwargs):
        status.update(phase=phase, updated_at=datetime.datetime.now().isoformat(), **kwargs)
        tmp = a.output.with_name(a.output.name + '.tmp')
        tmp.write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')
        tmp.replace(a.output)
        print(json.dumps({'phase': phase, **kwargs}), flush=True)

    def run(name, *args):
        subprocess.run([sys.executable, str(code / name), *map(str, args)], check=True)

    def wait_queue(queue):
        path = root / 'gpu-queue/queues' / queue / 'state.json'
        while True:
            try:
                state = json.loads(path.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                # systemctl start returns before the monitor's first JSON write.
                # Query its actual systemd PID rather than treating this
                # observation race as a terminal queue failure or resubmitting.
                info = subprocess.check_output(['systemctl', '--user', 'show',
                    'gpu-queue@' + queue + '.service',
                    '--property=MainPID,ActiveState'], text=True)
                service = dict(line.split('=', 1) for line in info.splitlines() if '=' in line)
                assert service.get('ActiveState') in ['active', 'activating'], service
                pid = int(service.get('MainPID', '0'))
                if pid:
                    assert Path('/proc', str(pid)).exists(), service
                time.sleep(1)
                continue
            phase = state['phase']
            if phase == 'completed':
                assert state['job']['exit_code'] == 0, (queue, state['job'])
                return
            if phase in ['failed', 'error', 'stopped', 'cancelled', 'canceled']:
                raise RuntimeError('Tracked queue terminated: ' + queue + ' ' + phase)
            # A live process is required to classify this as a verified wait.
            monitor = state.get('monitor_pid')
            job = state.get('job') or {}
            candidates = [pid for pid in [monitor, job.get('pid')] if pid]
            assert candidates and any((Path('/proc') / str(pid)).exists() for pid in candidates), (
                'No tracked queue or parent process remains', queue, phase)
            time.sleep(30)

    logs = root / 'logs/copsd_official_10lang'
    selection = code / 'sequential4b_selection_frozen_20261024_v1.LOCAL.json'
    prefix = 'f' if a.direction == 'forward' else 'r'
    try:
        record('waiting_stage2')
        for stage, language in [(2, 'te'), (3, 'zh' if a.direction == 'forward' else 'sw')]:
            queue = f'copsd-seq4b-{prefix}{stage}-{language}-v1-training'
            freeze = code / f'sequential4b_{a.direction}_s{stage}_{language}_frozen_20261024_v1.LOCAL.json'
            release = logs / f'selected4b_{a.direction}_s{stage}_RELEASE_20261024.PRIVATE.json'
            exposures = logs / f'selected4b_{a.direction}_s{stage}_EXPOSURES_20261024.PRIVATE.json'
            if stage == 3:
                parent = json.loads(previous_release.read_text())['checkpoint_relative']
                run_id = f'{a.direction}_s3_{language}_4b_pool{2 if language == "zh" else 0}_from_s2te20_seed42_20261024_v1'
                if not freeze.exists():
                    run('prepare_sequential4b_stage.py', '--root', root, '--selection', selection,
                        '--direction', a.direction, '--stage', 3, '--run-id', run_id,
                        '--container-relative', 'models/checkpoint/copsd_selected4b_sequential_v1',
                        '--parent-relative', parent, '--freeze', freeze)
                else:
                    f = read(freeze)
                    assert f['direction'] == a.direction and f['stage_index'] == 3
                    assert f['parent_checkpoint_relative'] == parent
                    assert f['parent_adapter_sha256'] == sha(root / parent / 'adapter_model.safetensors')
                    assert f['selection_record_sha256'] == sha(selection)
                queue_map = logs / f'sequential4b_{a.direction}_s3_queues_20261024.PRIVATE.json'
                if not queue_map.exists():
                    run('prepare_sequential4b_queues.py', '--root', root, '--freeze', freeze,
                        '--training', '--prefix', f'copsd-seq4b-{prefix}3-{language}-v1',
                        '--output', queue_map)
                assert json.loads(queue_map.read_text()) == {'training': queue}
                config = read(root / 'gpu-queue/queues' / queue / 'config.json')
                assert config['max_user_gpus'] == 4 and config['target_gpus'] == 2
                assert config['job']['args'] == [str(freeze)]
                state_path = root / 'gpu-queue/queues' / queue / 'state.json'
                if state_path.exists():
                    existing = read(state_path)
                    assert existing['phase'] in ['running', 'waiting', 'holding', 'completed'], existing
                    record('adopting_existing_stage3', queue=queue)
                else:
                    subprocess.run([sys.executable, str(root / 'gpu-queue/gpu_queue.py'),
                                    '--queue', queue, 'start'], check=True)
                    record('stage3_submitted', queue=queue)
            wait_queue(queue)
            record('accepting_stage' + str(stage), queue=queue)
            if not release.exists():
                run('collect_sequential4b_stage.py', '--root', root, '--freeze', freeze,
                    '--queue', queue, '--output', release)
            else:
                from collect_sequential4b_stage import collect
                assert collect(root, freeze, queue, None) == read(release)
            if not exposures.exists():
                run('recover_sequential4b_exposure_ids.py', '--root', root,
                    '--release', release, '--output', exposures)
            actual = json.loads(exposures.read_text())
            assert actual['freeze_sha256'] == sha(freeze)
            assert actual['release_sha256'] == sha(release)
            assert actual['recorded_prompt_sha256'] == sha(root / read(freeze)['checkpoint_relative'] / 'EXPOSURE_PROMPTS.PRIVATE.json')
            reference = json.loads((logs / f'selected4b_matched_{language}20_EXPOSURES_20261024.PRIVATE.json').read_text())
            assert actual['actual_exposure_original_ids_in_order'] == reference['actual_exposure_original_ids_in_order']
            status['events'].append({'stage': stage, 'language': language,
                                     'release': str(release), 'matched_order_verified': True})
            eval_freeze = code / f'sequential4b_{a.direction}_s{stage}_formal_frozen_20261024_v1.LOCAL.json'
            if not eval_freeze.exists():
                run('prepare_sequential4b_evaluation.py', '--root', root, '--selection', selection,
                    '--model-id', f'{a.direction}_s{stage}', '--release', release,
                    '--output-relative', 'logs/copsd_official_10lang/selected4b_formal_20261024_v1',
                    '--freeze', eval_freeze)
            assert read(eval_freeze)['model_provenance']['release_sha256'] == sha(release)
            eval_map = logs / f'sequential4b_{a.direction}_s{stage}_formal_queues_20261024.PRIVATE.json'
            if not eval_map.exists():
                run('prepare_sequential4b_queues.py', '--root', root, '--freeze', eval_freeze,
                    '--evaluation', '--prefix', f'copsd-seq4b-{prefix}{stage}-eval-v1',
                    '--output', eval_map)
            assert set(read(eval_map)) == set(read(eval_freeze)['cases'])
            previous_release = release
        record('completed_training_and_prepared_formal_queues')
    except BaseException as error:
        record('failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
