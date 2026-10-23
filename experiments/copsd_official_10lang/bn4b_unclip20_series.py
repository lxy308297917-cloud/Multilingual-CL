"""Manually launched one-shot BN clipping-control completion/evaluation chain.

Observe the already started training queue; after successful checkpoint release,
start exactly one preconfigured evaluation queue. No retry, boot hook, or task
cancellation. Observation timeouts leave the actual jobs unchanged.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*2**20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    root, code = args.root, Path(__file__).parent
    freeze_path = code/'bn4b_unclip20_series_frozen_20261023.PRIVATE.json'
    frozen = read(freeze_path)
    def preflight():
        for name,digest in frozen['code_sha256'].items():
            assert sha(code/name) == digest, name
        for name,digest in frozen['queue_config_sha256'].items():
            assert sha(root/'gpu-queue/queues'/name/'config.json') == digest, name
        train = read(code/'bn4b_unclip20_frozen_20261023.PUBLIC.json')
        evaluation = read(code/'bn4b_unclip20_eval_frozen_20261023.PUBLIC.json')
        assert evaluation['training_freeze_sha256'] == sha(code/'bn4b_unclip20_frozen_20261023.PUBLIC.json')
        for name,digest in train['code_sha256'].items():
            assert sha(code/name) == digest, name
        for name,digest in evaluation['runtime_code_sha256'].items():
            assert sha(code/name) == digest, name
        for queue in frozen['queue_config_sha256']:
            config = read(root/'gpu-queue/queues'/queue/'config.json')
            assert config['max_user_gpus'] == 4 and config['mode'] == 'job'
        return train,evaluation
    train,evaluation = preflight()
    if args.prepare_only:
        print('One-shot series source/queue bindings passed; no waiting or GPU launch')
        return
    environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': ''}
    python = root/'envs/copsd-vllm011-20261018/bin/python'
    queue_tool = root/'gpu-queue/gpu_queue.py'
    status_path = root/'logs/copsd_official_10lang/bn4b_unclip20_series_20261023.PRIVATE.json'
    assert not status_path.exists()
    status = {'started_at': datetime.datetime.now().isoformat(), 'observer_PID': os.getpid(),
              'freeze_sha256': sha(freeze_path), 'automatic_retries': False,
              'evaluation_start_calls': 0, 'phase': 'waiting_for_existing_training'}
    def save(phase):
        status['phase'] = phase
        status['updated_at'] = datetime.datetime.now().isoformat()
        status_path.write_text(json.dumps(status,indent=2)+'\n')
    def wait_queue(queue, expected_monitor=None):
        deadline = time.monotonic()+8*3600
        state_deadline = time.monotonic()+30
        path = root/'gpu-queue/queues'/queue/'state.json'
        while True:
            if not path.exists():
                assert expected_monitor and time.monotonic() < state_deadline, 'Queue did not publish initial state'
                os.kill(expected_monitor,0)
                time.sleep(1)
                continue
            state = read(path)
            if state['phase'] == 'completed' and state['job']['exit_code'] == 0:
                return state['job']
            if state['phase'] in ['failed','stopped','cancelled']:
                raise RuntimeError((queue,state['phase'],'No retry or cancellation performed'))
            if expected_monitor is not None:
                assert state['monitor_pid'] == expected_monitor, 'Existing queue monitor identity changed'
            pid = state.get('monitor_pid')
            assert pid, 'No authoritative queue monitor handle'
            try:
                os.kill(pid,0)
            except ProcessLookupError:
                time.sleep(2)
                final = read(path)
                if final['phase'] == 'completed' and final['job']['exit_code'] == 0:
                    return final['job']
                raise RuntimeError((queue,'Monitor handle missing; inspect original job. No retry performed'))
            if time.monotonic() > deadline:
                raise TimeoutError('Observer deadline only; actual queue/job unchanged')
            time.sleep(30)
    def run(command, **kwargs):
        subprocess.run([str(x) for x in command],env=environment,check=True,**kwargs)
    save('waiting_for_existing_training')
    try:
        job = wait_queue('copsd-bn4b-unclip20-v1',frozen['existing_training_monitor_PID'])
        preflight()
        assert len(job['gpu_uuids']) == 2 and job['config']['max_user_gpus'] == 4
        assert job['script_sha256'] == train['code_sha256']['run_bn4b_unclip20.sh']
        assert job['config']['job']['args'] == []
        save('releasing_checkpoint20')
        run([python,code/'release_bn4b_unclip20_checkpoint20.py','--root',root])
        release = read(root/evaluation['checkpoint_release_relative'])
        proof = read(root/train['checkpoint_relative']/'SERVER_PILOT20_UPDATES_DONE.json')
        seconds = (datetime.datetime.fromisoformat(job['finished_at'])-datetime.datetime.fromisoformat(job['started_at'])).total_seconds()
        completed = {'completed_at':datetime.datetime.now().isoformat(),'protocol':train['protocol'],
                     'global_step':20,'exposures':640,'training_valid_response_loss_tokens':proof['cost']['valid_response_loss_positions'],
                     'training_generated_tokens':sum(x['generated_tokens'] for x in proof['batch_generation']),
                     'training_parent_GPU_hours':seconds*2/3600,'adapter_sha256':release['adapter_sha256'],
                     'Base_unchanged_bitwise':proof['base_unchanged_bitwise'],
                     'matched640inputexposureorder':release['exposure_order_matches_clipped_reference'],
                     'scope':'Unclipped intervention training completion only; no downstream efficacy or officialdefault reproduction claim.'}
        with (code/'bn4b_unclip20_training_completed_20261023.PUBLIC.json').open('x') as stream:
            stream.write(json.dumps(completed,indent=2)+'\n')
        # Full evaluator preflight now has a real fixed checkpoint and release.
        subprocess.run(['bash',str(code/'run_bn4b_unclip20_followup200.sh'),'Unclip20'],
                       env={**environment,'COPSD_BN4B_UNCLIP_EVAL_PREPARE_ONLY':'1'},check=True)
        preflight()
        run(['python3',queue_tool,'--queue','copsd-bn4b-unclip20-eval-v1','validate'])
        save('starting_one_evaluation_queue')
        status['evaluation_start_calls'] = 1
        save('starting_one_evaluation_queue')
        run(['python3',queue_tool,'--queue','copsd-bn4b-unclip20-eval-v1','start'])
        monitor = subprocess.run(['systemctl','--user','show','gpu-queue@copsd-bn4b-unclip20-eval-v1.service','-p','MainPID','--value'],
                                 capture_output=True,text=True,check=True,timeout=15)
        evaluation_monitor = int(monitor.stdout.strip())
        assert evaluation_monitor > 0
        status['evaluation_monitor_PID'] = evaluation_monitor
        save('waiting_for_evaluation')
        wait_queue('copsd-bn4b-unclip20-eval-v1',evaluation_monitor)
        preflight()
        save('CPU_comparison_and_language_analysis')
        run([python,code/'collect_bn4b_unclip20.py','--root',root,'--output',code/'bn4b_unclip20_completed_20261023.PUBLIC.json'])
        for lang,label in [('bn','__label__ben_Beng'),('en','__label__eng_Latn')]:
            items = root/'logs/copsd_official_10lang/bn4b_unclip20_followup_development200'/f'bn4b_unclip20_followup200_Unclip20_{lang}_20261023/ITEMS.jsonl'
            output = root/'logs/copsd_official_10lang/bn4b_unclip20_language_20261023'/f'Unclip20_{lang}.PRIVATE.json'
            run([root/'envs/copsd-official-20261015/bin/python',code/'analyze_output_languages.py','--items',items,'--target-label',label,'--output',output])
        save('done')
        print('BN4B_UNCLIP20_SINGLE_TRAIN_EVAL_COMPARISON_SERIES_DONE',flush=True)
    except BaseException as error:
        status['error'] = repr(error)
        status['actual_jobs_cancelled_or_restarted'] = False
        save('observer_or_postprocessing_error_inspect_original_job')
        raise


if __name__ == '__main__':
    main()
