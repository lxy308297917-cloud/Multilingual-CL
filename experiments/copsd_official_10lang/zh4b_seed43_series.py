"""One manually attached confirmation: existing training -> one paired evaluation.

No retries, cancellation, boot startup, or training launches. Observer expiry
never implies job failure. Both accepted Base outputs are reused.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--train-monitor',type=int,required=True)
    a=parser.parse_args();root=a.root;code=Path(__file__).parent
    python=root/'envs/copsd-vllm011-20261018/bin/python';tool=root/'gpu-queue/gpu_queue.py'
    train='copsd-zh4b-seed43-v1';evaluation='copsd-zh4b-seed43-eval-v1'
    tf=code/'zh4b_seed43_frozen_20261023.PUBLIC.json';ef=code/'zh4b_seed43_eval_frozen_20261023.PUBLIC.json'
    t=read(tf);f=read(ef)
    pins={tf:sha(tf),ef:sha(ef),Path(__file__):sha(Path(__file__))}
    pins.update({code/n:h for n,h in t['code_sha256'].items()})
    pins.update({code/n:h for n,h in f['runtime_code_sha256'].items()})
    pins.update({root/'gpu-queue/queues'/q/'config.json':sha(root/'gpu-queue/queues'/q/'config.json') for q in [train,evaluation]})
    def verify():
        for path,digest in pins.items(): assert sha(path)==digest,str(path)
        assert f['training_freeze_sha256']==sha(tf)
        for q in [train,evaluation]:assert read(root/'gpu-queue/queues'/q/'config.json')['max_user_gpus']==4
    verify()
    status_path=root/'logs/copsd_official_10lang/zh4b_seed43_series_20261023.PRIVATE.json'
    status={'observer_PID':os.getpid(),'started_at':datetime.datetime.now().isoformat(),'training_monitor_PID':a.train_monitor,'evaluation_start_calls':0,'automatic_retries':False,'code_bindings':{str(p):h for p,h in pins.items()}}
    with status_path.open('x') as stream:stream.write(json.dumps(status,indent=2)+'\n')
    def save(phase):
        status.update(phase=phase,updated_at=datetime.datetime.now().isoformat())
        temp=status_path.with_suffix('.tmp');temp.write_text(json.dumps(status,indent=2)+'\n');temp.replace(status_path)
        print(phase,flush=True)
    def wait(queue,monitor):
        path=root/'gpu-queue/queues'/queue/'state.json';deadline=time.monotonic()+8*3600;missing=None
        while True:
            try:s=read(path);missing=None
            except (FileNotFoundError,json.JSONDecodeError):
                os.kill(monitor,0);missing=time.monotonic() if missing is None else missing
                if time.monotonic()-missing>30:raise TimeoutError('Observation state publication grace expired; inspect original job')
                time.sleep(1);continue
            if s['phase']=='completed' and s.get('job',{}).get('exit_code')==0:return
            if s['phase'] in ['failed','stopped','cancelled']:raise RuntimeError((queue,s['phase']))
            assert s['monitor_pid']==monitor
            try:os.kill(monitor,0)
            except ProcessLookupError:
                time.sleep(2);s=read(path)
                if s['phase']=='completed' and s.get('job',{}).get('exit_code')==0:return
                raise RuntimeError('Original monitor missing; inspect job without restarting')
            if time.monotonic()>deadline:raise TimeoutError('Observation deadline only; actual job unchanged')
            time.sleep(30)
    env={**os.environ,'CUDA_VISIBLE_DEVICES':''}
    def run(command,extra=None):subprocess.run([str(x) for x in command],env={**env,**(extra or {})},check=True)
    try:
        save('waiting_for_existing_seed43_training');wait(train,a.train_monitor);verify()
        save('releasing_checkpoint20');run([python,code/'release_zh4b_seed43_checkpoint20.py','--root',root])
        run(['bash',code/'run_zh4b_seed43_followup200.sh','ZH43'],{'COPSD_ZH4B_SEED43_EVAL_PREPARE_ONLY':'1'})
        verify();run(['python3',tool,'--queue',evaluation,'validate'])
        status['evaluation_start_calls']=1;save('starting_one_seed43_evaluation')
        run(['python3',tool,'--queue',evaluation,'start'])
        monitor=int(subprocess.check_output(['systemctl','--user','show','gpu-queue@'+evaluation+'.service','-p','MainPID','--value'],text=True).strip());assert monitor>0
        status['evaluation_monitor_PID']=monitor;save('waiting_for_seed43_evaluation');wait(evaluation,monitor);verify()
        save('collecting_paired_results');run([python,code/'collect_zh4b_seed43_followup200.py','--root',root,'--output',code/'zh4b_seed43_followup200_completed_20261023.PUBLIC.json'])
        for lang,label in [('zh','__label__cmn_Hani'),('en','__label__eng_Latn')]:
            items=root/'logs/copsd_official_10lang/zh4b_seed43_followup_development200'/f'zh4b_seed43_followup200_ZH43_{lang}_20261023/ITEMS.jsonl'
            output=root/'logs/copsd_official_10lang/zh4b_seed43_language_20261023'/f'ZH43_{lang}.PRIVATE.json'
            run([root/'envs/copsd-official-20261015/bin/python',code/'analyze_output_languages.py','--items',items,'--target-label',label,'--output',output])
        save('done')
    except BaseException as error:
        status.update(error=repr(error),actual_jobs_cancelled_or_restarted=False)
        save('observer_or_postprocessing_error_inspect_original_job');raise


if __name__=='__main__':main()
