"""Manual one-shot successor: observe ZH training, release20, evaluate once, collect.

No retry/cancellation, boot entry, checkpoint selection or scientific retuning.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--train-monitor',type=int,required=True);p.add_argument('--base-monitor',type=int,required=True);a=p.parse_args()
    root=a.root;code=Path(__file__).parent;python=root/'envs/copsd-vllm011-20261018/bin/python';tool=root/'gpu-queue/gpu_queue.py'
    tp=code/'zh4b_server20_frozen_20261023.PUBLIC.json';ep=code/'zh4b_eval_frozen_20261023.PUBLIC.json';train=read(tp);evaluation=read(ep)
    pins={tp:sha(tp),ep:sha(ep),**{code/n:d for n,d in train['code_sha256'].items()},**{code/n:d for n,d in evaluation['runtime_code_sha256'].items()},Path(__file__):sha(Path(__file__))}
    queues=['copsd-zh4b-server20-v1','copsd-zh4b-followup-base-v1','copsd-zh4b-followup-zh20-v1']
    pins.update({root/'gpu-queue/queues'/q/'config.json':sha(root/'gpu-queue/queues'/q/'config.json') for q in queues})
    def verify():
        assert evaluation['training_freeze_sha256']==sha(tp)
        for path,digest in pins.items():assert sha(path)==digest,str(path)
        for q in queues:assert read(root/'gpu-queue/queues'/q/'config.json')['max_user_gpus']==4
    verify();status_path=root/'logs/copsd_official_10lang/zh4b_candidate_series_20261023.PRIVATE.json';assert not status_path.exists()
    status={'observer_PID':os.getpid(),'started_at':datetime.datetime.now().isoformat(),'automatic_retries':False,'trained_evaluation_start_calls':0,'code_bindings':{str(p):d for p,d in pins.items()}}
    def save(phase):status.update(phase=phase,updated_at=datetime.datetime.now().isoformat());status_path.write_text(json.dumps(status,indent=2)+'\n')
    def wait(queue,monitor):
        deadline=time.monotonic()+8*3600
        while True:
            s=read(root/'gpu-queue/queues'/queue/'state.json')
            if s['phase']=='completed' and s['job']['exit_code']==0:return
            if s['phase'] in ['failed','stopped','cancelled']:raise RuntimeError((queue,s['phase']))
            assert s['monitor_pid']==monitor and monitor>0
            try:os.kill(monitor,0)
            except ProcessLookupError:
                time.sleep(2);s=read(root/'gpu-queue/queues'/queue/'state.json')
                if s['phase']=='completed' and s['job']['exit_code']==0:return
                raise RuntimeError((queue,'Original monitor missing; inspect original job, no restart'))
            if time.monotonic()>deadline:raise TimeoutError('Observer deadline only; actual jobs unchanged')
            time.sleep(30)
    environment={**os.environ,'CUDA_VISIBLE_DEVICES':''}
    def run(command,extra=None):subprocess.run([str(x) for x in command],env={**environment,**(extra or {})},check=True)
    try:
        save('waiting_for_existing_training');wait(queues[0],a.train_monitor);verify()
        run([python,code/'release_zh4b_checkpoint20.py','--root',root])
        run(['bash',code/'run_zh4b_followup200.sh','ZH20'],{'COPSD_ZH4B_EVAL_PREPARE_ONLY':'1'})
        verify();run(['python3',tool,'--queue',queues[2],'validate'])
        status['trained_evaluation_start_calls']=1;save('starting_one_trained_evaluation')
        run(['python3',tool,'--queue',queues[2],'start'])
        monitor=int(subprocess.check_output(['systemctl','--user','show','gpu-queue@'+queues[2]+'.service','-p','MainPID','--value'],text=True).strip());assert monitor>0
        status['trained_evaluation_monitor_PID']=monitor;save('waiting_for_fixed_evaluation_pair')
        wait(queues[1],a.base_monitor);wait(queues[2],monitor);verify()
        save('CPU_collection');run([python,code/'collect_zh4b_followup200.py','--root',root,'--output',code/'zh4b_followup200_completed_20261023.PUBLIC.json'])
        for lang,label in [('zh','__label__zho_Hans'),('en','__label__eng_Latn')]:
            items=root/'logs/copsd_official_10lang/zh4b_followup_development200'/f'zh4b_followup200_ZH20_{lang}_20261023/ITEMS.jsonl'
            output=root/'logs/copsd_official_10lang/zh4b_language_20261023'/f'ZH20_{lang}.PRIVATE.json'
            run([root/'envs/copsd-official-20261015/bin/python',code/'analyze_output_languages.py','--items',items,'--target-label',label,'--output',output])
        save('done');print('ZH4B_CANDIDATE_SERIES_DONE',flush=True)
    except BaseException as error:
        status['error']=repr(error);status['actual_jobs_cancelled_or_restarted']=False;save('observer_or_postprocessing_error_inspect_original_job');raise


if __name__=='__main__':main()
