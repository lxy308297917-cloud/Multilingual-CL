"""Attach only to the already-started ZH20 evaluation; CPU collection after exit0.

No GPU start/restart/cancel command exists here. Originalv2 failure was a read
before initialqueue state publication; bounded observation grace handles it.
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
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--eval-monitor',type=int,required=True);a=p.parse_args()
    root=a.root;code=Path(__file__).parent;python=root/'envs/copsd-vllm011-20261018/bin/python'
    fp=code/'zh4b_eval_frozen_20261023.PUBLIC.json';f=read(fp);tp=code/'zh4b_server20_frozen_20261023.PUBLIC.json';train=read(tp)
    pins={fp:sha(fp),tp:sha(tp),Path(__file__):sha(Path(__file__)),**{code/n:d for n,d in f['runtime_code_sha256'].items()},**{code/n:d for n,d in train['code_sha256'].items()}}
    for q in ['copsd-zh4b-server20-v1','copsd-zh4b-followup-base-v1','copsd-zh4b-followup-zh20-v1']:
        path=root/'gpu-queue/queues'/q/'config.json';pins[path]=sha(path)
    def verify():
        for path,digest in pins.items():assert sha(path)==digest,str(path)
        assert f['training_freeze_sha256']==sha(tp)
        for q in ['copsd-zh4b-server20-v1','copsd-zh4b-followup-base-v1']:
            s=read(root/'gpu-queue/queues'/q/'state.json');assert s['phase']=='completed' and s['job']['exit_code']==0,q
        release=read(root/f['checkpoint_release_relative']);assert release['evaluation_freeze_sha256']==sha(fp) and release['training_freeze_sha256']==sha(tp)
    verify();status_path=root/'logs/copsd_official_10lang/zh4b_candidate_finalizer_v3_20261023.PRIVATE.json';assert not status_path.exists()
    status={'observer_PID':os.getpid(),'started_at':datetime.datetime.now().isoformat(),'evaluation_monitor_PID':a.eval_monitor,'GPU_start_cancel_restart_calls':0,'automatic_retries':False,'code_bindings':{str(p):d for p,d in pins.items()}}
    def save(phase):status.update(phase=phase,updated_at=datetime.datetime.now().isoformat());status_path.write_text(json.dumps(status,indent=2)+'\n')
    def wait():
        deadline=time.monotonic()+8*3600;missing_since=None
        path=root/'gpu-queue/queues/copsd-zh4b-followup-zh20-v1/state.json'
        while True:
            try:s=read(path);missing_since=None
            except (FileNotFoundError,json.JSONDecodeError):
                missing_since=time.monotonic() if missing_since is None else missing_since
                os.kill(a.eval_monitor,0)
                if time.monotonic()-missing_since>30:raise TimeoutError('Statepublication observation grace expired; actualjob unchanged')
                time.sleep(1);continue
            if s['phase']=='completed' and s['job']['exit_code']==0:return
            if s['phase'] in ['failed','stopped','cancelled']:raise RuntimeError(('Originalevaluation',s['phase']))
            assert s['monitor_pid']==a.eval_monitor and s['job']['config']['job']['args']==['ZH20']
            try:os.kill(a.eval_monitor,0)
            except ProcessLookupError:
                time.sleep(2);last=read(path)
                if last['phase']=='completed' and last['job']['exit_code']==0:return
                raise RuntimeError('Originalevaluation monitor missing; inspect originaljob, no restart')
            if time.monotonic()>deadline:raise TimeoutError('Observer deadline only; actualjob unchanged')
            time.sleep(15)
    environment={**os.environ,'CUDA_VISIBLE_DEVICES':''}
    def run(command):subprocess.run([str(x) for x in command],env=environment,check=True)
    try:
        save('waiting_for_existing_evaluation');print('ATTACHED_EXISTING_ZH20_EVALUATION_CPU_ONLY',flush=True);wait();verify()
        save('CPU_collection');run([python,code/'collect_zh4b_followup200.py','--root',root,'--output',code/'zh4b_followup200_completed_20261023.PUBLIC.json'])
        for lang,label in [('zh','__label__cmn_Hani'),('en','__label__eng_Latn')]:
            items=root/'logs/copsd_official_10lang/zh4b_followup_development200'/f'zh4b_followup200_ZH20_{lang}_20261023/ITEMS.jsonl'
            output=root/'logs/copsd_official_10lang/zh4b_language_20261023'/f'ZH20_{lang}.PRIVATE.json'
            run([root/'envs/copsd-official-20261015/bin/python',code/'analyze_output_languages.py','--items',items,'--target-label',label,'--output',output])
        save('done');print('ZH4B_CPU_FINALIZATION_DONE',flush=True)
    except BaseException as error:
        status['error']=repr(error);status['actual_jobs_cancelled_or_restarted']=False;save('observer_or_postprocessing_error_inspect_original_job');raise


if __name__=='__main__':main()
