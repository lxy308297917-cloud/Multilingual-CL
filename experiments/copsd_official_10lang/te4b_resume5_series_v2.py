"""Attach to recovered TE training; release and evaluate its one final adapter.

No training launch/retry/cancel. Originalfailedrun remains failed and archived.
AcceptedBaseTE andBaseEN reused. Observer expiry is not experiment failure.
"""
import argparse,datetime,hashlib,json,os
from pathlib import Path
import subprocess,time


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--train-monitor',type=int,required=True);a=p.parse_args();root=a.root;code=Path(__file__).parent;python=root/'envs/copsd-vllm011-20261018/bin/python';tool=root/'gpu-queue/gpu_queue.py'
    train='copsd-te4b-resume5-v1';evaluation='copsd-te4b-resume5-eval-v1';tf=code/'te4b_resume5_frozen_20261024.PUBLIC.json';ef=code/'te4b_resume5_eval_frozen_20261024.PUBLIC.json';t=read(tf);f=read(ef)
    pins={tf:sha(tf),ef:sha(ef),Path(__file__):sha(Path(__file__))};pins.update({code/n:h for n,h in t['code_sha256'].items()});pins.update({code/n:h for n,h in f['runtime_code_sha256'].items()});pins.update({root/'gpu-queue/queues'/q/'config.json':sha(root/'gpu-queue/queues'/q/'config.json') for q in [train,evaluation]})
    def verify():
        for path,digest in pins.items():assert sha(path)==digest,str(path)
        assert f['training_freeze_sha256']==sha(tf)
        base=read(root/'gpu-queue/queues/copsd-te4b-followup-base-v1/state.json');assert base['phase']=='completed' and base['job']['exit_code']==0
        cache=f['cached_Base_TE'];assert sha(code/cache['accepted_report'])==cache['accepted_report_sha256']
        for q in [train,evaluation]:assert read(root/'gpu-queue/queues'/q/'config.json')['max_user_gpus']==4
    verify();path=root/'logs/copsd_official_10lang/te4b_resume5_series_v2_20261024.PRIVATE.json';status={'observer_PID':os.getpid(),'started_at':datetime.datetime.now().isoformat(),'training_monitor_PID':a.train_monitor,'evaluation_start_calls':0,'automatic_retries':False,'code_bindings':{str(p):h for p,h in pins.items()}}
    with path.open('x') as stream:stream.write(json.dumps(status,indent=2)+'\n')
    def save(phase):
        status.update(phase=phase,updated_at=datetime.datetime.now().isoformat());tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(status,indent=2)+'\n');tmp.replace(path);print(phase,flush=True)
    def wait(queue,monitor):
        state_path=root/'gpu-queue/queues'/queue/'state.json';deadline=time.monotonic()+8*3600;missing=None
        while True:
            try:s=read(state_path);missing=None
            except (FileNotFoundError,json.JSONDecodeError):
                os.kill(monitor,0);missing=time.monotonic() if missing is None else missing
                if time.monotonic()-missing>30:raise TimeoutError('Observation publication grace expired; inspect originaljob')
                time.sleep(1);continue
            if s['phase']=='completed' and s.get('job',{}).get('exit_code')==0:return
            if s['phase'] in ['failed','stopped','cancelled']:raise RuntimeError((queue,s['phase']))
            assert s['monitor_pid']==monitor
            try:os.kill(monitor,0)
            except ProcessLookupError:
                time.sleep(2);s=read(state_path)
                if s['phase']=='completed' and s.get('job',{}).get('exit_code')==0:return
                raise RuntimeError('Originalmonitor missing; inspect originaljob without restart')
            if time.monotonic()>deadline:raise TimeoutError('Observerdeadline only; actualjob unchanged')
            time.sleep(30)
    env={**os.environ,'CUDA_VISIBLE_DEVICES':''}
    def run(command,extra=None):subprocess.run([str(x) for x in command],env={**env,**(extra or {})},check=True)
    try:
        save('waiting_for_existing_checkpoint5_recovery');wait(train,a.train_monitor);verify()
        save('releasing_recovered_checkpoint20');run([python,code/'release_te4b_resume5_checkpoint20.py','--root',root])
        run(['bash',code/'run_te4b_resume5_followup200.sh','TE20R'],{'COPSD_TE4B_RESUME5_EVAL_PREPARE_ONLY':'1'});verify();run(['python3',tool,'--queue',evaluation,'validate'])
        status['evaluation_start_calls']=1;save('starting_one_recovered_model_evaluation');run(['python3',tool,'--queue',evaluation,'start'])
        monitor=int(subprocess.check_output(['systemctl','--user','show','gpu-queue@'+evaluation+'.service','-p','MainPID','--value'],text=True).strip());assert monitor>0;status['evaluation_monitor_PID']=monitor;save('waiting_for_recovered_model_evaluation');wait(evaluation,monitor);verify()
        save('collecting_paired_results_and_all_failed_work_cost');run([python,code/'collect_te4b_resume5_followup200.py','--root',root,'--output',code/'te4b_resume5_followup200_completed_20261024.PUBLIC.json'])
        for case,lang,label in [('Base','te','__label__tel_Telu'),('TE20R','te','__label__tel_Telu'),('TE20R','en','__label__eng_Latn')]:
            group='te4b_followup_development200' if case=='Base' else 'te4b_resume5_followup_development200';prefix='te4b_followup200' if case=='Base' else 'te4b_resume5_followup200';items=root/'logs/copsd_official_10lang'/group/f'{prefix}_{case}_{lang}_20261024/ITEMS.jsonl';output=root/'logs/copsd_official_10lang/te4b_resume5_language_20261024'/f'{case}_{lang}.PRIVATE.json'
            run([root/'envs/copsd-official-20261015/bin/python',code/'analyze_output_languages.py','--items',items,'--target-label',label,'--output',output])
        save('done')
    except BaseException as error:
        status.update(error=repr(error),actual_jobs_cancelled_or_restarted=False);save('observer_or_postprocessing_error_inspect_original_job');raise


if __name__=='__main__':main()
