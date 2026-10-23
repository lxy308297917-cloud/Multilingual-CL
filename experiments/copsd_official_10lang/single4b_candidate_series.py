"""Attach one fresh single4B training and Base evaluation; one final model eval.

No training start/cancel/retry. Queue limits include other jobs/reservations.
Observer timeouts leave original experiments unchanged.
"""
import argparse,datetime,hashlib,json,os
from pathlib import Path
import subprocess,time


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--freeze',type=Path,required=True);p.add_argument('--train-monitor',type=int,required=True);p.add_argument('--base-monitor',type=int,required=True);a=p.parse_args();root=a.root;code=Path(__file__).parent;python=root/'envs/copsd-vllm011-20261018/bin/python';tool=root/'gpu-queue/gpu_queue.py'
    ef=a.freeze;f=read(ef);tf=code/f['training_config_name'];t=read(tf);train=f['training_queue'];base=f['queues']['Base'];case=f['trained_case'];evaluation=f['queues'][case]
    pins={tf:sha(tf),ef:sha(ef),Path(__file__):sha(Path(__file__))};pins.update({code/n:h for n,h in t['code_sha256'].items()});pins.update({code/n:h for n,h in f['runtime_code_sha256'].items()});pins.update({root/'gpu-queue/queues'/q/'config.json':sha(root/'gpu-queue/queues'/q/'config.json') for q in [train,base,evaluation]})
    def verify():
        for path,digest in pins.items():assert sha(path)==digest,str(path)
        assert f['training_freeze_sha256']==sha(tf)
        for q in [train,base,evaluation]:assert read(root/'gpu-queue/queues'/q/'config.json')['max_user_gpus']==4
    verify();status_path=root/'logs/copsd_official_10lang'/(f['run_prefix']+'.series.PRIVATE.json');status={'observer_PID':os.getpid(),'started_at':datetime.datetime.now().isoformat(),'training_monitor_PID':a.train_monitor,'Base_monitor_PID':a.base_monitor,'evaluation_start_calls':0,'automatic_retries':False,'code_bindings':{str(p):h for p,h in pins.items()}}
    with status_path.open('x') as stream:stream.write(json.dumps(status,indent=2)+'\n')
    def save(phase):
        status.update(phase=phase,updated_at=datetime.datetime.now().isoformat());temp=status_path.with_suffix('.tmp');temp.write_text(json.dumps(status,indent=2)+'\n');temp.replace(status_path);print(phase,flush=True)
    def wait(queue,monitor):
        path=root/'gpu-queue/queues'/queue/'state.json';deadline=time.monotonic()+8*3600;missing=None
        while True:
            try:s=read(path);missing=None
            except (FileNotFoundError,json.JSONDecodeError):
                os.kill(monitor,0);missing=time.monotonic() if missing is None else missing
                if time.monotonic()-missing>30:raise TimeoutError('Observation publication grace expired; inspect originaljob')
                time.sleep(1);continue
            if s['phase']=='completed' and s.get('job',{}).get('exit_code')==0:return
            if s['phase'] in ['failed','stopped','cancelled']:raise RuntimeError((queue,s['phase']))
            assert s['monitor_pid']==monitor
            try:os.kill(monitor,0)
            except ProcessLookupError:
                time.sleep(2);s=read(path)
                if s['phase']=='completed' and s.get('job',{}).get('exit_code')==0:return
                raise RuntimeError('Originalmonitor missing; inspect actualjob, no restart')
            if time.monotonic()>deadline:raise TimeoutError('Observerdeadline only; actualjob unchanged')
            time.sleep(30)
    env={**os.environ,'CUDA_VISIBLE_DEVICES':''}
    def run(command,extra=None):subprocess.run([str(x) for x in command],env={**env,**(extra or {})},check=True)
    try:
        save('waiting_for_existing_training');wait(train,a.train_monitor);verify()
        save('releasing_checkpoint20');run([python,code/'release_single4b_checkpoint20.py','--root',root,'--freeze',ef]);run(['bash',code/'run_single4b_development.sh',ef,case],{'COPSD_SINGLE4B_EVAL_PREPARE_ONLY':'1'});verify();run(['python3',tool,'--queue',evaluation,'validate'])
        status['evaluation_start_calls']=1;save('starting_one_final_model_evaluation');run(['python3',tool,'--queue',evaluation,'start']);monitor=int(subprocess.check_output(['systemctl','--user','show','gpu-queue@'+evaluation+'.service','-p','MainPID','--value'],text=True).strip());assert monitor>0;status['evaluation_monitor_PID']=monitor
        save('waiting_for_existing_Base_and_final_model_evaluations');wait(base,a.base_monitor);wait(evaluation,monitor);verify()
        save('collecting_paired_development');run([python,code/'collect_single4b_development.py','--root',root,'--freeze',ef,'--output',code/f['completed_report_name']])
        target=f['target_language'].lower()
        for model,lang,label in [('Base',target,f['target_language_GlotLID_label']),(case,target,f['target_language_GlotLID_label']),(case,'en','__label__eng_Latn')]:
            run_id=f['run_prefix']+'_'+model+'_'+lang;items=root/f['output_relative']/run_id/'ITEMS.jsonl';output=root/f['language_analysis_relative']/(model+'_'+lang+'.PRIVATE.json')
            run([root/'envs/copsd-official-20261015/bin/python',code/'analyze_output_languages.py','--items',items,'--target-label',label,'--output',output])
        save('done')
    except BaseException as error:
        status.update(error=repr(error),actual_jobs_cancelled_or_restarted=False);save('observer_or_postprocessing_error_inspect_original_job');raise


if __name__=='__main__':main()
