"""One explicitly diagnosed evaluation-only recovery; original science and failures preserved."""
from settings import *
import subprocess
sys.path.insert(0,str(H/'gpu-queue'));import gpu_queue as q

def main():
 assert json.loads((REPORT/'RECOVERY_AUDIT.json').read_text())['CPU_CLI_regression_passed'];start=time.time();pending=[{'id':'confirmation-'+n,'model':n,'split':'confirmation'} for n in ('Base','LR077-s42','Attn-s42')]+[{'id':'prompt-'+n,'model':n,'split':'prompt'} for n in ('Base','LR077-s42','Attn-s42','MLP-s42')];live={};done=[];failures=[]
 while pending or live or not all((RUN/('GROUP_DONE_'+n+'.json')).exists() for n in JOBS):
  assert time.time()-start<12*3600
  for name,v in list(live.items()):
   rc=v['process'].poll()
   if rc is None:continue
   v['stream'].close();del live[name];rec={'id':name,'exit_code':rc,'finished_unix':time.time()};write(RUN/'recovery_receipts'/(name+'.json'),rec)
   if rc:failures.append(rec);print('RECOVERY FAILED',name,rc,flush=True)
   else:done.append(name);print('RECOVERY DONE',name,flush=True)
  if not failures:
   for job in list(pending):
    budget_guard();snap=q.snapshot();reserved=q.other_queue_gpus();assigned=q.user_gpu_assignments(snap,reserved);occupied={v['uuid'] for v in live.values()}
    if len(assigned|occupied)>=4:break
    choices=[u for u,g in snap.items() if u not in reserved|occupied and q.idle(g,{'max_idle_memory_mib':512,'max_idle_util_percent':5})]
    if not choices:break
    d=RUN/'evaluation'/job['split']/job['model'];assert not d.exists(),'Existing partial eval requires separate output/review, no overwrite';u=choices[0];log=H/'logs/module_mix50_20261023'/(job['id']+'_recovery1.log');stream=log.open('x');cmd=[sys.executable,str(REPORT/'code/recovery_worker.py'),'--uuid',u,'--script','evaluate.py','--model',job['model'],'--split',job['split']];proc=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,cwd=ROOT);live[job['id']]={'process':proc,'stream':stream,'uuid':u};pending.remove(job);write(RUN/'recovery_claims'/(job['id']+'.json'),{'pid':proc.pid,'uuid':u,'command':cmd,'log':str(log),'started_unix':time.time(),'account_cap':4});print('RECOVERY LAUNCH',job['id'],u,proc.pid,flush=True)
  write(RUN/'RECOVERY_PROGRESS.json',{'pending':[j['id'] for j in pending],'running':{n:{'pid':v['process'].pid,'uuid':v['uuid']} for n,v in live.items()},'done':done,'failures':failures,'GPU_context_hours':budget_seconds()[0]/3600,'updated_unix':time.time()})
  if failures and not live:raise RuntimeError('Recovery task failure; preserve output, no blind retry')
  if pending or live:time.sleep(30)
  elif not all((RUN/('GROUP_DONE_'+n+'.json')).exists() for n in JOBS):
   if any((RUN/('GROUP_FAILED_'+n+'.json')).exists() for n in JOBS):raise RuntimeError('A scientific group failed, cannot claim done')
   time.sleep(30)
 assert len(done)==7;write(RUN/'SERIES_DONE.json',{'done':list(JOBS)+done,'GPU_context_hours':budget_seconds()[0]/3600,'finished_unix':time.time(),'recovered_CLI_failure':True,'original_failure_preserved':True});print('ALL ORIGINAL SCIENCE AND RECOVERED EVALUATIONS COMPLETE',flush=True)
if __name__=='__main__':main()
