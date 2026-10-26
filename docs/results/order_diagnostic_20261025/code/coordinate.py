"""Manually launched one-shot queue, max2 evaluation workers, account limit4, no automatic retries."""
from settings import *
import subprocess,traceback
sys.path.insert(0,str(H/'gpu-queue'));import gpu_queue as q
start=time.time();pending=['LR077-s42-mid','LR077-s43-mid'];live={};done=[];failures=[]
try:
 while pending or live:
  assert time.time()-start<86400,'Manual24h wait deadline; preserve missing status'
  for name,info in list(live.items()):
   rc=info['process'].poll()
   if rc is not None:
    info['stream'].close();del live[name];write(RUN/('TASK_'+name+'.json'),{'exit_code':rc,'finished_unix':time.time()});(failures if rc else done).append(name)
  if not failures and (RUN/'AUDIT_DONE.json').exists():
   assert read(RUN/'AUDIT_DONE.json')['passed']
   for name in list(pending):
    if usage()[0]>6000:break
    snap=q.snapshot();reserved=q.other_queue_gpus();assigned=q.user_gpu_assignments(snap,reserved);occupied={i['uuid'] for i in live.values()}
    if len(assigned|occupied)>=4:break
    candidates=[u for u,g in snap.items() if u not in reserved|occupied and q.idle(g,{'max_idle_memory_mib':512,'max_idle_util_percent':5})]
    if not candidates:break
    uuid=candidates[0];log=RUN/(name+'.log');stream=log.open('x');cmd=[str(PYTHON),str(REPORT/'code/evaluate_mid.py'),'--uuid',uuid,'--model',name];proc=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,cwd=ROOT);live[name]={'process':proc,'stream':stream,'uuid':uuid};pending.remove(name);write(RUN/('LAUNCH_'+name+'.json'),{'pid':proc.pid,'uuid':uuid,'cmd':cmd,'log':str(log),'started_unix':time.time()})
  status={'pending':pending,'running':{n:{'pid':i['process'].pid,'uuid':i['uuid']} for n,i in live.items()},'done':done,'failures':failures,'GPU_context_hours':usage()[0]/3600,'updated_unix':time.time(),'waiting_for_CPU_audit':not (RUN/'AUDIT_DONE.json').exists(),'account_cap':4}
  write(RUN/'SERIES_PROGRESS.json',status)
  if failures and not live:raise RuntimeError('Retained failure requires diagnosis; no blind retry')
  if pending and not live and usage()[0]>6000:raise RuntimeError('Budget reserve reached; missing tasks reported')
  if pending or live:time.sleep(30)
 write(RUN/'SERIES_DONE.json',{'finished_unix':time.time(),'done':done,'GPU_context_hours':usage()[0]/3600})
except BaseException as e:write(RUN/'SERIES_FAILED.json',{'error':str(e),'traceback':traceback.format_exc(),'finished_unix':time.time()});raise
