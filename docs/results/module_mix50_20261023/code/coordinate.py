"""Manual, bounded scheduler; checks actual account ownership and shared reservations; no holds/retries."""
from settings import *
import subprocess
sys.path.insert(0,str(H/'gpu-queue'));import gpu_queue as q

def main():
 assert (REPORT/'SOURCE_FROZEN.json').exists() and (REPORT/'PROTOCOL_FROZEN.json').exists();start=time.time();pending=[{'id':n,'script':'group_pipeline.py','args':[],'env':{'MIX50_JOB':n}} for n in JOBS]
 pending += [{'id':'confirmation-'+n,'script':'evaluate.py','args':['--model',n,'--split','confirmation'],'env':{}} for n in ('Base','LR077-s42','Attn-s42')]
 pending += [{'id':'prompt-'+n,'script':'evaluate.py','args':['--model',n,'--split','prompt'],'env':{}} for n in ('Base','LR077-s42','Attn-s42','MLP-s42')]
 live={};done=[];failures=[]
 while pending or live:
  assert time.time()-start<24*3600,'Bounded manual24h series wait'
  for name,info in list(live.items()):
   rc=info['process'].poll()
   if rc is None:continue
   info['stream'].close();del live[name];record={'id':name,'exit_code':rc,'finished_unix':time.time()};write(RUN/'task_receipts'/(name+'.json'),record)
   if rc:failures.append(record);print('FAILED',name,rc,flush=True)
   else:done.append(name);print('DONE',name,flush=True)
  # A failed run suspends new launches, but never interrupts existing authorized jobs.
  if not failures:
   for job in list(pending):
    if budget_seconds()[0]>15*3600:break
    snap=q.snapshot();reserved=q.other_queue_gpus();assigned=q.user_gpu_assignments(snap,reserved);occupied={x['uuid'] for x in live.values()}
    if len(assigned|occupied)>=4:break
    candidates=[u for u,g in snap.items() if u not in reserved|occupied and q.idle(g,{'max_idle_memory_mib':512,'max_idle_util_percent':5})]
    if not candidates:break
    uuid=candidates[0];env=dict(os.environ,**job['env']);log=H/'logs/module_mix50_20261023'/(job['id']+'.log');stream=log.open('x');cmd=[sys.executable,str(REPORT/'code/worker.py'),'--uuid',uuid,'--script',job['script']]+job['args'];proc=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,cwd=ROOT,env=env);live[job['id']]={'process':proc,'stream':stream,'uuid':uuid};pending.remove(job);write(RUN/'claims'/(job['id']+'.json'),{'pid':proc.pid,'uuid':uuid,'command':cmd,'environment':job['env'],'log':str(log),'launched_unix':time.time(),'assigned_before':sorted(assigned),'max_account_gpus':4});print('LAUNCH',job['id'],uuid,proc.pid,flush=True)
  write(RUN/'SERIES_PROGRESS.json',{'pending':[j['id'] for j in pending],'running':{n:{'pid':i['process'].pid,'uuid':i['uuid']} for n,i in live.items()},'done':done,'failures':failures,'GPU_context_hours':budget_seconds()[0]/3600,'updated_unix':time.time(),'account_cap':4})
  if failures and not live:raise RuntimeError('Diagnose retained failures before authorized recovery, no blind retry')
  if pending and not live and budget_seconds()[0]>15*3600:raise RuntimeError('Budget reserve reached before launch; report missing work, no silent deletion')
  if pending or live:time.sleep(30)
 write(RUN/'SERIES_DONE.json',{'done':done,'GPU_context_hours':budget_seconds()[0]/3600,'finished_unix':time.time()});print('SERIES ALL GPU TASKS COMPLETE',flush=True)
if __name__=='__main__':main()
