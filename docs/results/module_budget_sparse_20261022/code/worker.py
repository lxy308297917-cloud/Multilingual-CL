"""Manual one-shot worker; shared account reservation lock, no hold or automatic retry."""
from settings import *
import argparse,fcntl,runpy,traceback
p=argparse.ArgumentParser();p.add_argument('--uuid',required=True);p.add_argument('--script',required=True);p.add_argument('args',nargs=argparse.REMAINDER);a=p.parse_args();label=a.script.replace('.py','')+'_'+str(os.getpid())
try:
 sys.path.insert(0,str(H/'gpu-queue'));import gpu_queue as q
 with (H/'gpu-queue/reservation.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  for _ in range(2):
   snap=q.snapshot();reserved=q.other_queue_gpus();assigned=q.user_gpu_assignments(snap,reserved);assert len(assigned)<8 and a.uuid not in reserved;assert q.idle(snap[a.uuid],{'max_idle_memory_mib':512,'max_idle_util_percent':5});time.sleep(1)
  os.environ['CUDA_VISIBLE_DEVICES']=a.uuid;os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';import torch
  torch.cuda.init();probe=torch.empty(1,device='cuda');torch.cuda.synchronize();del probe;write(RUN/('WORKER_'+label+'.json'),{'pid':os.getpid(),'uuid':a.uuid,'assigned_before':sorted(assigned),'started_unix':time.time(),'account_cap':8})
 sys.argv=[a.script]+a.args;runpy.run_path(str(REPORT/'code'/a.script),run_name='__main__');write(RUN/('WORKER_DONE_'+label+'.json'),{'finished_unix':time.time()})
except BaseException as e:write(RUN/('WORKER_FAILED_'+label+'.json'),{'error':str(e),'traceback':traceback.format_exc(),'finished_unix':time.time()});raise
