"""Adapter only: reuse immutable prior generation+scoring, change model IDs/output, enforce new budget."""
import settings as d
import argparse,fcntl,traceback,runpy
p=argparse.ArgumentParser();p.add_argument('--uuid',required=True);p.add_argument('--model',required=True);a=p.parse_args();label=a.model
assert d.read(d.RUN/'AUDIT_DONE.json')['passed'];d.guard()
try:
 d.sys.path.insert(0,str(d.H/'gpu-queue'));import gpu_queue as q
 with (d.H/'gpu-queue/reservation.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  for _ in range(2):
   snap=q.snapshot();reserved=q.other_queue_gpus();assigned=q.user_gpu_assignments(snap,reserved)
   assert len(assigned)<4 and a.uuid not in reserved;assert q.idle(snap[a.uuid],{'max_idle_memory_mib':512,'max_idle_util_percent':5});d.time.sleep(1)
  d.os.environ['CUDA_VISIBLE_DEVICES']=a.uuid;d.os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';import torch
  torch.cuda.init();probe=torch.empty(1,device='cuda');torch.cuda.synchronize();del probe
  d.write(d.RUN/('WORKER_'+label+'.json'),{'pid':d.os.getpid(),'uuid':a.uuid,'assigned_before':sorted(assigned),'started_unix':d.time.time(),'account_cap':4,'adapter_sha256':d.sha(__file__),'protocol_sha256':d.sha(d.REPORT/'PROTOCOL_FROZEN.json')})
 del d.sys.modules['settings'];d.sys.path.insert(0,str(d.PRIOR/'code'))
 import evaluate as ev
 ev.RUN=d.RUN;ev.budget_guard=d.guard
 d.sys.argv=['evaluate.py','--model',a.model,'--split','confirmation'];ev.main()
 d.write(d.RUN/('WORKER_DONE_'+label+'.json'),{'finished_unix':d.time.time()})
except BaseException as e:
 d.write(d.RUN/('WORKER_FAILED_'+label+'.json'),{'error':str(e),'traceback':traceback.format_exc(),'finished_unix':d.time.time()});raise
