import pathlib,sys,json,hashlib,time,os
ROOT=pathlib.Path(__file__).resolve().parents[3]
REPORT=ROOT/'outputs/infonce_strength_v1'
RUN=ROOT/'work/infonce_strength_v1'
OLD=ROOT/'outputs/Multilingual-CL-alignment-recovery-v3/experiments/representation_alignment_v1'
H=pathlib.Path('/data0/users/lixiaoyang')
TRAIN=H/'models/checkpoint/representation_alignment_20261021'
NEWTRAIN=H/'models/checkpoint/infonce_strength_20261022'
BASE=H/'models/Qwen2.5-1.5B-Instruct'
sys.path.insert(0,str(OLD))
from common import sha,write,rows

def acquire(uuid,label):
 import fcntl,time
 sys.path.insert(0,str(H/'gpu-queue'));import gpu_queue as q
 with (H/'gpu-queue/reservation.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  for _ in range(2):
   snapshot=q.snapshot();reserved=q.other_queue_gpus();assigned=q.user_gpu_assignments(snapshot,reserved)
   assert len(assigned)<4,(assigned,'account cap4')
   assert uuid not in reserved and q.idle(snapshot[uuid],{'max_idle_memory_mib':512,'max_idle_util_percent':5})
   time.sleep(1)
  os.environ['CUDA_VISIBLE_DEVICES']=uuid;os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
  import torch
  torch.cuda.init();x=torch.empty(1,device='cuda');torch.cuda.synchronize();del x
  write(RUN/('WORKER_'+label+'.json'),{'pid':os.getpid(),'uuid':uuid,'account_assigned_before':sorted(assigned),'started_unix':time.time(),'cap':4})
