"""Manually launched one-shot follow-up; starts only after full baseline evaluation."""
from pathlib import Path
import sys,json,os,time,hashlib,subprocess,fcntl,traceback
ROOT=Path(__file__).resolve().parents[3];REPORT=ROOT/'outputs/access_gap_fullbudget_followup_20261028'
RUN=ROOT/'work/access_gap_fullbudget_followup_20261028'
sys.path.insert(0,str(ROOT/'outputs/sw_fullbudget_baselines_20261028/code'))
from settings import H,PYTHON,read,write,sha
sys.path.insert(0,str(H/'gpu-queue'));import gpu_queue as q
sys.path.insert(0,str(ROOT/'outputs/benchmark_direction_20261026'));from resource_guard import all_queue_reservations

def snapshot(pending,done,live=None):
 write(RUN/'QUEUE_STATE.json',dict(pending=pending,done=done,running=live,max_account_GPUs=4,time_stop_limit=None,updated_unix=time.time()))

with (RUN/'coordinator.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  protocol=read(REPORT/'FOLLOWUP_PROTOCOL_FROZEN.json');dependency=Path(protocol['dependency'])
  pending=[(m,stage) for m in protocol['methods'] for stage in ('smoke','MCF')];done=[]
  while not dependency.exists():
   snapshot(pending,done)
   if (dependency.parent/'QUEUE_FAILED.json').exists():raise RuntimeError('Upstream baseline queue failed; inspect before starting follow-up')
   time.sleep(60)
  identities={}
  for model in protocol['methods']:
   receipt=read(H/'models/checkpoint/sw_fullbudget_baselines_20261028'/f'{model}_first/TRAINING_DONE.json')
   assert receipt['steps']==42465 and receipt['counts']['main_input']==15360935
   path=Path(receipt['final_model']);digest=sha(path/'model.safetensors')
   assert digest==receipt['final_weights_sha256']['model.safetensors']
   identities[model]=dict(model=str(path),sha256=digest)
  write(RUN/'MODELS_FROZEN.json',identities)
  while pending:
   model,stage=pending[0];label=model+'_'+stage;destination=RUN/stage/model
   if (destination/'DONE.json').exists():
    assert read(destination/'DONE.json')['passed'];done.append(label);pending.pop(0);continue
   assert not destination.exists(),'Incomplete output exists; preserve and inspect before recovery'
   snap=q.snapshot();reserved=all_queue_reservations(q);assigned=q.user_gpu_assignments(snap,reserved)
   idle=[uuid for uuid,gpu in snap.items() if uuid not in reserved and q.idle(gpu,dict(max_idle_memory_mib=512,max_idle_util_percent=5))]
   if len(assigned)>=4 or not idle:snapshot(pending,done);time.sleep(60);continue
   uuid=idle[0];cmd=[str(PYTHON),str(REPORT/'code/worker.py'),'--model',model,'--uuid',uuid]+(['--smoke'] if stage=='smoke' else [])
   logfile=RUN/f'{label}_{time.time_ns()}.log'
   with logfile.open('x') as stream:
    env=dict(os.environ,HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
    snapshot(pending,done,dict(pid=process.pid,uuid=uuid,job=label,log=str(logfile)))
    rc=process.wait()
   if rc==75:time.sleep(30);continue
   assert rc==0,f'{label} failed with exit {rc}; see {logfile}'
  subprocess.run([str(PYTHON),str(REPORT/'code/analyze.py')],cwd=ROOT,check=True)
  write(RUN/'ALL_DONE.json',dict(passed=True,done=done,finished_unix=time.time()))
 except BaseException as error:
  write(RUN/'QUEUE_FAILED.json',dict(error=str(error),traceback=traceback.format_exc(),failed_unix=time.time()));raise
