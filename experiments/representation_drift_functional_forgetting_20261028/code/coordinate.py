"""One inference-only stream on the GPU released by the user's MoFO pause."""
from pathlib import Path
import os,sys,json,time,subprocess,fcntl
ROOT=Path(__file__).resolve().parents[3];H=Path('/data0/users/lixiaoyang')
REPORT=ROOT/'docs/results/representation_drift_functional_forgetting_20261028'
WORK=ROOT/'work/representation_drift_functional_forgetting_20261028'
CODE=Path(__file__).parent;PY=H/'ssu-migration/envs/cl/bin/python'
sys.path.insert(0,str(ROOT/'outputs/sw_fullbudget_baselines_20261028/code'))
from settings import read,write,sha
protocol=read(REPORT/'PROTOCOL_FROZEN.json');uuid=protocol['execution']['dedicated_uuid']
models=protocol['models'];pending=list(models);done=[]
with (WORK/'coordinator.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 while pending:
  ready=[]
  for model in pending:
   if (WORK/model/'first/DONE.json').exists():done.append(model);continue
   if model=='Base' or (H/'models/checkpoint/sw_fullbudget_baselines_20261028'/f'{model}_first/TRAINING_DONE.json').exists():ready.append(model)
  pending=[m for m in pending if m not in done]
  write(WORK/'QUEUE_STATE.json',dict(pending=pending,completed_extractions=done,ready=ready,
   dedicated_uuid=uuid,max_account_GPUs=4,updated_unix=time.time(),no_training=True,
   awaiting_matched_fullbudget_endpoints=[m for m in pending if m not in ready]))
  if not ready:
   if pending:time.sleep(30)
   continue
  model=ready[0];logfile=WORK/(model+'_first.log')
  env=dict(os.environ,HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
   PYTHONHASHSEED='42',OMP_NUM_THREADS='4',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
  with logfile.open('a') as log:
   cmd=[str(PY),str(CODE/'extract.py'),'--model',model,'--uuid',uuid]
   proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
   write(WORK/'ACTIVE_JOB.json',dict(pid=proc.pid,model=model,uuid=uuid,command=cmd,started_unix=time.time()))
   rc=proc.wait()
  if rc==75:time.sleep(30);continue
  write(WORK/('EXIT_'+model+'_first.json'),dict(exit_code=rc,finished_unix=time.time()))
  if rc:
   write(WORK/'QUEUE_FAILED.json',dict(model=model,exit_code=rc,log=str(logfile)));raise RuntimeError('Inspect failed extraction before recovery')
  assert (WORK/model/'first/DONE.json').exists()
 write(WORK/'ALL_EXTRACTIONS_DONE.json',dict(models=done,finished_unix=time.time(),statistical_analysis_and_scientific_interpretation_pending=True))
