"""Runtime scheduling only: six core tasks, then MMLU/G-MMLU; frozen v3 workers."""
import sys,os,json,time,fcntl,subprocess,shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from run_baseline_eval import R,protocol,model_identity,digest,write,jobs,valid_done,sha
from baseline_eval_integrity import require_scores

def main():
 if len(sys.argv)!=2:raise SystemExit('Usage: baseline_language_first.py MODEL_PATH')
 model=Path(sys.argv[1]);cp=R/'configs/ig_baseline_eval_v3.json';c=json.loads(cp.read_text());root=Path(c['output_root'])
 with (root/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  frozen=json.loads((root/'protocol_lock.json').read_text())
  if protocol(c,cp)!=frozen:raise RuntimeError('Frozen protocol changed')
  mid=model_identity(model,c);dest=root/'evaluation'/digest(mid)[:16]/'primary'
  for n,v in [('model.json',mid),('protocol.json',frozen),('config.json',c)]:
   f=dest/n
   if f.exists() and json.loads(f.read_text())!=v:raise RuntimeError('Metadata mismatch: '+str(f))
   write(f,v)
  taskmap={j['name']:j for j in jobs(c,False)}
  stop=Event()
  def execute(name,gpu):
   if stop.is_set():return
   j=taskmap[name];ident={'protocol':digest(frozen),'model':digest(mid),'job':j,'smoke':False};unit=dest/(name+'-seed42')
   if any(valid_done(p,ident) for p in unit.glob('attempt-*')):return
   for mount in [root,Path(c['hf_cache']),Path('/')]:
    if shutil.disk_usage(mount).free<c['reserve_bytes']+268435456:raise RuntimeError('Insufficient storage: '+str(mount))
   attempt=unit/('attempt-'+str(time.time_ns()));attempt.mkdir(parents=True);write(attempt/'job.json',ident)
   write(attempt/'execution.json',{'physical_gpu':gpu,'pid':os.getpid(),'scheduler':'language_first_stage_barrier','allocator':'expandable_segments:True'})
   env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),HF_HOME=c['hf_cache'],HF_HUB_CACHE=c['hf_cache'],HF_DATASETS_CACHE=str(Path(c['hf_cache'])/'datasets'),HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',PYTHONHASHSEED='42',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
   with (attempt/'run.log').open('w') as log:
    child=subprocess.Popen([c['evaluation_python'],str(R/'scripts/baseline_eval_worker.py'),'--config',str(cp),'--model',str(model),'--output',str(attempt)],env=env,stdout=log,stderr=subprocess.STDOUT,pass_fds=(lock.fileno(),))
    write(dest/f'language_lane_gpu{gpu}.json',{'pid':os.getpid(),'child_pid':child.pid,'task':name,'attempt':str(attempt)});rc=child.wait()
   if rc:write(attempt/'FAILED.json',{'returncode':rc});raise RuntimeError('Task failed: '+str(attempt))
   require_scores(json.loads((attempt/'scores.json').read_text()))
   write(attempt/'DONE.json',{'identity':ident,'artifacts':{str(f.relative_to(attempt)):sha(f) for f in attempt.rglob('*') if f.is_file()}})
  for stage in [c['aggregate']['en']+c['aggregate']['target'],['mmlu_en','gmmlu_ig']]:
   def lane(index,gpu):
    try:
     for name in stage[index::len(c['evaluation_gpus'])]:execute(name,gpu)
    except BaseException:stop.set();raise
   with ThreadPoolExecutor(max_workers=len(c['evaluation_gpus'])) as pool:
    futures=[pool.submit(lane,i,g) for i,g in enumerate(c['evaluation_gpus'])]
    for future in futures:future.result()
  write(dest/'LANGUAGE_EIGHT_DONE.json',{'protocol':digest(frozen),'model':digest(mid),'order':['core_six','mmlu_en+gmmlu_ig'],'time':time.time()})
if __name__=='__main__':main()
