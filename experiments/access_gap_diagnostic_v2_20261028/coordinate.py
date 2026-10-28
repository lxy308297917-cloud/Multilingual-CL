"""Manual one-shot v2 evaluation queue. No training, time cap, autostart or retry."""
from settings import *
import sys,subprocess,fcntl,traceback
sys.path.insert(0,str(H/'gpu-queue'));import gpu_queue as q
sys.path.insert(0,str(ROOT/'outputs/benchmark_direction_20261026'));from resource_guard import all_queue_reservations
pending=[(m,True) for m in MODELS]+[(m,False) for m in MODELS];live={};done=[];failures=[];attempts={}
with (RUN/'coordinator.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  while pending or live:
   for label,info in list(live.items()):
    rc=info['process'].poll()
    if rc is None:continue
    info['stream'].close();del live[label]
    if rc==75:pending.append(info['job'])
    elif rc:failures.append(dict(task=label,exit_code=rc))
    else:done.append(label)
   if not failures:
    for model,smoke in list(pending):
     target=RUN/('smoke' if smoke else 'MCF')/model/'DONE.json'
     if target.exists():done.append(model+('_smoke' if smoke else '_formal'));pending.remove((model,smoke));continue
     if not smoke and not (RUN/'smoke'/model/'DONE.json').exists():continue
     snap=q.snapshot();reserved=all_queue_reservations(q);assigned=q.user_gpu_assignments(snap,reserved);occupied={i['uuid'] for i in live.values()}
     if len(assigned|occupied)>=4:break
     idle=[u for u,g in snap.items() if u not in reserved|occupied and q.idle(g,dict(max_idle_memory_mib=512,max_idle_util_percent=5))]
     if not idle:break
     uuid=idle[0];label=model+('_smoke' if smoke else '_formal');attempts[label]=attempts.get(label,0)+1
     logfile=RUN/(label+f'_attempt{attempts[label]}.log');stream=logfile.open('x')
     cmd=[str(PYTHON),str(REPORT/'code/worker.py'),'--uuid',uuid,'--model',model]+(['--smoke'] if smoke else [])
     env=dict(os.environ,HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
     proc=subprocess.Popen(cmd,env=env,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
     live[label]=dict(process=proc,stream=stream,uuid=uuid,job=(model,smoke));pending.remove((model,smoke))
     write(RUN/(label+f'_LAUNCH{attempts[label]}.json'),dict(pid=proc.pid,uuid=uuid,command=cmd,log=str(logfile),launched_unix=time.time()))
   write(RUN/'QUEUE_STATE.json',dict(pending=pending,running={k:dict(pid=v['process'].pid,uuid=v['uuid']) for k,v in live.items()},done=done,failures=failures,max_account_GPUs=4,time_stop_limit=None,updated_unix=time.time()))
   if failures and not live:raise RuntimeError(str(failures))
   if pending or live:time.sleep(30)
  subprocess.run([str(PYTHON),str(REPORT/'code/analyze.py')],cwd=ROOT,check=True)
  write(RUN/'ALL_DONE.json',dict(passed=True,done=done,finished_unix=time.time()))
 except BaseException as e:
  write(RUN/'QUEUE_FAILED.json',dict(error=str(e),traceback=traceback.format_exc(),finished_unix=time.time()));raise
