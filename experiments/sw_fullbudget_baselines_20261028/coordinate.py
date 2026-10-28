"""One manual full-budget suite, sharing global account reservations with v2."""
from settings import *
import sys,subprocess,fcntl,traceback
sys.path.insert(0,str(H/'gpu-queue'));import gpu_queue as q
sys.path.insert(0,str(ROOT/'outputs/benchmark_direction_20261026'));from resource_guard import all_queue_reservations
TASKS=('belebele_en','belebele_sw','knowledge_en','knowledge_sw','translation','ifeval','gsm8k')
jobs=[dict(kind='preflight',method=m) for m in METHODS]+[dict(kind='smoke',method='Base',task=t) for t in TASKS]
jobs += [dict(kind='train',method=m) for m in METHODS]
jobs += [dict(kind='eval',method=m,task=t) for m in ('Base',*METHODS) for t in TASKS]
live={};done=[];failed=[];attempts={}
def label(j):return '_'.join(j[k] for k in ('kind','method','task') if k in j)
def destination(j):
 if j['kind']=='preflight':return REPORT/f'PREFLIGHT_{j["method"]}.json'
 if j['kind']=='train':return H/'models/checkpoint/sw_fullbudget_baselines_20261028'/f'{j["method"]}_first'/'TRAINING_DONE.json'
 return evaluation_output(j['method'],'smoke' if j['kind']=='smoke' else 'formal',j['task'])/'DONE.json'
def technical_gate():return all((REPORT/f'PREFLIGHT_{m}.json').exists() for m in METHODS) and all((evaluation_output('Base','smoke',t)/'DONE.json').exists() for t in TASKS) and read(REPORT/'SCORER_ACCEPTANCE.json')['passed']
def ready(j):
 if j['kind']=='preflight':
  existing=H/'models/checkpoint/sw_fullbudget_baselines_20261028'/f'{j["method"]}_preflight_budgetv2'
  if existing.exists() and not destination(j).exists():return False # external explicit preflight still running; never duplicate.
  return True
 if j['kind']=='smoke':return True
 if j['kind']=='train':return technical_gate()
 if j['method']=='Base':return technical_gate()
 return technical_gate() and destination(dict(kind='train',method=j['method'])).exists()
with (RUN/'coordinator.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  while jobs or live:
   for name,info in list(live.items()):
    rc=info['process'].poll()
    if rc is None:continue
    info['stream'].close();del live[name]
    write(RUN/f'EXIT_{name}_attempt{attempts[name]}.json',dict(exit_code=rc,finished_unix=time.time()))
    if rc==75:jobs.append(info['job'])
    elif rc:failed.append(dict(job=info['job'],exit_code=rc))
    else:
     assert destination(info['job']).exists(),name+' missing completion receipt'
     done.append(name)
   if not failed:
    for job in list(jobs):
     name=label(job)
     if destination(job).exists():done.append(name);jobs.remove(job);continue
     if not ready(job):continue
     if job['kind'] in ('train','eval') and not (REPORT/'EXECUTION_FROZEN.json').exists():
      write(REPORT/'EXECUTION_FROZEN.json',dict(frozen_unix=time.time(),technical_gates_passed=True,
       training_config_sha256=sha(REPORT/'TRAINING_CONFIG_FROZEN.json'),evaluation_sha256=sha(REPORT/'EVALUATION_FROZEN.json'),
       common_budget_sha256=sha(REPORT/'COMMON_BUDGET_20261028.json'),code_hashes={p.name:sha(p) for p in (REPORT/'code').glob('*.py')},
       groups=list(METHODS),max_account_GPUs=4,GPU_hour_stop_limit=None,wall_clock_stop_limit=None))
     snap=q.snapshot();reserved=all_queue_reservations(q);assigned=q.user_gpu_assignments(snap,reserved);occupied={v['uuid'] for v in live.values()}
     if len(assigned|occupied)>=4:break
     idle=[u for u,g in snap.items() if u not in reserved|occupied and q.idle(g,dict(max_idle_memory_mib=512,max_idle_util_percent=5))]
     if not idle:break
     uuid=idle[0];attempts[name]=attempts.get(name,0)+1;logfile=RUN/(name+f'_attempt{attempts[name]}.log')
     while logfile.exists():attempts[name]+=1;logfile=RUN/(name+f'_attempt{attempts[name]}.log')
     stream=logfile.open('x')
     if job['kind'] in ('train','preflight'):
      cmd=[str(PYTHON),str(REPORT/'code/train.py'),'--uuid',uuid,'--method',job['method']]
      if job['kind']=='preflight':cmd+=['--preflight','--attempt','budgetv2']
     else:
      cmd=[str(EVALPY),str(REPORT/'code/eval_worker.py'),'--uuid',uuid,'--model',job['method'],'--task',job['task']]
      if job['kind']=='smoke':cmd+=['--smoke']
     legacy=H/'reproductions/sw_d2_20260921';cache=H/'data/sw-d2-eval-inputs/hf_cache'
     env=dict(os.environ,HF_HOME=str(cache),HF_HUB_CACHE=str(cache/'hub'),HF_DATASETS_CACHE=str(cache/'datasets'),
      HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',NLTK_DATA=str(H/'nltk_data'),
      PYTHONHASHSEED='42',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
      PYTHONPATH=':'.join(str(p) for p in [legacy/'eval_project/lighteval_latest/src',legacy/'harness_local',legacy/'lm_eval_metadata']))
     proc=subprocess.Popen(cmd,env=env,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
     live[name]=dict(process=proc,stream=stream,uuid=uuid,job=job);jobs.remove(job)
     write(RUN/f'LAUNCH_{name}_attempt{attempts[name]}.json',dict(pid=proc.pid,uuid=uuid,command=cmd,log=str(logfile),started_unix=time.time()))
   write(RUN/'QUEUE_STATE.json',dict(pending=jobs,running={k:dict(pid=v['process'].pid,uuid=v['uuid'],job=v['job']) for k,v in live.items()},
    done=done,failed=failed,updated_unix=time.time(),max_account_GPUs=4,GPU_hour_stop_limit=None,wall_clock_stop_limit=None))
   if failed and not live:raise RuntimeError(str(failed))
   if jobs or live:time.sleep(30)
  write(RUN/'ALL_JOBS_DONE.json',dict(finished_unix=time.time(),done=done,scientific_result_acceptance_pending=True))
 except BaseException as e:
  write(RUN/'QUEUE_FAILED.json',dict(error=str(e),traceback=traceback.format_exc(),failed_unix=time.time()));raise
