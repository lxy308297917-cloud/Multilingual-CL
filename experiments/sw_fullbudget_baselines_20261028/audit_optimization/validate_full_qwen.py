"""Wait for HFT to finish, then use its slot for isolated technical acceptance.

No existing training is stopped. At HFT step40000 only its dispatcher is briefly
paused, so this authorized technical task can claim the next free slot. The
dispatcher resumes as soon as this process has claimed that card. No fifth GPU,
no production optimizer change, no new scientific endpoint.
"""
from pathlib import Path
import os,sys,json,time,signal,subprocess,fcntl,threading,traceback,argparse,gc,hashlib
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT/'outputs/sw_fullbudget_baselines_20261028/code'))
from settings import H,BASE,REPORT as BASE_REPORT,RUN as BASE_RUN,PYTHON,read,write,sha
REPORT=BASE_REPORT/'audit_optimization_20261028';RUN=BASE_RUN/'audit_optimization_20261028'
ap=argparse.ArgumentParser();ap.add_argument('--uuid',required=True);ap.add_argument('--coordinator-pid',type=int,required=True);a=ap.parse_args()
sys.path.insert(0,str(H/'gpu-queue'));import gpu_queue as q
sys.path.insert(0,str(ROOT/'outputs/benchmark_direction_20261026'));from resource_guard import all_queue_reservations
assert read(REPORT/'MOFO_CPU_ACCEPTANCE.json')['passed'] and read(REPORT/'MOFO_CUDA_BLOCK_PROFILE.json')['passed']
assert not (REPORT/'MOFO_REAL_QWEN_ACCEPTANCE.json').exists()
paused=False;gpu_started=None;physical=[];stop=threading.Event()
dispatcher_start_ticks=Path(f'/proc/{a.coordinator_pid}/stat').read_text().rsplit(')',1)[1].split()[19]
def own_dispatcher():
 p=Path(f'/proc/{a.coordinator_pid}');assert p.stat().st_uid==os.getuid()
 argv=(p/'cmdline').read_bytes().decode().split('\0');script=Path(argv[1])
 if not script.is_absolute():script=(p/'cwd').resolve()/script
 assert script.resolve()==(BASE_REPORT/'code/coordinate.py').resolve()
 assert (p/'stat').read_text().rsplit(')',1)[1].split()[19]==dispatcher_start_ticks
def resume_dispatcher():
 global paused
 if paused:
  own_dispatcher();os.kill(a.coordinator_pid,signal.SIGCONT);paused=False
  write(RUN/'DISPATCHER_RESUMED.json',dict(pid=a.coordinator_pid,unix=time.time(),other_training_never_stopped=True))
try:
 own_dispatcher()
 hft=H/'models/checkpoint/sw_fullbudget_baselines_20261028/HFT15_first'
 while not (hft/'TRAINING_DONE.json').exists():
  if (BASE_RUN/'QUEUE_FAILED.json').exists():raise RuntimeError('Upstream queue failure: inspect before claiming slot')
  if (hft/'PROGRESS.json').exists() and read(hft/'PROGRESS.json')['step']>=40000 and not paused:
   own_dispatcher();os.kill(a.coordinator_pid,signal.SIGSTOP);paused=True
   write(RUN/'DISPATCHER_PAUSED_FOR_TECHNICAL_SLOT.json',dict(pid=a.coordinator_pid,unix=time.time(),
    reason='reserve next HFT-completed slot for authorized isolated optimizer validation',no_training_paused=True))
  time.sleep(2)
 # HFT's full completion receipt exists; no ongoing HFT training is interrupted.
 if not paused:
  own_dispatcher();os.kill(a.coordinator_pid,signal.SIGSTOP);paused=True
 with (H/'gpu-queue/reservation.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  while True:
   snap=q.snapshot();reserved=all_queue_reservations(q);assigned=q.user_gpu_assignments(snap,reserved)
   if a.uuid in snap and q.idle(snap[a.uuid],dict(max_idle_memory_mib=512,max_idle_util_percent=5)) and a.uuid not in reserved and len(assigned)<4:break
   # If another user took the card, release the dispatcher and report, never take over.
   foreign=[p for p in snap[a.uuid]['pids'] if os.stat('/proc/'+str(p)).st_uid!=os.getuid()]
   if foreign:raise RuntimeError('Another user claimed HFT card; isolated validation must wait for a new slot')
   time.sleep(1)
  os.environ['CUDA_VISIBLE_DEVICES']=a.uuid;os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
  import torch,numpy as np
  torch.cuda.init();probe=torch.empty(1,device='cuda');torch.cuda.synchronize();del probe
  gpu_started=time.time()
  write(RUN/'ISOLATED_GPU_CLAIMED.json',dict(pid=os.getpid(),uuid=a.uuid,started_unix=gpu_started,
   completed_HFT_receipt_sha256=sha(hft/'TRAINING_DONE.json'),account_GPUs_before=sorted(assigned)))
 resume_dispatcher()
 def physical_monitor():
  while not stop.wait(5):
   try:physical.append(int(subprocess.check_output(['nvidia-smi','--id='+a.uuid,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True,timeout=10).strip()))
   except Exception:pass
 threading.Thread(target=physical_monitor,daemon=True).start()
 torch.set_num_threads(4)
 import engine
 from optimizer import BaselineAdamW
 from mofo_deferred import DeferredMoFO
 candidates=sorted((H/'models/checkpoint/sw_fullbudget_baselines_20261028/MoFO_first').glob('checkpoint_*/RECOVERABLE.json'))
 assert candidates;source=candidates[-1].parent;receipt=read(source/'RECOVERABLE.json')
 for filename,digest in receipt['files'].items():assert sha(source/filename)==digest
 source_state=torch.load(source/'state.pt',map_location='cpu',mmap=True,weights_only=False)
 initial_step=source_state['optimizer']['steps'];del source_state
 rows,pool,events=engine.data();fixed=[[rows[2*(initial_step+i)],rows[2*(initial_step+i)+1]] for i in range(20)]
 wanted=set(read(BASE_REPORT/'DATA_FROZEN.json')['longest_source_indices']);longest={}
 with Path(read(BASE_REPORT/'DATA_FROZEN.json')['source']).open() as stream:
  for index,line in enumerate(stream):
   if index in wanted:longest[index]=json.loads(line)
 fixtures=fixed[:8];fixtures[2]=list(longest.values())
 artifact=H/'models/checkpoint/sw_fullbudget_baselines_20261028'/('mofo_engineering_acceptance_'+str(time.time_ns()))
 write(RUN/'ISOLATED_SOURCE_FROZEN.json',dict(source=str(source),source_receipt_sha256=sha(source/'RECOVERABLE.json'),
  initial_step=initial_step,fixture_lengths=[[len(r['input_ids']) for r in batch] for batch in fixtures],
  timed_batches=20,warmup_batches=2,technical_only=True,artifact=str(artifact),
  main_config_sha256=sha(BASE_REPORT/'TRAINING_CONFIG_FROZEN.json'),code_hashes={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}))
 def tensor_digest(named,opt,gradients=False):
  digest=hashlib.sha256()
  def feed(t):
   value=t.detach().cpu().contiguous().view(torch.uint8).numpy();digest.update(memoryview(value))
  for name,p in named.items():
   digest.update(name.encode())
   if gradients:feed(p.grad);continue
   feed(p)
   for key in ('m','v'):feed(opt.state[name][key])
   feed(opt.indices[name]);union=opt.unions[name].detach().cpu().numpy();digest.update(np.packbits(union,bitorder='little').tobytes())
  return digest.hexdigest()
 def numerical_stats(stats):return {k:v for k,v in stats.items() if k!='seconds_by_phase'}
 def rng_digest(state):
  h=hashlib.sha256();h.update(repr(state['python']).encode());h.update(state['numpy'][0].encode())
  h.update(state['numpy'][1].tobytes());h.update(repr(state['numpy'][2:]).encode())
  h.update(state['torch'].cpu().numpy().tobytes())
  for value in state['cuda']:h.update(value.cpu().numpy().tobytes())
  return h.hexdigest()
 def checked_load(path,model,opt):
  # Perturb every RNG first; no-dropout Qwen replay alone would not test restore.
  import random
  expected=torch.load(path/'state.pt',map_location='cpu',mmap=True,weights_only=False)['rng']
  random.random();np.random.rand();torch.rand(1);torch.rand(1,device='cuda')
  counts=engine.load(path,model,opt);assert rng_digest(engine.rng())==rng_digest(expected)
  return counts
 def one_pass(cls,label):
  engine.BaselineAdamW=cls;model,tok,named,opt=engine.setup('MoFO',artifact/label)
  counts=checked_load(source,model,opt);proof=[];gradients=[];original_step=opt.step
  torch.cuda.reset_peak_memory_stats()
  def checked_step(lr):gradients.append(tensor_digest(named,opt,gradients=True));return original_step(lr)
  opt.step=checked_step
  for i,batch in enumerate(fixtures):
   stats=engine.update(model,named,opt,batch)
   proof.append(dict(gradient_sha256=gradients[-1],state_sha256=tensor_digest(named,opt),rng_sha256=rng_digest(engine.rng()),stats=numerical_stats(stats)))
   print(label,'checked real update',i+1,'/8',flush=True)
   if i==3:engine.checkpoint(artifact/label/'checkpoint_fixture_4',model,opt,dict(technical_fixture_updates=4))
  fixture_peak=torch.cuda.max_memory_allocated()
  restored_counts=checked_load(artifact/label/'checkpoint_fixture_4',model,opt);assert restored_counts==dict(technical_fixture_updates=4)
  for i in range(4,8):
   stats=engine.update(model,named,opt,fixtures[i]);record=dict(gradient_sha256=gradients[-1],state_sha256=tensor_digest(named,opt),rng_sha256=rng_digest(engine.rng()),stats=numerical_stats(stats))
   assert record==proof[i],(label,'full checkpoint replay mismatch',i)
  # Timed work is separate from digest/serialization and uses the same real stream.
  opt.step=original_step;checked_load(source,model,opt);timings=[]
  for i,batch in enumerate(fixed):
   torch.cuda.synchronize();start=time.perf_counter();stats=engine.update(model,named,opt,batch);torch.cuda.synchronize()
   elapsed=time.perf_counter()-start
   if i==1:torch.cuda.reset_peak_memory_stats()
   if i>=2:timings.append(dict(wall_seconds=elapsed,**stats['seconds_by_phase']))
  value=dict(proof=proof,checkpoint_replay_exact=True,timed_updates=18,warmup_updates=2,timings=timings,
   fixture_peak_allocated_bytes=fixture_peak,all_RNG_states_perturbed_then_restored_and_verified=True,
   peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
  del model,tok,named,opt,original_step,checked_step;gc.collect();torch.cuda.empty_cache()
  return value
 frozen=one_pass(BaselineAdamW,'frozen');optimized=one_pass(DeferredMoFO,'deferred')
 assert frozen['proof']==optimized['proof'],'Real Qwen per-step gradients/weights/moments/masks/statistics differ'
 f=np.array([r['wall_seconds'] for r in frozen['timings']]);o=np.array([r['wall_seconds'] for r in optimized['timings']])
 ratio=float(f.sum()/o.sum())
 result=dict(passed=True,real_fixed_batch_bitwise=True,updates_checked=8,full_checkpoint_restore_and_replay=True,
  longest_batch_lengths=[len(r['input_ids']) for r in fixtures[2]],frozen=frozen,optimized=optimized,
  isolated_mean_step_speedup=ratio,isolated_median_step_speedup=float(np.median(f)/np.median(o)),
  source_checkpoint=str(source),source_step=initial_step,technical_only=True,no_formal_training_switched=True,
  eligible_for_checkpoint_handoff=ratio>1.1,finished_unix=time.time())
 write(REPORT/'MOFO_REAL_QWEN_ACCEPTANCE.json',result)
 print('Isolated full-Qwen acceptance passed; speedup',ratio,'; no formal switch yet.',flush=True)
except BaseException as error:
 write(REPORT/f'ISOLATED_VALIDATION_FAILED_{time.time_ns()}.json',dict(error=str(error),traceback=traceback.format_exc(),failed_unix=time.time()))
 raise
finally:
 resume_dispatcher();stop.set()
 if gpu_started is not None:
  write(RUN/'ISOLATED_GPU_END.json',dict(finished_unix=time.time(),wall_seconds=time.time()-gpu_started,
   observed_physical_peak_MiB=max(physical) if physical else None))
