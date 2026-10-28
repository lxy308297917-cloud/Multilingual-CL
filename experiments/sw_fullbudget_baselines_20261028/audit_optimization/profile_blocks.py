"""Short shared-card CUDA identity and attributed timings, not final throughput.

Real Qwen matrix shapes and saved MoFO state; synthetic fixed gradients. Timed
sections synchronize CUDA correctly. Other own HFT work may interleave; final
switch decision requires an isolated full-Qwen fixed-batch comparison later.
"""
from pathlib import Path
import os,sys,argparse,json,time,math,copy,gc,subprocess
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT/'outputs/sw_fullbudget_baselines_20261028/code'))
from settings import H,BASE,read,write,sha
REPORT=ROOT/'outputs/sw_fullbudget_baselines_20261028/audit_optimization_20261028'
RUN=ROOT/'work/sw_fullbudget_baselines_20261028/audit_optimization_20261028'
ap=argparse.ArgumentParser();ap.add_argument('--uuid',required=True);ap.add_argument('--sharing-with-pid',type=int,required=True);a=ap.parse_args()
sys.path.insert(0,str(H/'gpu-queue'));import gpu_queue as q
snap=q.snapshot();gpu=snap[a.uuid];assert a.sharing_with_pid in gpu['pids']
assert all(os.stat('/proc/'+str(pid)).st_uid==os.getuid() for pid in gpu['pids'])
assert 'HFT15' in Path(f'/proc/{a.sharing_with_pid}/cmdline').read_text()
assert len(q.user_gpu_assignments(snap,q.other_queue_gpus()))<=4
free=int(subprocess.check_output(['nvidia-smi','--id='+a.uuid,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>8000
os.environ['CUDA_VISIBLE_DEVICES']=a.uuid;os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch,numpy as np
from safetensors import safe_open
from mofo_deferred import DeferredMoFO,BaselineAdamW
torch.cuda.set_per_process_memory_fraction(.1);torch.set_num_threads(4);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
write(RUN/'MOFO_BLOCK_PROFILE_LAUNCH.json',dict(pid=os.getpid(),uuid=a.uuid,shared_own_HFT_PID=a.sharing_with_pid,
 started_unix=time.time(),GPU_memory_fraction_limit=.1,isolated_throughput=False))

def timed(out,key,fn):
 torch.cuda.synchronize();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
 t=time.perf_counter();start.record();value=fn();end.record();torch.cuda.synchronize()
 out[key]=dict(wall_seconds=time.perf_counter()-t,cuda_event_interval_ms=start.elapsed_time(end))
 return value

def phased(opt,lr):
 name,p=next(iter(opt.named.items()));m,v=opt.state[name]['m'],opt.state[name]['v'];t=opt.steps+1;out={}
 timed(out,'finite_grad_check',lambda:bool(torch.isfinite(p.grad).all()))
 g=p.grad.float()
 def momentum():m.mul_(.9).add_(g,alpha=.1);v.mul_(.999).addcmul_(g,g,value=.001)
 timed(out,'FP32_full_moments',momentum)
 delta=timed(out,'full_Adam_div_sqrt',lambda:-lr*(m/(1-.9**t))/((v/(1-.999**t)).sqrt()+1e-8))
 score=m.abs();k=math.floor(p.numel()*.15)
 threshold=timed(out,'exact_topk_threshold',lambda:torch.topk(score.flatten(),k,sorted=False).values.min())
 idx=timed(out,'nonzero_indices',lambda:torch.where(score.flatten()>=threshold)[0])
 def union():opt.unions[name][idx.cpu()]=True;return int(opt.unions[name].sum())
 timed(out,'D2H_indices_and_CPU_union',union)
 old_full=p.detach().float().clone()
 def writeback():
  p.mul_(1-lr*.01);old=p.flatten()[idx];new=(old.float()+delta.flatten()[idx]).to(p.dtype);p.flatten().index_copy_(0,idx,new);return new
 with torch.no_grad():new=timed(out,'BF16_full_decay_and_sparse_write',writeback)
 def statistics():
  actual=p.float()-old_full
  return dict(a2=float(actual.square().sum()),nz=int(torch.count_nonzero(actual)),
   d2=float((delta.flatten()[idx]-lr*.01*old_full.flatten()[idx]).square().sum()),finite=bool(torch.isfinite(new).all()))
 timed(out,'actual_displacement_and_sync_statistics',statistics)
 out['index_D2H_bytes']=idx.numel()*idx.element_size();out['CPU_union_bytes']=opt.unions[name].numel()
 return out

checkpoint=H/'models/checkpoint/sw_fullbudget_baselines_20261028/MoFO_first/checkpoint_002000'
state=torch.load(checkpoint/'state.pt',map_location='cpu',mmap=True,weights_only=False)['optimizer']
names=['model.layers.0.self_attn.q_proj.weight','model.layers.0.self_attn.k_proj.weight','model.layers.0.mlp.gate_proj.weight','model.layers.0.mlp.down_proj.weight']
records=[]
with safe_open(str(checkpoint/'model/model.safetensors'),framework='pt',device='cpu') as model:
 for name in names:
  weight=model.get_tensor(name).clone();gradient=torch.randn(weight.shape,dtype=weight.dtype)*.001
  def setup(cls):
   p=torch.nn.Parameter(weight.cuda());p.grad=gradient.cuda();opt=cls({name:p},'MoFO')
   small=dict(state,names=[name],state={name:state['state'][name]},indices={name:state['indices'][name]},unions={name:state['unions'][name]})
   opt.load_state_dict(small);return p,opt
  p,old=setup(BaselineAdamW);w,new=setup(DeferredMoFO)
  oldstats=old.step(5e-5);newstats=new.step(5e-5)
  assert torch.equal(p,w),(name,'weight mismatch')
  assert oldstats==newstats,(name,'statistics mismatch',oldstats,newstats)
  assert torch.equal(old.indices[name],new.indices[name]),name+' mask mismatch'
  for key in ('m','v'):assert torch.equal(old.state[name][key],new.state[name][key]),name+' moment mismatch'
  assert torch.equal(old.unions[name],new.unions[name].cpu())
  del p,w,old,new;gc.collect();torch.cuda.empty_cache()
  timings={}
  for label,cls in [('frozen',BaselineAdamW),('deferred',DeferredMoFO)]:
   samples=[]
   for repeat in range(3):
    p,opt=setup(cls);torch.cuda.synchronize();start=time.perf_counter();opt.step(5e-5);torch.cuda.synchronize();samples.append(time.perf_counter()-start)
    del opt,p;gc.collect();torch.cuda.empty_cache()
   timings[label]=samples
  p,opt=setup(BaselineAdamW)
  with torch.no_grad():attribution=phased(opt,5e-5)
  records.append(dict(name=name,shape=list(weight.shape),CUDA_same_gradient_weights_moments_masks_and_stats_exact=True,
   timings=timings,attribution=attribution))
  print(name,'CUDA identity passed',timings,flush=True)
  del p,opt,weight,gradient;gc.collect();torch.cuda.empty_cache()
# Adversarial equal/zero threshold, zero LR and step19/20 boundary, plus restore.
name='model.layers.0.mlp.up_proj.weight';p=torch.nn.Parameter(torch.zeros(23,37,device='cuda',dtype=torch.bfloat16));w=torch.nn.Parameter(p.detach().clone())
old=BaselineAdamW({name:p},'MoFO');new=DeferredMoFO({name:w})
for i in range(40):
 g=torch.ones_like(p) if i in (0,20) else torch.zeros_like(p) if i==19 else torch.randn_like(p)
 p.grad=g;w.grad=g.clone();lr=0. if i in (0,19,20) else 5e-5
 assert old.step(lr)==new.step(lr);assert torch.equal(p,w)
 assert torch.equal(old.indices[name],new.indices[name])
 for key in ('m','v'):assert torch.equal(old.state[name][key],new.state[name][key])
 if i==21:
  path=RUN/'MOFO_SMALL_CUDA_STATE.pt';torch.save(new.state_dict(),path);restored=DeferredMoFO({name:w});restored.load_state_dict(torch.load(path,map_location='cpu',weights_only=False));new=restored
write(REPORT/'MOFO_CUDA_BLOCK_PROFILE.json',dict(passed=True,records=records,small_CUDA_updates=40,CUDA_state_roundtrip=True,
 source_checkpoint=str(checkpoint),gradients='fixed synthetic BF16 on actual Qwen matrix shapes and actual optimizer state',
 shared_with_own_training=True,not_isolated_throughput=True,real_Qwen_fixed_batch_and_full_checkpoint_acceptance_pending=True,
 timing_note='Synchronized section wall times include host work. CUDA event intervals may include stream idle/other-context interference; not pure kernel sums. Final speedup decision requires isolated end-to-end measurement.',
 peak_allocated_bytes=torch.cuda.max_memory_allocated(),finished_unix=time.time()))
print('CUDA block profiling and identity complete; no formal training switched.',flush=True)
