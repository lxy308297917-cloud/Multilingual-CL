"""One shared-card Base pass, preserving per-sample sufficient statistics."""
from pathlib import Path
import sys,os,json,time,hashlib,fcntl,argparse,subprocess
ROOT=Path(__file__).resolve().parents[4]
CODE=ROOT/'outputs/sw_fullbudget_baselines_20261028/code';sys.path.insert(0,str(CODE))
from settings import H,BASE,read,write,sha
REPORT=ROOT/'outputs/sw_fullbudget_baselines_20261028/audit_optimization_20261028'
RUN=ROOT/'work/sw_fullbudget_baselines_20261028/audit_optimization_20261028'
ap=argparse.ArgumentParser();ap.add_argument('--uuid',required=True);ap.add_argument('--sharing-with-pid',type=int,required=True);a=ap.parse_args()
sys.path.insert(0,str(H/'gpu-queue'));import gpu_queue as q
sys.path.insert(0,str(ROOT/'outputs/benchmark_direction_20261026'));from resource_guard import all_queue_reservations
assert not (RUN/'SSU_ACTIVATION_CACHE.pt').exists()
with (H/'gpu-queue/reservation.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 snap=q.snapshot();assigned=q.user_gpu_assignments(snap,all_queue_reservations(q));gpu=snap[a.uuid]
 assert len(assigned)<=4 and a.uuid in assigned and a.sharing_with_pid in gpu['pids']
 assert all(os.stat('/proc/'+str(p)).st_uid==os.getuid() for p in gpu['pids'])
 assert 'HFT15' in Path(f'/proc/{a.sharing_with_pid}/cmdline').read_text()
 memory=int(subprocess.check_output(['nvidia-smi','--id='+a.uuid,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
 assert memory>12000,'Need at least12GB free on own existing card for capped6GB diagnostic'
 os.environ['CUDA_VISIBLE_DEVICES']=a.uuid;os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
 import torch,numpy as np,random
 torch.cuda.init();torch.cuda.set_per_process_memory_fraction(.25)
 write(RUN/'SSU_COLLECT_LAUNCH.json',dict(pid=os.getpid(),uuid=a.uuid,sharing_with_PID=a.sharing_with_pid,
  assigned_GPUs=sorted(assigned),started_unix=time.time(),GPU_memory_fraction_limit=.25,
  shared_timing_not_used_for_throughput_claim=True))
torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
from transformers import AutoModelForCausalLM
started=time.monotonic();model=AutoModelForCausalLM.from_pretrained(BASE,torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda().eval()
model.config.use_cache=False
pool={r['id']:r for r in read(ROOT/'work/sw_fullbudget_baselines_20261028/CALIBRATION_POOL.json')}
ids=read(ROOT/'work/sw_fullbudget_baselines_20261028/STATIC_CALIBRATION_IDS.json');rows=[pool[i] for i in ids]
candidates={n:p for n,p in model.named_parameters() if '.layers.' in n and p.ndim==2};assert len(candidates)==196
stats={n:dict(sum_fp32=[],mean_fp64=[],mean_bf16=[],weight_norm_fp32=p.float().cpu().norm(dim=0),
 weight_norm_fp64=p.double().cpu().norm(dim=0),weight_norm_bf16=p.abs().norm(dim=0).cpu()) for n,p in candidates.items()}
modules=dict(model.named_modules());hooks=[]
for n in candidates:
 def hook(module,inputs,output,name=n):
  x=inputs[0].detach();assert x.dtype==torch.bfloat16 and x.shape[0]==1
  flat=x.reshape(-1,x.shape[-1]);s=stats[name]
  s['sum_fp32'].append(flat.float().square().sum(0).cpu())
  s['mean_fp64'].append(flat.double().square().mean(0).cpu())
  s['mean_bf16'].append(flat.square().mean(0).cpu())
 hooks.append(modules[n.rsplit('.weight',1)[0]].register_forward_hook(hook))
with torch.inference_mode():
 for i,row in enumerate(rows):
  token=torch.tensor([row['input_ids']],device='cuda')
  model(input_ids=token,attention_mask=torch.ones_like(token),use_cache=False)
  if (i+1)%16==0:print('Captured',i+1,'/128 samples',flush=True)
for h in hooks:h.remove()
for n,s in stats.items():
 for key in ('sum_fp32','mean_fp64','mean_bf16'):s[key]=torch.stack(s[key]);assert len(s[key])==128
payload=dict(model_sha256=sha(BASE/'model.safetensors'),calibration_ids=ids,lengths=[len(r['input_ids']) for r in rows],
 statistics=stats,collector_sha256=sha(Path(__file__)),original_selection_sha256=sha(H/'models/checkpoint/sw_fullbudget_baselines_20261028/SSU_first/selection.pt'))
temp=RUN/'SSU_ACTIVATION_CACHE.pt.partial';torch.save(payload,temp);temp.rename(RUN/'SSU_ACTIVATION_CACHE.pt')
write(REPORT/'SSU_COLLECTION.json',dict(passed=True,samples=128,matrices=196,total_tokens=sum(payload['lengths']),
 min_tokens=min(payload['lengths']),max_tokens=max(payload['lengths']),cache=str(RUN/'SSU_ACTIVATION_CACHE.pt'),
 cache_sha256=sha(RUN/'SSU_ACTIVATION_CACHE.pt'),wall_seconds=time.monotonic()-started,
 peak_allocated_bytes=torch.cuda.max_memory_allocated(),finished_unix=time.time(),
 one_forward_per_sample=True,old_selection_untouched=True,shared_own_GPU=True,no_training=True))
print('SSU sufficient-statistics cache complete.',flush=True)
