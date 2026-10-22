import json,hashlib,random,os,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parent;HOME=Path('/data0/users/lixiaoyang');BASE=HOME/'models/Qwen2.5-1.5B-Instruct';OUT=HOME/'models/checkpoint/representation_alignment_20261021';TECH=OUT/'technical_deterministic_v2'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def write(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,p)
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def seed():
 torch.set_num_threads(4)
 torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 # Deterministic backward required by real serialized-resume parity; old full dense is diagnostic only.
def pack(items,pad,device='cuda',labels=False):
 ids=[r['input_ids'] if isinstance(r,dict) else r for r in items];w=max(map(len,ids));result={'input_ids':torch.tensor([x+[pad]*(w-len(x)) for x in ids],device=device),'attention_mask':torch.tensor([[1]*len(x)+[0]*(w-len(x)) for x in ids],device=device)}
 if labels:result['labels']=torch.tensor([r['labels']+[-100]*(w-len(r['labels'])) for r in items],device=device)
 return result
def pool(h,mask,method='mean'):
 mask=mask.float()
 if method=='last':return h[torch.arange(len(h),device=h.device),mask.sum(1).long()-1].float()
 if method=='weighted':mask=mask*torch.arange(1,h.shape[1]+1,device=h.device)
 return (h.float()*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]
def ce(logits,labels):
 l=labels[:,1:];v=F.cross_entropy(logits[:,:-1].float().transpose(1,2),l,ignore_index=-100,reduction='none');return (v.sum(1)/(l!=-100).sum(1)).mean()
def retrieval(en,sw):
 en=F.normalize(en.float(),dim=-1);sw=F.normalize(sw.float(),dim=-1);c=en@sw.T;n=len(c);eye=torch.eye(n,dtype=torch.bool,device=c.device);diag=c.diag();negative=c.masked_fill(eye,-float('inf'))
 return {'en_to_sw_p1':float((c.argmax(1)==torch.arange(n,device=c.device)).float().mean()),'sw_to_en_p1':float((c.argmax(0)==torch.arange(n,device=c.device)).float().mean()),'mexa_strict':float(((diag>negative.max(1).values)&(diag>negative.max(0).values)).float().mean()),'positive_cosine':float(diag.mean()),'deranged_cosine':float(c[torch.arange(n),torch.roll(torch.arange(n),1)].mean()),'off_diagonal_mean':float(c[~eye].mean()),'positive_negative_gap':float(diag.mean()-c[~eye].mean()),'en_variance':float(en.var(0).mean()),'sw_variance':float(sw.var(0).mean())}
def auxiliary(v,bank,idx,kind):
 v=F.normalize(v.float(),dim=-1);bank=F.normalize(bank.detach().float(),dim=-1)
 if kind=='C':return (1-(v*bank[idx]).sum(-1)).mean()
 if kind=='D':return F.cross_entropy(v@bank.T/.07,idx)
 raise ValueError(kind)

def check_compute():
 import subprocess
 pids=subprocess.check_output(['nvidia-smi','--id='+os.environ['CUDA_VISIBLE_DEVICES'],'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines()
 if any(p.strip() and int(p)!=os.getpid() for p in pids):raise RuntimeError('Another compute process appeared on assigned GPU; exiting safely')

def throughput_receipt(durations,steps,ceiling):
 estimate=float(np.mean(durations[1:])*steps)
 return {'projected_seconds':estimate,'ceiling_seconds':float(ceiling),'passed':bool(estimate<float(ceiling)*.9)}
