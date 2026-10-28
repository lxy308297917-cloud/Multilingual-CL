from settings import *
import torch,random,numpy as np,math,importlib.util,shutil
from transformers import AutoTokenizer,AutoModelForCausalLM
from optimizer import BaselineAdamW
from calibration import collect,calibration_mode

def rate(step):
 p=read(REPORT/'TRAINING_CONFIG_FROZEN.json');total=p['steps'];warm=p['warmup_steps'];peak=p['learning_rate']
 assert 0<=step<=total
 return peak*step/warm if step<warm else peak*.5*(1+math.cos(math.pi*(step-warm)/(total-warm)))
def data():
 d=read(REPORT/'DATA_FROZEN.json');assert sha(d['source'])==d['source_sha256'];assert sha(RUN/'INDICES.json')==d['indices_sha256']
 spec=importlib.util.spec_from_file_location('fullbudget_indexed',H/'reproductions/sw_recipe_fft_20261010/train.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 rows=mod.IndexedRows(d['source'],read(RUN/'INDICES.json'));pool={r['id']:r for r in read(RUN/'CALIBRATION_POOL.json')}
 return rows,pool,{e['step']:e['ids'] for e in read(RUN/'CALIBRATION_SCHEDULE.json')}
def setup(method,artifact):
 torch.set_num_threads(4);random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
 torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert os.environ['CUDA_VISIBLE_DEVICES'].startswith('GPU-') and torch.cuda.device_count()==1
 tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
 model=AutoModelForCausalLM.from_pretrained(BASE,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda()
 model.config.use_cache=False;model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
 all_named=dict(model.named_parameters());candidates={n:p for n,p in all_named.items() if '.layers.' in n and p.ndim==2}
 assert len(candidates)==196 and sum(p.numel() for p in candidates.values())==1310195712
 static=None;named=all_named if method=='Dense' else candidates
 budget=read(REPORT/'COMMON_BUDGET_20261028.json')
 if method=='HFT15':
  from selection import hft_masks
  chosen,metadata=hft_masks(candidates,.15,42,artifact/'selection.pt');write(artifact/'SELECTION.json',metadata)
  named={n:p for n,p in candidates.items() if chosen[n]}
  assert sum(p.numel() for p in named.values())==budget['common_selected_elements']
 elif method=='SSU':
  from selection import ssu_masks
  _,pool,_=data();cal=[pool[k] for k in read(RUN/'STATIC_CALIBRATION_IDS.json')]
  def collate(rows):
   assert len(rows)==1;r=rows[0];ids=torch.tensor([r['input_ids']]);return dict(input_ids=ids,attention_mask=torch.ones_like(ids))
  with calibration_mode(model):masks,metadata=ssu_masks(model,candidates,cal,collate,.15,artifact/'selection.pt',column_quota=budget['ssu_columns'])
  assert metadata['selected_elements']==budget['common_selected_elements']
  static={n:torch.where(m.flatten())[0] for n,m in masks.items()};del masks
  write(artifact/'SELECTION.json',metadata)
 for n,p in all_named.items():p.requires_grad_(n in named)
 model.train();opt=BaselineAdamW(named,method,static=static,quota=budget['element_quotas'] if method in ('Element','Protected') else None)
 write(artifact/'PARAMETER_SCOPE.json',dict(method=method,candidate_elements=sum(p.numel() for p in candidates.values()),
  trainable_elements=sum(p.numel() for p in named.values()),unique_model_elements=sum(p.numel() for p in all_named.values()),
  shapes={n:list(p.shape) for n,p in named.items()},optimizer_moments_dtype='FP32',weights_dtype='BF16'))
 return model,tok,named,opt
def ce(model,row):
 ids=torch.tensor([row['input_ids']],device='cuda');labels=torch.tensor(row['labels'][1:],device='cuda');keep=labels!=-100
 logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits
 return torch.nn.functional.cross_entropy(logits[0,:-1][keep].float(),labels[keep])
def update(model,named,opt,batch,cal=None):
 model.train();model.zero_grad(set_to_none=True);cost={};calstats=None
 if opt.method=='Protected' and opt.steps%200==0:
  assert cal and len(cal)==32
  grads,calstats=collect(model,named,cal,lambda r:ce(model,r));opt.calibrate(grads);del grads
  cost['calibration']=calstats['seconds']
 start=time.monotonic();lossvalue=0.
 for r in batch:
  loss=ce(model,r)/len(batch);lossvalue+=float(loss.detach());loss.backward();del loss
 torch.cuda.synchronize();cost['main']=time.monotonic()-start
 if opt.method=='SSU':
  # Shield selected-out coordinates before global clipping, as gradient hooks do.
  for n,p in named.items():
   flat=p.grad.flatten();idx=opt.indices[n];chosen=flat.index_select(0,idx).clone();flat.zero_();flat.index_copy_(0,idx,chosen)
 norm=float(torch.nn.utils.clip_grad_norm_(list(named.values()),1.));assert math.isfinite(norm) and math.isfinite(lossvalue)
 start=time.monotonic();stats=opt.step(rate(opt.steps));torch.cuda.synchronize();cost['optimizer']=time.monotonic()-start
 if opt.method in ('Element','Protected','HFT15','SSU'):
  assert stats['selected']==read(REPORT/'COMMON_BUDGET_20261028.json')['common_selected_elements']
 model.zero_grad(set_to_none=True)
 return dict(stats,main_CE=lossvalue,gradient_norm=norm,calibration=calstats,seconds_by_phase=cost)
def rng():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def restore_rng(s):random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch']);torch.cuda.set_rng_state_all(s['cuda'])
def checkpoint(path,model,opt,counts):
 path=Path(path);assert not path.exists();path.parent.mkdir(parents=True,exist_ok=True)
 assert shutil.disk_usage(path.parent).free>30*1024**3
 temp=path.with_name(path.name+'.partial');assert not temp.exists();temp.mkdir()
 model.save_pretrained(temp/'model',safe_serialization=True)
 torch.save(dict(optimizer=opt.state_dict(),rng=rng(),counts=counts,protocol_sha256=sha(REPORT/'TRAINING_CONFIG_FROZEN.json'),
  data_position=2*opt.steps,scheduler=dict(step=opt.steps,next_lr=rate(opt.steps))),temp/'state.pt')
 write(temp/'RECOVERABLE.json',dict(step=opt.steps,files={str(p.relative_to(temp)):sha(p) for p in temp.rglob('*') if p.is_file()}))
 temp.rename(path)
def load(path,model,opt):
 from safetensors.torch import load_file
 r=read(path/'RECOVERABLE.json')
 for f,h in r['files'].items():assert sha(path/f)==h
 s=torch.load(path/'state.pt',map_location='cpu',mmap=True,weights_only=False)
 assert s['protocol_sha256']==sha(REPORT/'TRAINING_CONFIG_FROZEN.json')
 result=model.load_state_dict(load_file(str(path/'model/model.safetensors')),strict=False)
 assert result.missing_keys==['lm_head.weight'] and not result.unexpected_keys
 opt.load_state_dict(s['optimizer']);restore_rng(s['rng'])
 assert s['scheduler']==dict(step=opt.steps,next_lr=rate(opt.steps)) and s['data_position']==2*opt.steps
 return s['counts']
