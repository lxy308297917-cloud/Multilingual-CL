"""Common sparse main loss, model range, scheduler and lossless checkpoint transport."""
from settings import *
import torch,numpy as np,random,math,shutil,threading,subprocess,importlib.util
from optimizer import BudgetSparseAdamW
from transformers import AutoModelForCausalLM,AutoTokenizer

def learning_rate(step,total=42465,warmup=2124,peak=5e-5):
 if not 0<=step<=total or not 0<warmup<total:raise ValueError('Invalid scheduler')
 return peak*step/warmup if step<warmup else peak*.5*(1+math.cos(math.pi*(step-warmup)/(total-warmup)))

def per_example_ce(model,rows,pad,device):
 width=max(len(r['input_ids']) for r in rows);ids=torch.tensor([r['input_ids']+[pad]*(width-len(r['input_ids'])) for r in rows],device=device);mask=torch.tensor([[1]*len(r['input_ids'])+[0]*(width-len(r['input_ids'])) for r in rows],device=device);logits=model(input_ids=ids,attention_mask=mask,use_cache=False).logits;losses=[]
 for i,r in enumerate(rows):
  target=torch.tensor(r['labels'][1:]+[-100]*(width-len(r['labels'])),device=device);valid=target!=-100;assert bool(valid.any());losses.append(torch.nn.functional.cross_entropy(logits[i,:-1][valid].float(),target[valid]))
 return torch.stack(losses)

def data():
 m=json.loads((OLD/'data/main_manifest.json').read_text());assert sha(m['source'])==m['source_sha256'];indices=json.loads((OLD/'data/main_prefix.json').read_text());assert sha(OLD/'data/main_prefix.json')==m['main_prefix_sha256'];spec=importlib.util.spec_from_file_location('module_sparse_recipe',H/'reproductions/sw_recipe_fft_20261010/train.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod.IndexedRows(m['source'],indices),m

def setup(group):
 seed();assert os.environ['CUDA_VISIBLE_DEVICES'].startswith('GPU-') and torch.cuda.device_count()==1;tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True);model=AutoModelForCausalLM.from_pretrained(BASE,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda();model.config.use_cache=False;model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.train();budget=json.loads((REPORT/'BUDGET_FROZEN.json').read_text());names=budget['shapes'];named={n:p for n,p in model.named_parameters() if n in names};assert len(named)==196 and sum(p.numel() for p in named.values())==1310195712
 for n,p in model.named_parameters():p.requires_grad_(n in named)
 opt=BudgetSparseAdamW(named,budget['groups'][group]['quota']);return model,tok,named,opt

def rng():return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}
def restore(r):random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch']);torch.cuda.set_rng_state_all(r['cuda'])
def checkpoint(path,model,opt,counts,group):
 estimated=sum(p.numel()*p.element_size() for p in model.parameters())+sum(v.numel()*v.element_size() for s in opt.state.values() for v in s.values())+sum(p.numel()/4 for p in opt.named.values())
 assert shutil.disk_usage(OUT).free>estimated+10*1024**3,'Insufficient disk for fullcheckpoint plus10GiB reserve';start=time.time();path.mkdir(parents=True,exist_ok=False);model.save_pretrained(path/'model',safe_serialization=True);state={'optimizer':opt.state_dict(),'rng':rng(),'step':opt.steps,'counts':dict(counts),'scheduler':{'step':opt.steps,'next_lr':learning_rate(opt.steps),'total':42465,'warmup':2124},'protocol_sha256':sha(REPORT/'PROTOCOL_FROZEN.json'),'group':group};torch.save(state,path/'state.pt');del state;receipt={'step':opt.steps,'files':{str(p.relative_to(path)):sha(p) for p in path.rglob('*') if p.is_file()},'save_seconds':time.time()-start};write(path/'RECOVERABLE.json',receipt);return path

def load_checkpoint(path,model,opt,group):
 from safetensors.torch import load_file
 receipt=json.loads((path/'RECOVERABLE.json').read_text())
 for rel,value in receipt['files'].items():assert sha(path/rel)==value
 state=torch.load(path/'state.pt',map_location='cpu',mmap=True,weights_only=False);assert state['group']==group and state['protocol_sha256']==sha(REPORT/'PROTOCOL_FROZEN.json');loaded=model.load_state_dict(load_file(str(path/'model/model.safetensors'),device='cpu'),strict=False);assert loaded.missing_keys==['lm_head.weight'] and not loaded.unexpected_keys;opt.load_state_dict(state['optimizer']);restore(state['rng']);assert state['step']==opt.steps==state['scheduler']['step'];assert state['scheduler']['next_lr']==learning_rate(opt.steps);return state['counts']

def update(model,tok,named,opt,batch):
 model.train();model.zero_grad(set_to_none=True);value=0.
 for row in batch:
  loss=per_example_ce(model,[row],tok.pad_token_id,'cuda')[0];(loss/2).backward();value+=float(loss.detach())/2;del loss
 norm=float(torch.nn.utils.clip_grad_norm_(list(named.values()),1.));assert math.isfinite(value) and math.isfinite(norm);stats=opt.step(learning_rate(opt.steps));torch.cuda.synchronize();stats.update(main_CE=value,full_candidate_preclip_norm=norm);return stats
