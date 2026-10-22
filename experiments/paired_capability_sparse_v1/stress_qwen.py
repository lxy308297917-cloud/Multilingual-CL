"""One long-sequence update with full FP32 moments already resident on GPU."""
import argparse,json,os,random,subprocess,threading,time,traceback
from pathlib import Path
import torch,numpy as np
from core import SparseProjectedAdamW,calibration_context
from smoke_qwen import H,BASE,RECIPE,PREV,write,sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--audit',required=True);args=ap.parse_args()
 out=Path(args.output);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();physical=[];stop=threading.Event()
 def monitor():
  while not stop.is_set():
   try:physical.append(int(subprocess.check_output(['nvidia-smi','--id='+os.environ['CUDA_VISIBLE_DEVICES'],'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip()))
   except Exception:pass
   stop.wait(.5)
 thread=threading.Thread(target=monitor,daemon=True);thread.start()
 try:
  from transformers import AutoModelForCausalLM,AutoTokenizer
  if not os.environ.get('CUDA_VISIBLE_DEVICES','').startswith('GPU-') or torch.cuda.device_count()!=1:raise RuntimeError('One queue GPU UUID required')
  torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
  torch.manual_seed(42);random.seed(42);np.random.seed(42)
  audit=json.loads(Path(args.audit).read_text());indices=audit['checks']['longest_training_indices'];m=json.loads((RECIPE/'manifest.json').read_text());pool={}
  with open(m['source_file']) as f:
   for i,line in enumerate(f):
    if i in indices:pool[i]=json.loads(line)
  tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True);model=AutoModelForCausalLM.from_pretrained(BASE,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda()
  model.config.use_cache=False;model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
  named={n:p for n,p in model.named_parameters() if '.layers.' in n and p.ndim==2}
  for n,p in model.named_parameters():p.requires_grad_(n in named)
  opt=SparseProjectedAdamW(named,bf16_policy='reject');checkpoint=H/'models/checkpoint/paired_capability_sparse_20261021/technical_v1/resume.pt'
  state=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
  if state['next_step']!=2:raise RuntimeError('Checkpoint step identity')
  oldcfg=json.loads(checkpoint.with_name('CONFIG.json').read_text())
  if sha(checkpoint.with_name('CONFIG.json'))!=state['config_sha256']:raise RuntimeError('Checkpoint config identity')
  if sha(m['source_file'])!=m['source_sha256'] or oldcfg['base_weights_sha256']!=audit['files'][str(BASE/'model.safetensors')]['sha256']:raise RuntimeError('Training identity')
  with torch.no_grad():
   for n,p in model.named_parameters():p.copy_(state['model'][n])
  opt.load_state_dict(state['optimizer']);del state
  pair=next(x for x in json.loads(PREV.read_text()) if x['id']==oldcfg['guard_id']);guards=[]
  for lang in ('en','sw'):
   prefix=tok.apply_chat_template([{'role':'system','content':'Solve the problem. Give the final numeric answer.'},{'role':'user','content':pair['question_'+lang]}],tokenize=True,add_generation_prompt=True)
   answer=tok(pair['gold']+'<|im_end|>',add_special_tokens=False)['input_ids'];guards.append({'input_ids':prefix+answer,'labels':[-100]*len(prefix)+answer})
  write(out/'CONFIG.json',{'scope':'One additional technical stress update, NOT a matched-training score','full_moments_resident_before_forward':True,'checkpoint':str(checkpoint),'checkpoint_config_sha256':sha(checkpoint.with_name('CONFIG.json')),'source_indices':indices,'lengths':[len(pool[i]['input_ids']) for i in indices],'core_sha256':sha(Path(__file__).with_name('core.py')),'script_sha256':sha(__file__),'GPU_UUID':os.environ['CUDA_VISIBLE_DEVICES'],'max_new_updates':1,'lr':5e-5,'precision':'BF16 weights, FP32 moments, SDPA, deterministic'})
  def loss(row):
   ids=torch.tensor([row['input_ids']],device='cuda');label=torch.tensor([row['labels'][1:]],device='cuda');valid=label!=-100
   logits=model(input_ids=ids,use_cache=False).logits[:,:-1][valid].float()
   return torch.nn.functional.cross_entropy(logits,label[valid])
  torch.cuda.reset_peak_memory_stats();model.train();model.zero_grad(set_to_none=True);tic=time.monotonic();target_loss=0
  for i in indices:
   value=loss(pool[i]);target_loss+=value.item()/2;(value/2).backward()
  torch.nn.utils.clip_grad_norm_(list(named.values()),1.);torch.cuda.synchronize();target_seconds=time.monotonic()-tic
  gs=[];tic=time.monotonic()
  with calibration_context(model):
   for row in guards:
    value=loss(row);g=torch.autograd.grad(value,list(named.values()));gs.append({n:v.detach().float().cpu() for (n,p),v in zip(named.items(),g)});del g,value
  torch.cuda.synchronize();calibration_seconds=time.monotonic()-tic;tic=time.monotonic();stats=opt.step(5e-5,gs);del gs;torch.cuda.synchronize();optimizer_seconds=time.monotonic()-tic
  write(out/'DONE.json',{'scope':'Technical stress only; no efficacy conclusion','full_moment_long_sequence_pass':True,'lengths':[len(pool[i]['input_ids']) for i in indices],'target_loss':target_loss,'stats':stats,'seconds':{'target':target_seconds,'calibration':calibration_seconds,'optimizer':optimizer_seconds,'wall':time.monotonic()-start},'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'physical_device_peak_mib':max(physical) if physical else None,'counts':{'input':sum(len(pool[i]['input_ids']) for i in indices),'supervised':sum(sum(v!=-100 for v in pool[i]['labels'][1:]) for i in indices),'calibration_input':sum(len(x['input_ids']) for x in guards),'calibration_supervised':sum(sum(v!=-100 for v in x['labels'][1:]) for x in guards),'generation':0}})
 except BaseException as e:
  write(out/'FAILED.json',{'error':str(e),'traceback':traceback.format_exc(),'wall_seconds':time.monotonic()-start});raise
 finally:stop.set();thread.join(timeout=3)
if __name__=='__main__':main()
