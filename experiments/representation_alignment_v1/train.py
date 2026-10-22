"""Dense original-weight SFT, common main prefix, fixed auxiliary schedule."""
import argparse,sys,traceback,subprocess
from common import *
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM,AutoTokenizer,get_cosine_schedule_with_warmup
import importlib.util
_spec=importlib.util.spec_from_file_location('original_recipe',HOME/'reproductions/sw_recipe_fft_20261010/train.py');_recipe=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_recipe);IndexedRows=_recipe.IndexedRows

def rng():return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}
def checkpoint(path,model,opt,sched,step,counts):
 path.mkdir(exist_ok=False); model.save_pretrained(path/'model',safe_serialization=True)
 state={'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'rng':rng(),'step':step,'counts':counts,'aux_cursor':step//4*8,'protocol_sha256':sha(ROOT/'protocol.json')};torch.save(state,path/'state.pt');del state
 # Real serialized state readback without duplicate optimizer moment allocation on GPU.
 restored=torch.load(path/'state.pt',map_location='cpu',weights_only=False,mmap=True);assert restored['step']==step and restored['scheduler']['last_epoch']==step
 for s in restored['optimizer']['state'].values():assert s['exp_avg'].dtype==torch.bfloat16 and s['exp_avg_sq'].dtype==torch.bfloat16
 write(path/'RECOVERABLE.json',{'step':step,'files':{str(p.relative_to(path)):sha(p) for p in path.rglob('*') if p.is_file()}})

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--method',choices=list('ABCD'),required=True);ap.add_argument('--output',required=True);ap.add_argument('--smoke',action='store_true');ap.add_argument('--resume');ap.add_argument('--smoke-steps',type=int,default=4);ap.add_argument('--smoke-no-checkpoint',action='store_true');a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);t=time.monotonic();counts={'main_input':0,'main_supervised':0,'aux_input':0,'aux_supervised':0,'aux_pairs':0,'aux_padded_processed':0};step=0
 try:
  torch.set_num_threads(4);seed();assert os.environ.get('CUDA_VISIBLE_DEVICES','').startswith('GPU-') and torch.cuda.device_count()==1
  protocol=json.loads((ROOT/'protocol.json').read_text());m=json.loads((ROOT/'data/main_manifest.json').read_text());indices=json.loads((ROOT/'data/main_prefix.json').read_text());assert sha(m['source'])==m['source_sha256'];ds=IndexedRows(m['source'],indices);auxrows=rows(ROOT/'data/train.jsonl');bank=None
  tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True);model=AutoModelForCausalLM.from_pretrained(BASE,torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda();model.config.use_cache=False;model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.train()
  opt=torch.optim.AdamW(model.parameters(),lr=5e-5,betas=(.9,.999),eps=1e-8,weight_decay=.01);sched=get_cosine_schedule_with_warmup(opt,2124,42465)
  layer=None
  if a.method in 'CD':
   teacher=torch.load(OUT/'EN_TEACHER.pt',map_location='cpu',weights_only=False);assert teacher['train_sha256']==sha(ROOT/'data/train.jsonl') and teacher['model_sha256']==sha(BASE/'model.safetensors') and teacher['tokenizer_sha256']==sha(BASE/'tokenizer.json');bank=teacher['vectors'].cuda();layer=teacher['layer'];del teacher
  if a.resume:
   cp=Path(a.resume);receipt=json.loads((cp/'RECOVERABLE.json').read_text())
   for rel,expected in receipt['files'].items():assert sha(cp/rel)==expected
   state=torch.load(cp/'state.pt',map_location='cpu',weights_only=False,mmap=True);assert state['protocol_sha256']==sha(ROOT/'protocol.json');from safetensors.torch import load_file
   loaded=model.load_state_dict(load_file(str(cp/'model/model.safetensors'),device='cuda'),strict=False);assert not loaded.unexpected_keys and loaded.missing_keys==(['lm_head.weight'] if model.config.tie_word_embeddings else []);opt.load_state_dict(state['optimizer']);sched.load_state_dict(state['scheduler']);step=state['step'];counts=state['counts'];random.setstate(state['rng']['python']);np.random.set_state(state['rng']['numpy']);torch.set_rng_state(state['rng']['torch']);torch.cuda.set_rng_state_all(state['rng']['cuda']);del state
  writer=SummaryWriter(str(HOME/'logs/representation_alignment_20261021/tensorboard'/out.relative_to(OUT)));write(out/'CONFIG.json',{'method':a.method,'smoke':a.smoke,'protocol':protocol,'main':m,'release_sha256':sha(ROOT/'RELEASE.json'),'git_commit':json.loads((ROOT/'RELEASE.json').read_text())['git_commit'],'code_sha256':sha(__file__),'common_sha256':sha(ROOT/'common.py'),'model_sha256':sha(BASE/'model.safetensors'),'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES'],'resume':a.resume,'started_unix':time.time(),'teacher_sha256':sha(OUT/'EN_TEACHER.pt') if a.method in 'CD' else None,'data_manifest_sha256':sha(ROOT/'data/manifest.json'),'optimizer_moments':'BF16 to match previous dense recipe; all 338 original tensors trainable'})
  assert not a.smoke_no_checkpoint or (a.smoke and a.method!='A');total=a.smoke_steps if a.smoke else m['steps'];max_seconds=protocol['per_train_gpu_hours']*3600;durations=[];log=(out/'STEPS.jsonl').open('w');start_step=step
  # Smoke uses the two actual longest prefix examples twice, including auxiliary step4.
  smokeidx=sorted(range(len(ds)),key=lambda i:len(ds[i]['input_ids']),reverse=True)[:2] if a.smoke else None
  while step<total:
   if time.monotonic()-t>max_seconds:raise RuntimeError('Preregistered training GPU time ceiling')
   if step%20==0:check_compute()
   st=time.monotonic();opt.zero_grad(set_to_none=True);mainvalue=0.;auxvalue=0.
   for j in range(2):
    row=ds[smokeidx[j] if a.smoke else step*2+j];x=pack([row],tok.pad_token_id,labels=True);labels=x.pop('labels');loss=ce(model(**x,use_cache=False).logits,labels);(loss/2).backward();mainvalue+=float(loss.detach())/2;counts['main_input']+=len(row['input_ids']);counts['main_supervised']+=sum(v!=-100 for v in row['labels'][1:]);del loss,x,labels
   if a.method!='A' and (step+1)%4==0:
    # Entire pool deterministically permuted once; same 8 pair IDs on B/C/D at a given step.
    order=sorted(range(len(auxrows)),key=lambda i:hashlib.sha256(('aux42:'+auxrows[i]['id']).encode()).hexdigest());ids=[order[(((step+1)//4-1)*8+j)%len(order)] for j in range(8)]
    if a.method=='B':
     ce_rows=[row for idx in ids for row in auxrows[idx]['ce_rows']]
     for begin in range(0,len(ce_rows),4):
      chunk=ce_rows[begin:begin+4];x=pack(chunk,tok.pad_token_id,labels=True);labels=x.pop('labels');loss=ce(model(**x,use_cache=False).logits,labels);weight=len(chunk)/len(ce_rows);(loss*.1*weight).backward();auxvalue+=float(loss.detach())*weight;counts['aux_input']+=sum(len(row['input_ids']) for row in chunk);counts['aux_supervised']+=sum(sum(z!=-100 for z in row['labels'][1:]) for row in chunk);counts['aux_padded_processed']+=x['input_ids'].numel();del loss,x,labels
    else:
     for begin in range(0,len(ids),4):
      ii=ids[begin:begin+4];x=pack([auxrows[i]['sw_ids'] for i in ii],tok.pad_token_id);hs=model.model(**x,output_hidden_states=True,use_cache=False).hidden_states;v=pool(hs[layer],x['attention_mask']);loss=auxiliary(v,bank,torch.tensor(ii,device='cuda'),a.method);weight=len(ii)/len(ids);(loss*.1*weight).backward();auxvalue+=float(loss.detach())*weight;counts['aux_input']+=sum(len(auxrows[i]['sw_ids']) for i in ii);counts['aux_padded_processed']+=x['input_ids'].numel();del loss,x,hs,v
    counts['aux_pairs']+=len(ids)
   norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.));assert np.isfinite(norm);lr=opt.param_groups[0]['lr'];watched=model.model.layers[0].self_attn.q_proj.weight;before=watched.detach().clone() if a.smoke else None;opt.step();sched.step();step+=1;torch.cuda.synchronize();durations.append(time.monotonic()-st)
   record={'step':step,'lr':lr,'main_ce':mainvalue,'aux_loss':auxvalue,'grad_norm':norm,'seconds':durations[-1],'counts':dict(counts),'allocated':torch.cuda.max_memory_allocated(),'reserved':torch.cuda.max_memory_reserved()}
   if a.smoke:
    delta=watched.detach().float()-before.float();record['technical_q_proj_BF16_nonzero']=int(torch.count_nonzero(delta));record['technical_q_proj_actual_displacement_L2']=float(delta.norm());del before,delta
   if step%20==0 or step==total:record['physical_memory_mib']=int(subprocess.check_output(['nvidia-smi','--id='+os.environ['CUDA_VISIBLE_DEVICES'],'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
   log.write(json.dumps(record)+'\n');log.flush()
   if step%20==0 or step==total:
    for k in ('main_ce','aux_loss','grad_norm','lr'):writer.add_scalar(k,record[k],step)
    record['elapsed_seconds']=time.monotonic()-t;write(out/'PROGRESS.json',record);print(json.dumps(record),flush=True)
   if step==32 and not a.smoke:
    gate=throughput_receipt(durations,m['steps'],max_seconds);write(out/'THROUGHPUT_GATE.json',gate)
    if not gate['passed']:raise RuntimeError('Throughput does not support frozen budget')
   if not a.smoke_no_checkpoint and (step==total or (not a.smoke and step==m['steps']//2)):checkpoint(out/('checkpoint_%06d'%step),model,opt,sched,step,counts)
  if a.smoke_no_checkpoint:
   from safetensors import safe_open
   ref=torch.load(TECH/'A/checkpoint_000004/state.pt',map_location='cpu',mmap=True,weights_only=False);current=opt.state_dict();md=0;wd=0
   for k,sa in ref['optimizer']['state'].items():md+=int(torch.count_nonzero(sa['exp_avg']!=current['state'][k]['exp_avg'].cpu()))
   state=model.state_dict()
   with safe_open(str(TECH/'A/checkpoint_000004/model/model.safetensors'),framework='pt',device='cpu') as refmodel:
    for key in refmodel.keys():wd+=int(torch.count_nonzero(refmodel.get_tensor(key)!=state[key].cpu()))
   write(out/'AUXILIARY_ACTUAL_EFFECT.json',{'moment_coordinates_different_from_A':md,'actual_BF16_weight_coordinates_different_from_A':wd,'same_main_LR_prefix':True,'not_downstream_efficacy':True});assert md>0,'Auxiliary loss completely rounded away from BF16 moment update'
  else:tok.save_pretrained(out/('checkpoint_%06d'%step)/'model')
  writer.close();write(out/'TRAINING_DONE.json',{'method':a.method,'steps':step,'counts':counts,'elapsed_seconds':time.monotonic()-t,'final_model':None if a.smoke_no_checkpoint else str(out/('checkpoint_%06d'%step)/'model'),'protocol_sha256':sha(ROOT/'protocol.json'),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
 except BaseException as e:write(out/'FAILED.json',{'error':str(e),'traceback':traceback.format_exc(),'step':step,'counts':counts,'elapsed_seconds':time.monotonic()-t});raise
if __name__=='__main__':main()
