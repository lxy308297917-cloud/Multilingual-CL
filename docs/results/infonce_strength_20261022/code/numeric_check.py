"""Actual accumulation, clipping and copied AdamW/BF16 updates. Never alter old model/state."""
from settings import *
import torch,math,gc,importlib.util
from common import seed,pack,ce,pool,auxiliary,check_compute
from transformers import AutoTokenizer,AutoModelForCausalLM

def main():
 started=time.time();seed();alpha=json.loads((RUN/'CALIBRATION_DONE.json').read_text())['alpha'];batches=json.loads((RUN/'CALIBRATION_BATCHES_FROZEN.json').read_text())['batches'][:2];out=RUN/'numeric_check';out.mkdir(exist_ok=False)
 cp=TRAIN/'comparison_v3/D/checkpoint_002146';receipt=json.loads((cp/'RECOVERABLE.json').read_text());assert sha(cp/'state.pt')==receipt['files']['state.pt'];state=torch.load(cp/'state.pt',map_location='cpu',weights_only=False,mmap=True);assert state['step']==2146
 tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True);model=AutoModelForCausalLM.from_pretrained(cp/'model',torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda();model.config.use_cache=False;model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.train();params=list(model.parameters());rr=rows(OLD/'data/train.jsonl');bank=torch.load(TRAIN/'EN_TEACHER.pt',map_location='cpu',weights_only=False)['vectors'].cuda()
 spec=importlib.util.spec_from_file_location('recipe',H/'reproductions/sw_recipe_fft_20261010/train.py');recipe=importlib.util.module_from_spec(spec);spec.loader.exec_module(recipe);manifest=json.loads((OLD/'data/main_manifest.json').read_text());ds=recipe.IndexedRows(manifest['source'],json.loads((OLD/'data/main_prefix.json').read_text()))
 def collect():return [p.grad.cpu().clone() if p.grad is not None else None for p in params]
 def main_back(b):
  value=0.
  for pos in b['main_prefix_positions']:
   row=ds[pos];x=pack([row],tok.pad_token_id,labels=True);lab=x.pop('labels');loss=ce(model(**x,use_cache=False).logits,lab);(loss/2).backward();value+=float(loss.detach())/2;del x,lab,loss
  return value
 def aux_back(b,kind,scale):
  value=0.
  if kind=='B':
   cr=[row for i in b['aux_indices'] for row in rr[i]['ce_rows']]
   for begin in range(0,16,4):
    chunk=cr[begin:begin+4];x=pack(chunk,tok.pad_token_id,labels=True);lab=x.pop('labels');loss=ce(model(**x,use_cache=False).logits,lab);(loss*scale*(len(chunk)/16)).backward();value+=float(loss.detach())*len(chunk)/16;del x,lab,loss
  else:
   for begin in range(0,8,4):
    ii=b['aux_indices'][begin:begin+4];x=pack([rr[i]['sw_ids'] for i in ii],tok.pad_token_id);hs=model.model(**x,output_hidden_states=True,use_cache=False).hidden_states;v=pool(hs[10],x['attention_mask']);loss=auxiliary(v,bank,torch.tensor(ii,device='cuda'),'D');(loss*scale*(len(ii)/8)).backward();value+=float(loss.detach())*len(ii)/8;del x,hs,v,loss
  return value
 results=[]
 for bi,b in enumerate(batches):
  check_compute();assert time.time()-started<1800
  gradients={};stats={}
  for kind,scale in [('main',0.),('D0',.1),('D1',alpha),('B',.1)]:
   model.zero_grad(set_to_none=True);ml=main_back(b);al=0. if kind=='main' else aux_back(b,'B' if kind=='B' else 'D',scale)
   norm=float(torch.nn.utils.clip_grad_norm_(params,1.));assert math.isfinite(norm);gradients[kind]=collect();stats[kind]={'main_CE':ml,'raw_aux_loss':al,'scale':scale,'preclip_norm_BF16':norm,'clip_factor':min(1.,1/(norm+1e-6))};model.zero_grad(set_to_none=True)
  summaries={k:{'weight_changed_vs_main':0,'moment_changed_vs_main':0,'actual_delta_sq':0.,'difference_vs_main_sq':0.,'gradient_difference_vs_main_sq':0.,'gradient_main_dot':0.,'gradient_sq':0.,'finite':True} for k in ('D0','D1','B')};main_delta_sq=0.;main_grad_sq=0.;sg=state['optimizer']['param_groups'][0]
  with torch.no_grad():
   for pi,(p,paramid) in enumerate(zip(params,sg['params'])):
    st=state['optimizer']['state'][paramid];outs={};mom={};gg={}
    for kind in ('main','D0','D1','B'):
     q=torch.nn.Parameter(p.detach().clone());o=torch.optim.AdamW([q],lr=sg['lr'],betas=tuple(sg['betas']),eps=sg['eps'],weight_decay=sg['weight_decay'],foreach=True);o.state[q]={'step':st['step'].clone(),'exp_avg':st['exp_avg'].cuda().clone(),'exp_avg_sq':st['exp_avg_sq'].cuda().clone()};g=gradients[kind][pi];q.grad=g.cuda().clone() if g is not None else torch.zeros_like(q);gg[kind]=q.grad.float().clone();o.step();outs[kind]=q.detach().clone();mom[kind]=o.state[q]['exp_avg'].clone();del q,o
    main_delta_sq+=float((outs['main'].float()-p.float()).square().sum());main_grad_sq+=float(gg['main'].square().sum())
    for kind,s in summaries.items():
     s['weight_changed_vs_main']+=int(torch.count_nonzero(outs[kind]!=outs['main']));s['moment_changed_vs_main']+=int(torch.count_nonzero(mom[kind]!=mom['main']));s['actual_delta_sq']+=float((outs[kind].float()-p.float()).square().sum());s['difference_vs_main_sq']+=float((outs[kind].float()-outs['main'].float()).square().sum());s['gradient_difference_vs_main_sq']+=float((gg[kind]-gg['main']).square().sum());s['gradient_main_dot']+=float((gg[kind]*gg['main']).sum());s['gradient_sq']+=float(gg[kind].square().sum());s['finite']=s['finite'] and bool(torch.isfinite(outs[kind]).all())
    del outs,mom,gg
  for kind,s in summaries.items():s['relative_update_difference_vs_main']=math.sqrt(s['difference_vs_main_sq']/main_delta_sq);s['clipped_gradient_main_cosine']=s['gradient_main_dot']/math.sqrt(s['gradient_sq']*main_grad_sq)
  gain=math.sqrt(summaries['D1']['difference_vs_main_sq']/summaries['D0']['difference_vs_main_sq']);passed=gain>=2 and summaries['D1']['weight_changed_vs_main']>0 and all(s['finite'] for s in summaries.values());result={'batch':bi,'event':b['event'],'checkpoint_step':2146,'stats':stats,'candidate':summaries,'main_only_actual_delta_L2':math.sqrt(main_delta_sq),'D1_vs_D0_effect_L2_multiplier':gain,'passed':passed};results.append(result);write(out/'PARTIAL.json',results);print(json.dumps(result),flush=True);del gradients;gc.collect()
 done={'passed':all(r['passed'] for r in results),'alpha':alpha,'probes':results,'hypothetical_read_only_copied_state':True,'original_checkpoint_untouched':True,'no_trajectory_guarantee':True,'calibration_sha256':sha(RUN/'CALIBRATION_DONE.json'),'code_sha256':sha(__file__),'wall_seconds':time.time()-started,'gpu_hours':(time.time()-started)/3600,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES']};write(RUN/'NUMERIC_CHECK_DONE.json',done);write(REPORT/'NUMERIC_CHECK_DONE.json',done);assert done['passed'],'Fixed numeric strength acceptance failed; no alpha retry'
if __name__=='__main__':main()
