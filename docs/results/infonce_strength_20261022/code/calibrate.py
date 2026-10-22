"""One preregistered coefficient from16 actual Base training-side batches, never development."""
from settings import *
import importlib.util,torch,math,numpy as np,gc
from transformers import AutoTokenizer,AutoModelForCausalLM
from common import seed,pack,ce,pool,auxiliary,check_compute

def main():
 started=time.time();assert not (RUN/'CALIBRATION_DONE.json').exists();seed()
 plan=json.loads((REPORT/'PROTOCOL_FROZEN.json').read_text());batchplan=json.loads((RUN/'CALIBRATION_BATCHES_FROZEN.json').read_text());assert sha(RUN/'CALIBRATION_BATCHES_FROZEN.json')==plan['calibration']['batches_sha256']
 spec=importlib.util.spec_from_file_location('recipe',H/'reproductions/sw_recipe_fft_20261010/train.py');recipe=importlib.util.module_from_spec(spec);spec.loader.exec_module(recipe)
 manifest=json.loads((OLD/'data/main_manifest.json').read_text());indices=json.loads((OLD/'data/main_prefix.json').read_text());assert sha(manifest['source'])==manifest['source_sha256'];ds=recipe.IndexedRows(manifest['source'],indices);rr=rows(OLD/'data/train.jsonl');teacher=torch.load(TRAIN/'EN_TEACHER.pt',map_location='cpu',weights_only=False);assert teacher['ids']==[r['id'] for r in rr] and teacher['layer']==10 and teacher['train_sha256']==sha(OLD/'data/train.jsonl');assert teacher['model_sha256']==sha(BASE/'model.safetensors');bank=teacher['vectors'].cuda()
 tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True);model=AutoModelForCausalLM.from_pretrained(BASE,torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda();model.config.use_cache=False;model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.train();params=list(model.parameters())
 def norm():
  # FP32 reduction per tensor; no giant concatenation and no precision change to training.
  return math.sqrt(sum(float(p.grad.detach().float().square().sum()) for p in params if p.grad is not None))
 records=[];counts={'main_input':0,'main_supervised':0,'CE_aux_input':0,'CE_aux_supervised':0,'NCE_aux_input':0}
 for bi,b in enumerate(batchplan['batches']):
  check_compute();assert time.time()-started<plan['budget']['calibration_GPUh']*3600
  rec={'batch':bi,'event':b['event']};model.zero_grad(set_to_none=True);ml=0.
  for pos in b['main_prefix_positions']:
   row=ds[pos];x=pack([row],tok.pad_token_id,labels=True);lab=x.pop('labels');loss=ce(model(**x,use_cache=False).logits,lab);(loss/2).backward();ml+=float(loss.detach())/2;counts['main_input']+=len(row['input_ids']);counts['main_supervised']+=sum(v!=-100 for v in row['labels'][1:]);del x,lab,loss
  rec['main_CE']=ml;rec['main_grad_L2']=norm();model.zero_grad(set_to_none=True)
  ce_rows=[x for i in b['aux_indices'] for x in rr[i]['ce_rows']];al=0.
  for begin in range(0,16,4):
   chunk=ce_rows[begin:begin+4];x=pack(chunk,tok.pad_token_id,labels=True);lab=x.pop('labels');loss=ce(model(**x,use_cache=False).logits,lab);(loss*(len(chunk)/16)).backward();al+=float(loss.detach())*len(chunk)/16;counts['CE_aux_input']+=sum(len(r['input_ids']) for r in chunk);counts['CE_aux_supervised']+=sum(sum(z!=-100 for z in r['labels'][1:]) for r in chunk);del x,lab,loss
  rec['translation_CE']=al;rec['raw_CE_grad_L2']=norm();model.zero_grad(set_to_none=True);al=0.
  for begin in range(0,8,4):
   ii=b['aux_indices'][begin:begin+4];x=pack([rr[i]['sw_ids'] for i in ii],tok.pad_token_id);hs=model.model(**x,output_hidden_states=True,use_cache=False).hidden_states;v=pool(hs[10],x['attention_mask']);loss=auxiliary(v,bank,torch.tensor(ii,device='cuda'),'D');(loss*.5).backward();al+=float(loss.detach())*.5;counts['NCE_aux_input']+=sum(len(rr[i]['sw_ids']) for i in ii);del x,hs,v,loss
  rec['InfoNCE']=al;rec['raw_NCE_grad_L2']=norm();model.zero_grad(set_to_none=True)
  assert all(math.isfinite(rec[k]) and rec[k]>0 for k in ('main_grad_L2','raw_CE_grad_L2','raw_NCE_grad_L2'))
  rec['CE_scaled_to_main']=.1*rec['raw_CE_grad_L2']/rec['main_grad_L2'];rec['NCE_raw_to_main']=rec['raw_NCE_grad_L2']/rec['main_grad_L2'];rec['elapsed_seconds']=time.time()-started;records.append(rec);write(RUN/'CALIBRATION_PARTIAL.json',records);print(json.dumps(rec),flush=True)
 assert len(records)==16
 target=float(np.median([r['CE_scaled_to_main'] for r in records]));unscaled=float(np.median([r['NCE_raw_to_main'] for r in records]));alpha=target/unscaled;assert math.isfinite(alpha) and alpha>0
 result={'alpha':alpha,'formula':plan['calibration']['formula'],'batches':16,'CE_scaled_to_main_median':target,'NCE_original_scaled_to_main_median':.1*unscaled,'NCE_calibrated_scaled_to_main_median':alpha*unscaled,'coefficient_multiplier_vs_D0':alpha/.1,'records':records,'counts':counts,'base_model_sha256':sha(BASE/'model.safetensors'),'teacher_sha256':sha(TRAIN/'EN_TEACHER.pt'),'batchplan_sha256':sha(RUN/'CALIBRATION_BATCHES_FROZEN.json'),'protocol_sha256':sha(REPORT/'PROTOCOL_FROZEN.json'),'code_sha256':sha(__file__),'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES'],'wall_seconds':time.time()-started,'gpu_hours':(time.time()-started)/3600,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'fixed_once_no_development_selection':True}
 write(RUN/'CALIBRATION_DONE.json',result);write(REPORT/'CALIBRATION_DONE.json',result);print('CALIBRATED_ALPHA',alpha,flush=True)
if __name__=='__main__':main()
