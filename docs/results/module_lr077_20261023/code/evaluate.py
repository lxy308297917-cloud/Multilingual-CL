"""Frozen512generation/511scoring endpoint; identical task-wise prompt, batching and generation calls."""
from evaluation_common import *
import torch,traceback,subprocess
from common import seed,check_compute
from transformers import AutoTokenizer,AutoModelForCausalLM

def main():
 started=time.time();meta,rr=inputs();group=os.environ['MODULE_BUDGET_GROUP'];assert group=='S-All';out=RUN/'evaluation'/STUDY_NAME;out.mkdir(parents=True,exist_ok=False);done=json.loads((OUT/group/'TRAINING_DONE.json').read_text());assert done['steps']==4293 and done['counts']['main_input']==1536351;modelpath=pathlib.Path(done['model']);cp=modelpath.parent;receipt=json.loads((cp/'RECOVERABLE.json').read_text());assert sha(modelpath/'model.safetensors')==receipt['files']['model/model.safetensors']
 seed();tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True);model=AutoModelForCausalLM.from_pretrained(modelpath,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval();eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos]);physical=[];counts={'prompt_tokens':0,'generated_tokens':0};completed=0
 write(out/'CONFIG.json',{'model_sha256':sha(modelpath/'model.safetensors'),'model':str(modelpath),'data_sha256':meta['data_sha256'],'code_sha256':sha(__file__),'scorer_hashes':{p:sha(REPORT/'code'/p) for p in ('reading_scoring.py','aux_scoring.py')},'generation_config':model.generation_config.to_dict(),'torch_version':torch.__version__,'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES'],'batch':4,'execution_flags_per_partition':{'reading':'deterministic_algorithmsFalse/cudnn_deterministicFalse, same v2worker','translation_instruction':'deterministic_algorithmsTrue/cudnn_deterministicTrue, same originalseed','all':'TF32off, CUBLAS_WORKSPACE_CONFIG=:4096:8, benchmarkFalse'},'caps':meta['caps'],'protocol_sha256':sha(REPORT/'PROTOCOL_FROZEN.json')})
 try:
  with (out/'PREDICTIONS.jsonl').open('w') as stream,torch.inference_mode():
   for partition in ('sw_reading','en_reading','translation','en_instruction'):
    # Match the cached reading-v2 worker (default nondeterministic flag) and original other-task seed() flags.
    torch.use_deterministic_algorithms(partition in ('translation','en_instruction'));torch.backends.cudnn.deterministic=partition in ('translation','en_instruction')
    rows_here=[r for r in rr if r['study_partition']==partition];cap=meta['caps'][partition]
    for begin in range(0,len(rows_here),4):
     check_compute();assert time.time()-started<3060
     batch=rows_here[begin:begin+4];prompts=[]
     for r in batch:
      messages=([{'role':'system','content':r['system']}] if r.get('system') else [])+[{'role':'user','content':r['prompt']}];prompts.append(tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True))
     width=max(map(len,prompts));x={'input_ids':torch.tensor([[tok.pad_token_id]*(width-len(p))+p for p in prompts],device='cuda'),'attention_mask':torch.tensor([[0]*(width-len(p))+[1]*len(p) for p in prompts],device='cuda')};yy=model.generate(**x,max_new_tokens=cap,do_sample=False,pad_token_id=tok.pad_token_id,eos_token_id=sorted(eos),use_cache=True)[:,width:].cpu().tolist()
     for r,ids,pr in zip(batch,yy,prompts):
      end=next((i+1 for i,v in enumerate(ids) if v in eos),len(ids));ids=ids[:end];text=tok.decode(ids,skip_special_tokens=True).strip()
      try:
       from langdetect import detect,DetectorFactory
       DetectorFactory.seed=0;language=detect(text)
      except Exception:language='und'
      rec={'id':r['id'],'group_id':r['group_id'],'task':r['task'],'model':STUDY_NAME,'text':text,'token_ids':ids,'prompt_tokens':len(pr),'generated_tokens':len(ids),'capped':not ids or ids[-1] not in eos,'ending_token_id':ids[-1] if ids else None,'language_diagnostic':language,'scores':scoring(r,text),'behavior':behavior(text),'study_partition':partition,'score_eligible':r['score_eligible']};stream.write(json.dumps(rec,ensure_ascii=False)+'\n');stream.flush();completed+=1;counts['prompt_tokens']+=len(pr);counts['generated_tokens']+=len(ids)
     physical.append(int(subprocess.check_output(['nvidia-smi','--id='+os.environ['CUDA_VISIBLE_DEVICES'],'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip()));write(out/'PROGRESS.json',{'completed':completed,'partition':partition,'elapsed_seconds':time.time()-started});del x,yy
  assert completed==512
  write(out/'DONE.json',{'rows':completed,'scored_rows':511,'reused':False,'model_sha256':sha(modelpath/'model.safetensors'),'data_sha256':meta['data_sha256'],'counts':counts,'wall_seconds':time.time()-started,'gpu_hours':(time.time()-started)/3600,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'physical_peak_mib_observed':max(physical),'predictions_sha256':sha(out/'PREDICTIONS.jsonl'),'not_formal_test':True})
 except BaseException as e:write(out/'FAILED.json',{'error':str(e),'traceback':traceback.format_exc(),'completed':completed});raise
if __name__=='__main__':main()
