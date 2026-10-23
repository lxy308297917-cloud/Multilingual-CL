"""Generic fixed-protocol generation for old, confirmation and single prompt diagnostic."""
from evaluation_common import *
import torch,traceback,subprocess,argparse
from transformers import AutoTokenizer,AutoModelForCausalLM

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--split',choices=['old','confirmation','prompt'],required=True);a=ap.parse_args();start=time.time();models=json.loads((RUN/'MODELS_FROZEN.json').read_text());spec=models[a.model];modelpath=pathlib.Path(spec['model']);assert sha(modelpath/'model.safetensors')==spec['sha256'] if spec.get('sha256') else json.loads((modelpath.parent.parent/'TRAINING_DONE.json').read_text())['steps']==4293
 meta=json.loads((RUN/({'old':'OLD_FROZEN.json','confirmation':'CONFIRMATION_FROZEN.json','prompt':'PROMPT_FROZEN.json'}[a.split])).read_text());assert sha(meta['data_path'])==meta['data_sha256'];rr=rows(meta['data_path']);out=RUN/'evaluation'/a.split/a.model;out.mkdir(parents=True,exist_ok=False)
 seed();tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True);model=AutoModelForCausalLM.from_pretrained(modelpath,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval();torch.cuda.reset_peak_memory_stats();eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos]);physical=[];counts={'prompt_tokens':0,'generated_tokens':0};completed=0
 write(out/'CONFIG.json',{'model_sha256':sha(modelpath/'model.safetensors'),'model':str(modelpath),'data_sha256':meta['data_sha256'],'code_sha256':sha(__file__),'scorer_hashes':{p:sha(REPORT/'code'/p) for p in ('reading_scoring.py','aux_scoring.py')},'generation_config':model.generation_config.to_dict(),'torch_version':torch.__version__,'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES'],'batch':4,'caps':meta['caps'],'protocol_sha256':sha(REPORT/'PROTOCOL_FROZEN.json')})
 try:
  with (out/'PREDICTIONS.jsonl').open('w') as stream,torch.inference_mode():
   for partition in ('sw_reading','en_reading','translation','en_instruction'):
    torch.use_deterministic_algorithms(partition in ('translation','en_instruction'));torch.backends.cudnn.deterministic=partition in ('translation','en_instruction');here=[r for r in rr if r['study_partition']==partition];cap=meta['caps'][partition]
    for begin in range(0,len(here),4):
     check_compute();budget_guard();assert time.time()-start<3900,'Endpoint technical time ceiling';batch=here[begin:begin+4];prompts=[]
     for r in batch:
      messages=([{'role':'system','content':r['system']}] if r.get('system') else [])+[{'role':'user','content':r['prompt']}];prompts.append(tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True))
     width=max(map(len,prompts));x={'input_ids':torch.tensor([[tok.pad_token_id]*(width-len(p))+p for p in prompts],device='cuda'),'attention_mask':torch.tensor([[0]*(width-len(p))+[1]*len(p) for p in prompts],device='cuda')};yy=model.generate(**x,max_new_tokens=cap,do_sample=False,pad_token_id=tok.pad_token_id,eos_token_id=sorted(eos),use_cache=True)[:,width:].cpu().tolist()
     for r,ids,pr in zip(batch,yy,prompts):
      end=next((i+1 for i,v in enumerate(ids) if v in eos),len(ids));ids=ids[:end];text=tok.decode(ids,skip_special_tokens=True).strip()
      try:
       from langdetect import detect,DetectorFactory
       DetectorFactory.seed=0;language=detect(text)
      except Exception:language='und'
      rec={'id':r['id'],'group_id':r['group_id'],'task':r['task'],'model':a.model,'text':text,'token_ids':ids,'prompt_tokens':len(pr),'generated_tokens':len(ids),'capped':not ids or ids[-1] not in eos,'ending_token_id':ids[-1] if ids else None,'language_diagnostic':language,'scores':scoring(r,text),'behavior':behavior(text),'study_partition':partition,'score_eligible':r['score_eligible']};stream.write(json.dumps(rec,ensure_ascii=False)+'\n');stream.flush();completed+=1;counts['prompt_tokens']+=len(pr);counts['generated_tokens']+=len(ids)
     physical.append(int(subprocess.check_output(['nvidia-smi','--id='+os.environ['CUDA_VISIBLE_DEVICES'],'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip()));write(out/'PROGRESS.json',{'completed':completed,'partition':partition,'elapsed_seconds':time.time()-start});del x,yy
  assert completed==len(rr);write(out/'DONE.json',{'rows':completed,'scored_rows':sum(r['score_eligible'] for r in rr),'model_sha256':sha(modelpath/'model.safetensors'),'data_sha256':meta['data_sha256'],'counts':counts,'wall_seconds':time.time()-start,'gpu_hours':(time.time()-start)/3600,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'physical_peak_mib_observed':max(physical),'predictions_sha256':sha(out/'PREDICTIONS.jsonl'),'not_formal_test':True})
 except BaseException as e:write(out/'FAILED.json',{'error':str(e),'traceback':traceback.format_exc(),'completed':completed});raise
if __name__=='__main__':main()
