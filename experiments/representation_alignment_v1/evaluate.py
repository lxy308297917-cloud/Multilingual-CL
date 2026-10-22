"""Same frozen independent downstream development protocol as sparse comparison."""
import argparse,sys,subprocess,time,traceback
from common import *
from transformers import AutoTokenizer,AutoModelForCausalLM
sys.path.insert(0,str(ROOT.parent/'paired_capability_sparse_v1'))
from development_scoring import score_record,summarize

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--name',required=True);a=p.parse_args();out=OUT/'development'/a.name;out.mkdir(parents=True,exist_ok=False);t=time.monotonic();records=[];counts={'prompt':0,'generated':0};physical=[]
 try:
  seed();torch.set_num_threads(4);protocol=json.loads((ROOT/'protocol.json').read_text());old=json.loads((ROOT.parent/'paired_capability_sparse_v1/comparison_protocol.v1.json').read_text());manifest=json.loads(Path(old['development']['manifest']).read_text());tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True);model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval();eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos]);write(out/'CONFIG.json',{'model':a.model,'model_sha256':sha(Path(a.model)/'model.safetensors'),'label':a.name,'protocol_sha256':sha(ROOT/'protocol.json'),'dev_protocol_sha256':sha(ROOT.parent/'paired_capability_sparse_v1/comparison_protocol.v1.json'),'greedy':True,'batch':4,'no_formal_test':True,'max_new_tokens':old['development']['max_new_tokens']})
  with (out/'PREDICTIONS.jsonl').open('w') as f,torch.inference_mode():
   for partition,info in manifest['partitions'].items():
    rr=rows(info['path']);assert sha(info['path'])==info['sha256'];cap=old['development']['max_new_tokens'][partition]
    for start in range(0,len(rr),4):
     check_compute()
     if time.monotonic()-t>protocol['per_development_model_gpu_hours']*3600:raise RuntimeError('Development GPUh ceiling')
     batch=rr[start:start+4];prompts=[]
     for r in batch:
      messages=([{'role':'system','content':r['system']}] if r.get('system') else [])+[{'role':'user','content':r['prompt']}];prompts.append(tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True))
     w=max(map(len,prompts));x={'input_ids':torch.tensor([[tok.pad_token_id]*(w-len(z))+z for z in prompts],device='cuda'),'attention_mask':torch.tensor([[0]*(w-len(z))+[1]*len(z) for z in prompts],device='cuda')};yy=model.generate(**x,max_new_tokens=cap,do_sample=False,pad_token_id=tok.pad_token_id,eos_token_id=sorted(eos),use_cache=True)[:,w:].cpu().tolist()
     for r,prompt,ids in zip(batch,prompts,yy):
      end=next((i+1 for i,v in enumerate(ids) if v in eos),len(ids));ids=ids[:end];text=tok.decode(ids,skip_special_tokens=True).strip()
      try:
       from langdetect import detect,DetectorFactory
       DetectorFactory.seed=0;lang=detect(text)
      except Exception:lang='und'
      rec={'id':r['id'],'group_id':r['group_id'],'task':r['task'],'partition':partition,'label':a.name,'text':text,'token_ids':ids,'generated_tokens':len(ids),'prompt_tokens':len(prompt),'capped':not ids or ids[-1] not in eos,'ending_token_id':ids[-1] if ids else None,'language_diagnostic':lang,'scores':score_record(partition,r,text)};records.append(rec);f.write(json.dumps(rec,ensure_ascii=False)+'\n');f.flush();counts['prompt']+=len(prompt);counts['generated']+=len(ids)
     physical.append(int(subprocess.check_output(['nvidia-smi','--id='+os.environ['CUDA_VISIBLE_DEVICES'],'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip()));write(out/'PROGRESS.json',{'completed':len(records),'partition':partition,'elapsed_seconds':time.monotonic()-t});del x,yy
  write(out/'DONE.json',{'label':a.name,'scores':summarize(records),'counts':counts,'predictions_sha256':sha(out/'PREDICTIONS.jsonl'),'wall_seconds':time.monotonic()-t,'physical_peak_mib_observed':max(physical),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'not_formal_benchmark':True})
 except BaseException as e:write(out/'FAILED.json',{'error':str(e),'traceback':traceback.format_exc(),'completed':len(records)});raise
if __name__=='__main__':main()
