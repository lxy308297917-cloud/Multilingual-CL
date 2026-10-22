"""Only frozen layer10/content mean; fixed1000 independent pairs, no new layer search."""
from settings import *
import torch,torch.nn.functional as F,numpy as np
from common import seed,pack,pool,retrieval,check_compute
from transformers import AutoTokenizer,AutoModelForCausalLM

def main():
 start=time.time();out=RUN/'representation/D1';out.mkdir(parents=True,exist_ok=False);seed();modelpath=NEWTRAIN/'D1_seed42/checkpoint_004293/model';tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True);model=AutoModelForCausalLM.from_pretrained(modelpath,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval();model.config.use_cache=False;rr=rows(OLD/'data/diagnostic.jsonl');assert len(rr)==1000;allvec={};counts={}
 with torch.inference_mode():
  for lang in ('en','sw'):
   vectors=[];counts[lang]=sum(len(r[lang+'_ids']) for r in rr)
   for begin in range(0,len(rr),8):
    check_compute();assert time.time()-start<1800;x=pack([r[lang+'_ids'] for r in rr[begin:begin+8]],tok.pad_token_id);hs=model.model(**x,output_hidden_states=True,use_cache=False).hidden_states;vectors.append(pool(hs[10],x['attention_mask']).cpu());del x,hs
   allvec[lang]=torch.cat(vectors)
 torch.save({'en':allvec['en'],'sw':allvec['sw']},out/'LAYER10_VECTORS.pt')
 write(out/'DONE.json',{'model_sha256':sha(modelpath/'model.safetensors'),'data_sha256':sha(OLD/'data/diagnostic.jsonl'),'vectors_sha256':sha(out/'LAYER10_VECTORS.pt'),'counts':counts,'pairs':1000,'layer':10,'wall_seconds':time.time()-start,'gpu_hours':(time.time()-start)/3600,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES']})
 print('REPRESENTATION_D1_DONE',flush=True)
if __name__=='__main__':main()
