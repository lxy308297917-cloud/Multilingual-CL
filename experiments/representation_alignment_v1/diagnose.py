"""Content-only all-layer retrieval; preselect on Base/dev, never diagnostic/downstream."""
import argparse,time,os
from common import *
from transformers import AutoModelForCausalLM,AutoTokenizer

def extract(model,tok,rr,batch=8):
 result={lang:{method:[] for method in ('mean','last','weighted')} for lang in ('en','sw')}
 for lang in result:
  for start in range(0,len(rr),batch):
   check_compute()
   inputs=pack([r[lang+'_ids'] for r in rr[start:start+batch]],tok.pad_token_id)
   with torch.inference_mode():states=model.model(**inputs,output_hidden_states=True,use_cache=False).hidden_states
   for method in result[lang]:result[lang][method].append(torch.stack([pool(h,inputs['attention_mask'],method).cpu() for h in states]))
  for method in result[lang]:result[lang][method]=torch.cat(result[lang][method],dim=1)
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',default=str(BASE));p.add_argument('--name',default='Base');p.add_argument('--batch',type=int,default=8);a=p.parse_args();target=OUT/'diagnostics'/a.name;target.mkdir(parents=True,exist_ok=False);t=time.monotonic();seed()
 tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True);model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda().eval();model.config.use_cache=False
 costs={};splits=['dev','diagnostic']
 for split in splits:
  rr=rows(ROOT/'data'/(split+'.jsonl'));emb=extract(model,tok,rr,a.batch);torch.save(emb,target/(split+'_representations.pt'));metrics={method:[retrieval(emb['en'][method][i],emb['sw'][method][i]) for i in range(29)] for method in ('mean','last','weighted')};write(target/(split+'_metrics.json'),metrics);costs[split]={'pairs':len(rr),'tokens':sum(len(r['en_ids'])+len(r['sw_ids']) for r in rr)}
  if split=='dev' and a.name=='Base':
   score=lambda i: (-(metrics['mean'][i]['en_to_sw_p1']+metrics['mean'][i]['sw_to_en_p1'])/2,abs(i-14),i)
   layer=min(range(1,29),key=score);write(OUT/'LAYER_SELECTION.json',{'layer':layer,'rule':'Base/dev mean-pooled average bidirectional cosine P@1; ties nearest14 then lower. hidden_states[28] is final-normalized. No downstream/diagnostic scores used.','dev_sha256':sha(ROOT/'data/dev.jsonl'),'metrics_sha256':sha(target/'dev_metrics.json')})
 hard=json.loads((ROOT/'hard_pairs.json').read_text())
 for r in hard:
  for lang in ('en','sw'):r[lang+'_ids']=tok.encode(r[lang],add_special_tokens=False)
 hard_emb=extract(model,tok,hard,a.batch);hard_scores={}
 for method in ('mean','last','weighted'):
  hard_scores[method]=[]
  for i in range(29):
   en=F.normalize(hard_emb['en'][method][i],dim=-1);sw=F.normalize(hard_emb['sw'][method][i],dim=-1);cos=en@sw.T;ix=torch.arange(len(hard));margin=cos.diag()-cos[ix,ix^1];hard_scores[method].append({'accuracy':float((margin>0).float().mean()),'mean_margin':float(margin.mean())})
 write(target/'hard_pair_metrics.json',hard_scores)
 if a.name=='Base':
  layer=json.loads((OUT/'LAYER_SELECTION.json').read_text())['layer'];rr=rows(ROOT/'data/train.jsonl');bank=[]
  for start in range(0,len(rr),a.batch):
   x=pack([r['en_ids'] for r in rr[start:start+a.batch]],tok.pad_token_id)
   with torch.inference_mode():hs=model.model(**x,output_hidden_states=True,use_cache=False).hidden_states;bank.append(pool(hs[layer],x['attention_mask']).cpu())
  torch.save({'vectors':torch.cat(bank),'ids':[r['id'] for r in rr],'layer':layer,'train_sha256':sha(ROOT/'data/train.jsonl'),'model_sha256':sha(BASE/'model.safetensors'),'tokenizer_sha256':sha(BASE/'tokenizer.json')},OUT/'EN_TEACHER.pt');costs['teacher']={'tokens':sum(len(r['en_ids']) for r in rr),'pairs':len(rr)}
 write(target/'DONE.json',{'model':a.model,'release_sha256':sha(ROOT/'RELEASE.json'),'git_commit':json.loads((ROOT/'RELEASE.json').read_text())['git_commit'],'model_sha256':sha(Path(a.model)/'model.safetensors'),'elapsed_seconds':time.monotonic()-t,'costs':costs,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES']})
if __name__=='__main__':main()
