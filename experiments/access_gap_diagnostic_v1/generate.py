"""Frozen A-E evaluation only; preserve every answer, including failed parsing."""
from common import *
from resource_context import GPUContext
from scoring import score
import argparse,traceback
p=argparse.ArgumentParser();p.add_argument('--uuid',required=True);p.add_argument('--model',choices=MODELS,required=True);a=p.parse_args()
proto=read(DEST/'PROTOCOL_FROZEN.json');spec=read(RUN/'SCREEN_MODELS_FROZEN.json')[a.model]
assert sha(Path(spec['model'])/'model.safetensors')==spec['sha256']
for s in proto['data'].values():assert sha(s['path'])==s['sha256']
rows=read(RUN/'PROMPTS_PRIVATE.json');out=RUN/'generation'/a.model
assert not out.exists()
try:
 with GPUContext(a.uuid,'access_generation_'+a.model,evaluation=True) as ctx:
  import torch,random,numpy as np
  from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig
  random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
  torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
  model=AutoModelForCausalLM.from_pretrained(spec['model'],local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
  cfg=GenerationConfig.from_pretrained(BASE,local_files_only=True);eos=cfg.eos_token_id;eos=sorted(set(eos if isinstance(eos,list) else [eos]))
  out.mkdir(parents=True);counts=dict(prompt_tokens=0,generated_tokens=0);n=0
  write(out/'IDENTITY.json',dict(model_sha256=spec['sha256'],protocol_sha256=sha(DEST/'PROTOCOL_FROZEN.json'),generation=proto['generation'],uuid=a.uuid))
  with (out/'PREDICTIONS.jsonl').open('x') as stream,torch.inference_mode():
   for begin in range(0,len(rows),4):
    batch=rows[begin:begin+4]
    prompts=[tok.apply_chat_template([dict(role='user',content=r['prompt'])],tokenize=True,add_generation_prompt=True) for r in batch]
    assert all(len(ids)==r['prompt_tokens'] and len(ids)<=proto['generation']['max_input_tokens'] for ids,r in zip(prompts,batch))
    width=max(map(len,prompts))
    ids=torch.tensor([[tok.pad_token_id]*(width-len(x))+x for x in prompts],device='cuda')
    mask=torch.tensor([[0]*(width-len(x))+[1]*len(x) for x in prompts],device='cuda')
    output=model.generate(input_ids=ids,attention_mask=mask,generation_config=cfg,do_sample=False,num_beams=1,max_new_tokens=256,pad_token_id=tok.pad_token_id,eos_token_id=eos,use_cache=True)[:,width:].cpu().tolist()
    for row,tokens,prompt in zip(batch,output,prompts):
     stop=next((i+1 for i,t in enumerate(tokens) if t in eos),len(tokens));tokens=tokens[:stop]
     text=tok.decode(tokens,skip_special_tokens=True).strip();scores=score(text,row['gold']);explanation=scores.pop('explanation')
     import re,collections
     language='und';confidence=None
     if len(re.findall(r'[^\W\d_]+',explanation))>=5:
      try:
       from langdetect import detect_langs,DetectorFactory
       DetectorFactory.seed=0;det=detect_langs(explanation)[0];confidence=det.prob
       if confidence>=.7:language=det.lang
      except Exception:pass
     words=text.split();rep=sum(v-1 for v in collections.Counter(tuple(words[i:i+4]) for i in range(max(0,len(words)-3))).values())
     record={k:row[k] for k in ('id','task','group','condition','gold','prompt_sha256')}
     record.update(model=a.model,text=text,token_ids=tokens,scores=scores,prompt_tokens=len(prompt),generated_tokens=len(tokens),capped=not tokens or tokens[-1] not in eos,explanation_language=language,language_confidence=confidence,repeated_4grams=rep)
     stream.write(json.dumps(record,ensure_ascii=False)+'\n');n+=1;counts['prompt_tokens']+=len(prompt);counts['generated_tokens']+=len(tokens)
    stream.flush();write(out/'PROGRESS.json',dict(n=n,counts=counts,updated_unix=time.time()))
    del ids,mask,output
  assert n==640
  write(out/'DONE.json',dict(n=n,counts=counts,predictions_sha256=sha(out/'PREDICTIONS.jsonl'),model_sha256=spec['sha256'],peak_allocated_bytes=torch.cuda.max_memory_allocated(),finished_unix=time.time()))
except BaseException as e:
 if isinstance(e,SystemExit) and e.code==75:raise
 write(RUN/('FAILED_generation_'+a.model+'.json'),dict(error=str(e),traceback=traceback.format_exc(),finished_unix=time.time()))
 raise
