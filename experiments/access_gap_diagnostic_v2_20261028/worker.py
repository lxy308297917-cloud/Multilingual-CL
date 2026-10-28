from settings import *
from resources import GPUContext
import argparse,traceback
ap=argparse.ArgumentParser();ap.add_argument('--model',choices=MODELS,required=True);ap.add_argument('--uuid',required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
proto=read(REPORT/'PROTOCOL_FROZEN.json');spec=read(RUN/'MODELS_FROZEN.json')[a.model]
assert sha(Path(spec['model'])/'model.safetensors')==spec['sha256']
assert sha(RUN/'PROMPTS_PRIVATE.json')==proto['data_hashes']['prompts']
out=RUN/('smoke' if a.smoke else 'MCF')/a.model;assert not out.exists()
if not a.smoke:assert read(RUN/'smoke'/a.model/'DONE.json')['passed']
try:
 with GPUContext(a.uuid,a.model+('_smoke' if a.smoke else '_formal'),evaluation=True) as context:
  import torch,random,numpy as np
  from transformers import AutoModelForCausalLM
  torch.set_num_threads(4);random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  model=AutoModelForCausalLM.from_pretrained(spec['model'],local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
  prompts=read(RUN/'PROMPTS_PRIVATE.json');out.mkdir(parents=True)
  if a.smoke:
   prompts=[next(r for r in prompts if r['condition']==c and r['permutation']=='original') for c in proto['primary_conditions']]
  started=time.monotonic();counts=dict(prompt_tokens=0,candidate_tokens=0,generated_tokens=0)
  with (out/'PREDICTIONS.jsonl').open('x') as stream,torch.inference_mode():
   for i,r in enumerate(prompts):
    ids=torch.tensor([r['input_ids']],device='cuda');assert ids.shape[1]==r['input_tokens']<=proto['maximum_input_tokens']
    logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits[0,-1].float()
    full_logp=torch.log_softmax(logits,dim=-1);raw=full_logp[r['candidate_ids']]
    normalized=torch.softmax(raw,dim=-1);prediction=int(torch.argmax(raw));assert torch.isfinite(raw).all()
    if a.smoke:
     # Verify actual next-token indexing against an ordinary full-sequence forward.
     reference=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits[0,-1].float()
     torch.testing.assert_close(logits,reference,atol=0,rtol=0)
     assert int(torch.argmax(reference[r['candidate_ids']]))==prediction
     assert len(set(r['candidate_ids']))==4 and abs(float(normalized.sum())-1)<1e-5
    row={k:r[k] for k in ('id','task','group','condition','permutation','gold','original_gold','permutation_new_to_original','input_tokens','control_source_id')}
    row.update(model=a.model,model_sha256=spec['sha256'],raw_logprob=raw.cpu().tolist(),normalized_probability=normalized.cpu().tolist(),
     predicted='ABCD'[prediction],correct='ABCD'[prediction]==r['gold'],input_ids_sha256=hashlib.sha256(json.dumps(r['input_ids']).encode()).hexdigest(),
     candidate_ids=r['candidate_ids'],prompt_sha256=hashlib.sha256(r['prompt'].encode()).hexdigest())
    stream.write(json.dumps(row,ensure_ascii=False)+'\n');counts['prompt_tokens']+=len(r['input_ids']);counts['candidate_tokens']+=4
    if (i+1)%32==0:stream.flush();write(out/'PROGRESS.json',dict(n=i+1,total=len(prompts),counts=counts,updated_unix=time.time()))
  torch.cuda.synchronize()
  write(out/'DONE.json',dict(passed=True,n=len(prompts),counts=counts,model_sha256=spec['sha256'],
   protocol_sha256=sha(REPORT/'PROTOCOL_FROZEN.json'),predictions_sha256=sha(out/'PREDICTIONS.jsonl'),batch_size=1,
   smoke=a.smoke,standard_fullsequence_forward=True,deterministic_repeat_checked=a.smoke,no_generation=True,wall_seconds=time.monotonic()-started,
   peak_allocated_bytes=torch.cuda.max_memory_allocated(),finished_unix=time.time()))
except BaseException as e:
 if isinstance(e,SystemExit) and e.code==75:raise
 write(RUN/f'FAILED_{a.model}_{"smoke" if a.smoke else "formal"}.json',dict(error=str(e),traceback=traceback.format_exc(),failed_unix=time.time()))
 raise
