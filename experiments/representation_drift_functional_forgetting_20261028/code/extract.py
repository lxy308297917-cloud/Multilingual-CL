"""Inference-only pooled residual-stream cache under the frozen diagnostic protocol."""
from pathlib import Path
import os,sys,json,time,hashlib,argparse,traceback,re
ROOT=Path(__file__).resolve().parents[3]
REPORT=ROOT/'docs/results/representation_drift_functional_forgetting_20261028'
WORK=ROOT/'work/representation_drift_functional_forgetting_20261028'
H=Path('/data0/users/lixiaoyang');BASE=H/'models/Qwen2.5-1.5B-Instruct'
sys.path.insert(0,str(ROOT/'outputs/sw_fullbudget_baselines_20261028/code'))
from settings import sha,read,write
import resources
resources.RUN=WORK
GPUContext=resources.GPUContext
ap=argparse.ArgumentParser();ap.add_argument('--model',required=True,choices=['Base','Dense','Element','Protected']);ap.add_argument('--uuid',required=True);ap.add_argument('--attempt',default='first');a=ap.parse_args()
protocol=read(REPORT/'PROTOCOL_FROZEN.json');source=protocol['input_manifest'];assert sha(source['path'])==source['sha256']
rows=[json.loads(line) for line in Path(source['path']).open()]
out=WORK/a.model/a.attempt;assert not out.exists()
modelpath=BASE;done=None
if a.model!='Base':
 donepath=H/'models/checkpoint/sw_fullbudget_baselines_20261028'/f'{a.model}_first/TRAINING_DONE.json'
 done=read(donepath);req=protocol['adapted_model_requirement']
 assert done['steps']==req['steps'] and done['counts']['main_input']==req['main_input_tokens']
 assert done['counts']['main_supervised']==req['main_supervised_tokens']
 ident=read(donepath.parent/'RUN_IDENTITY.json');assert ident['config_sha256']==req['training_config_sha256']
 modelpath=Path(done['final_model'])
 assert sha(modelpath/'model.safetensors')==done['final_weights_sha256']['model.safetensors']
try:
 with GPUContext(a.uuid,'drift_'+a.model+'_'+a.attempt) as context:
  import torch,numpy as np,random
  from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig
  from fractions import Fraction
  random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
  torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
  model=AutoModelForCausalLM.from_pretrained(modelpath,torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda().eval()
  assert model.config.num_hidden_layers==28 and model.config.hidden_size==1536
  assert not model.training and all(not p.requires_grad or p.grad is None for p in model.parameters())
  for p in model.parameters():p.requires_grad_(False)
  out.mkdir(parents=True);start=time.time()
  identity=dict(model=a.model,model_path=str(modelpath),model_sha256=sha(modelpath/'model.safetensors'),
   training_done=done,protocol_sha256=sha(REPORT/'PROTOCOL_FROZEN.json'),input_sha256=source['sha256'],
   tokenizer_config_sha256=sha(BASE/'tokenizer_config.json'),source_sha256=sha(Path(__file__)),
   pid=os.getpid(),uuid=a.uuid,started_unix=start,scientific_training=False)
  write(out/'IDENTITY.json',identity)
  cache=np.lib.format.open_memmap(out/'REPRESENTATIONS.npy',mode='w+',dtype=np.float32,shape=(len(rows),28,3,1536))
  capture=[False];pooled={};current=[None]
  def hook(index):
   def collect(module,args,result):
    if not capture[0]:return
    hidden=result[0] if isinstance(result,tuple) else result
    assert hidden.ndim==3 and hidden.shape[0]==1
    r=current[0];positions=torch.tensor(r['content_indices'],device=hidden.device)
    vectors=torch.stack((hidden[0,r['decision_position']].float(),hidden[0,r['last_content_position']].float(),
      hidden[0].index_select(0,positions).float().mean(0)))
    assert torch.isfinite(vectors).all();pooled[index]=vectors.cpu().numpy()
   return collect
  handles=[layer.register_forward_hook(hook(i)) for i,layer in enumerate(model.model.layers)]
  label_ids=torch.tensor(protocol['prompt_protocol']['label_token_ids'],device='cuda')
  generation=GenerationConfig.from_pretrained(BASE,local_files_only=True)
  eos=generation.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
  def math_number(s):
   cleaned=s.strip().replace(',','').replace('−','-')
   if not re.fullmatch(r'[-+]?\d+(?:\.\d+)?(?:\s*/\s*[-+]?\d+(?:\.\d+)?)?',cleaned):return None
   try:
    if '/' in cleaned:
     left,right=cleaned.split('/');return Fraction(left.strip())/Fraction(right.strip())
    return Fraction(cleaned)
   except (ValueError,ZeroDivisionError):return None
  def final_math(text):
   matches=re.findall(r'(?m)^\s*####\s*([^\n]+)\s*$',text)
   if len(matches)!=1:return None,'missing_or_multiple_final_markers'
   value=math_number(matches[0]);return value,('parsed' if value is not None else 'unsupported_or_ambiguous_final_expression')
  # Parser acceptance precedes any actual model comparison.
  assert final_math('reasoning 17\n#### 18')[0]==Fraction(18)
  assert final_math('reasoning 18, final answer absent')[0] is None
  assert final_math('#### 18\n#### 19')[0] is None
  assert final_math('#### 1/2')[0]==Fraction('0.5')
  with (out/'PREDICTIONS.jsonl').open('x') as stream,torch.inference_mode():
   for i,r in enumerate(rows):
    current[0]=r;capture[0]=True;pooled.clear()
    ids=torch.tensor([r['input_ids']],device='cuda');mask=torch.ones_like(ids)
    result=model(input_ids=ids,attention_mask=mask,use_cache=False,output_hidden_states=(i==0),return_dict=True)
    assert set(pooled)==set(range(28));capture[0]=False
    cache[i]=np.stack([pooled[j] for j in range(28)])
    if i==0:
     # Full hidden-state API confirms our raw post-block boundary. Last API output is normed.
     for j in (0,13,26):
      expected=result.hidden_states[j+1][0,r['decision_position']].float().cpu().numpy()
      assert np.array_equal(expected,pooled[j][0]),j
     raw=torch.from_numpy(pooled[27][0]).to(device='cuda',dtype=torch.bfloat16)
     assert torch.equal(model.model.norm(raw),result.hidden_states[-1][0,r['decision_position']])
     write(out/'EXTRACTION_ACCEPTANCE.json',dict(passed=True,real_Base_or_endpoint_forward=True,
      block0_13_26_match_hidden_state_API=True,block27_pre_final_norm_verified=True,
      decision_is_last_effective_input=True,no_padding=True,no_truncation=True,no_gradients=True))
    record={k:r[k] for k in ('id','pair_id','group','phase','task','language','input_sha256')}
    record.update(model=a.model,input_length=len(r['input_ids']),activation_row=i)
    if r['task']!='gsm8k':
     logits=result.logits[0,-1].float();scores=logits.index_select(0,label_ids).log_softmax(0).cpu().numpy()
     pred=int(scores.argmax());gold=r['gold_index'];sorted_scores=np.sort(scores)
     record.update(gold_index=gold,prediction=pred,correct=pred==gold,label_logprobs=scores.tolist(),
      gold_label_probability=float(np.exp(scores[gold])),top_two_margin=float(sorted_scores[-1]-sorted_scores[-2]),
      gold_vs_best_other_margin=float(scores[gold]-max(s for k,s in enumerate(scores) if k!=gold)),
      confidence_caveat='conditional over four labels, not calibrated reasoning confidence',status='parsed')
     del logits
    del result
    if r['task']=='gsm8k':
     tokens=model.generate(input_ids=ids,attention_mask=mask,generation_config=generation,
      do_sample=False,num_beams=1,max_new_tokens=1024,pad_token_id=tok.pad_token_id,
      eos_token_id=eos,use_cache=True)[0,ids.shape[1]:].cpu().tolist()
     text=tok.decode(tokens,skip_special_tokens=True);answer,status=final_math(text)
     reference=math_number(r['reference_answer'].split('####')[-1]);assert reference is not None
     record.update(generated_token_ids=tokens,text=text,reference_final=str(reference),
      extracted_answer=str(answer) if answer is not None else None,status=status,
      correct=(answer==reference) if answer is not None else None,
      capped=not tokens or tokens[-1] not in eos,generated_tokens=len(tokens),
      strict_format=bool(re.search(r'(?m)^\s*####\s*[^\n]+\s*$',text)),
      confidence_caveat='No probability proxy for mathematical reasoning correctness')
    stream.write(json.dumps(record,ensure_ascii=False)+'\n');stream.flush()
    if (i+1)%32==0 or i+1==len(rows):
     cache.flush();write(out/'PROGRESS.json',dict(completed=i+1,total=len(rows),updated_unix=time.time()))
     print(a.model,i+1,'/',len(rows),flush=True)
    del ids,mask
  for handle in handles:handle.remove()
  cache.flush();del cache
  write(out/'DONE.json',dict(model=a.model,n=len(rows),protocol_sha256=identity['protocol_sha256'],
   model_sha256=identity['model_sha256'],finished_unix=time.time(),wall_seconds=time.time()-start,
   peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
   files={name:sha(out/name) for name in ('IDENTITY.json','EXTRACTION_ACCEPTANCE.json','REPRESENTATIONS.npy','PREDICTIONS.jsonl')},
   no_training=True,formal_benchmark_scores_not_replaced=True))
except BaseException as error:
 if isinstance(error,SystemExit) and error.code==75:raise
 WORK.mkdir(parents=True,exist_ok=True)
 write(WORK/f'FAILED_{a.model}_{a.attempt}.json',dict(error=str(error),traceback=traceback.format_exc(),unix=time.time()))
 raise
