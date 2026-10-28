"""Full-scope endpoint task; no score-selected checkpoints and no old scores mixed in."""
from settings import *
from resources import GPUContext
import argparse,sys,traceback,runpy,importlib.util,collections
ap=argparse.ArgumentParser();ap.add_argument('--uuid',required=True);ap.add_argument('--model',choices=('Base',*METHODS),required=True)
ap.add_argument('--task',choices=('belebele_en','belebele_sw','knowledge_en','knowledge_sw','translation','ifeval','gsm8k'),required=True)
ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
stage='smoke' if a.smoke else 'formal';out=evaluation_output(a.model,stage,a.task);assert not out.exists()
if a.model=='Base':modelpath=BASE
else:
 trainout=H/'models/checkpoint/sw_fullbudget_baselines_20261028'/f'{a.model}_first'
 done=read(trainout/'TRAINING_DONE.json');assert done['steps']==42465;modelpath=Path(done['final_model'])
proto=read(REPORT/'EVALUATION_FROZEN.json');modelsha=sha(modelpath/'model.safetensors')
for r in proto['sources'].values():assert sha(r['path'])==r['sha256']
try:
 with GPUContext(a.uuid,'eval_'+a.model+'_'+stage+'_'+a.task+('_'+out.name if out.name!=a.task else ''),evaluation=True):
  import torch,random,numpy as np
  random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  out.mkdir(parents=True);write(out/'IDENTITY.json',dict(model=a.model,model_path=str(modelpath),model_sha256=modelsha,
   protocol_sha256=sha(REPORT/'EVALUATION_FROZEN.json'),smoke=a.smoke,uuid=a.uuid))
  from transformers import AutoTokenizer
  original=AutoTokenizer.from_pretrained
  def fixed(path,*args,**kwargs):
   kwargs.pop('revision',None);kwargs['local_files_only']=True;return original(BASE,*args,**kwargs)
  AutoTokenizer.from_pretrained=staticmethod(fixed)
  if a.task.startswith(('belebele','knowledge')):
   import pyarrow.parquet as pq
   from lighteval.models.transformers.transformers_model import TransformersModel
   for n in ('greedy_until','loglikelihood','loglikelihood_rolling'):
    fn=getattr(TransformersModel,n)
    if hasattr(fn,'__wrapped__'):setattr(TransformersModel,n,fn.__wrapped__)
   if a.task.startswith('belebele'):
    old=ROOT/'outputs/benchmark_direction_20261026';sys.path.insert(0,str(old/'bridge/code'));from fewshot_fix import install
    install({l:proto['sources']['belebele_'+l]['path'] for l in ('en','sw')})
    legacy=H/'reproductions/sw_d2_20260921/eval_project/experiments/sw_fineweb2_aya_v1/scripts'
    c=read(ROOT/'work/benchmark_direction_20261026/bridge/legacy_config.json');c['batch_size']=1
    write(out/'config.json',c);job=next(j for j in c['tasks'] if j['name']==a.task);write(out/'job.json',dict(job=job,smoke=a.smoke))
    sys.path.insert(0,str(legacy));sys.argv=[str(legacy/'baseline_eval_worker.py'),'--config',str(out/'config.json'),'--model',str(modelpath),'--output',str(out)]
    runpy.run_path(str(legacy/'baseline_eval_worker.py'),run_name='__main__');expected=1 if a.smoke else 900
   else:
    os.environ['KNOWLEDGE_LANG']=a.task[-2:]
    import knowledge_tasks
    tasks=knowledge_tasks.TASKS_TABLE[:1] if a.smoke else knowledge_tasks.TASKS_TABLE
    from lighteval.__main__ import app
    sys.argv=['lighteval','accelerate',f'model_name={modelpath},batch_size=1,dtype=bfloat16,override_chat_template=true',
     ','.join(t.name+'|5' for t in tasks),'--custom-tasks',str(REPORT/'code/knowledge_tasks.py'),'--save-details','--output-dir',str(out)]
    if a.smoke:sys.argv+=['--max-samples','1']
    try:app()
    except SystemExit as e:
     if e.code not in (0,None):raise
    expected=1 if a.smoke else 14042
   rows=[r for p in out.glob('details/**/*.parquet') for r in pq.read_table(p).to_pylist()]
   assert len(rows)==expected and all(len(r['model_response']['logprobs'])==4 for r in rows)
   correct=sum(r['metric']['acc'] for r in rows);scores=dict(n=expected,correct=correct,ACC=100*correct/expected)
  elif a.task=='gsm8k':
   from lm_eval import evaluator
   from lm_eval.utils import handle_non_serializable
   result=evaluator.simple_evaluate(model='hf',model_args={'pretrained':str(modelpath),'tokenizer':str(BASE),'dtype':'bfloat16'},
    tasks=['gsm8k'],num_fewshot=5,batch_size=4,device='cuda:0',apply_chat_template=True,fewshot_as_multiturn=True,
    system_instruction='',gen_kwargs='do_sample=false,temperature=0.0,num_beams=1,max_gen_toks=1024',
    random_seed=42,numpy_random_seed=42,torch_random_seed=42,fewshot_random_seed=42,log_samples=True,limit=1 if a.smoke else None)
   samples=result['samples']['gsm8k'];groups={r['doc_id'] for r in samples};assert len(groups)==(1 if a.smoke else 1319)
   (out/'harness_results.json').write_text(json.dumps(result,ensure_ascii=False,default=handle_non_serializable))
   (out/'PREDICTIONS.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,default=handle_non_serializable)+'\n' for r in samples))
   m=result['results']['gsm8k'];scores=dict(n=len(groups),strict_ACC=100*m['exact_match,strict-match'],flexible_ACC=100*m['exact_match,flexible-extract'])
  else:
   from transformers import AutoModelForCausalLM,GenerationConfig
   from scoring import instruction,translation
   tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True);net=AutoModelForCausalLM.from_pretrained(modelpath,local_files_only=True,
    torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval();gen=GenerationConfig.from_pretrained(BASE,local_files_only=True)
   eos=gen.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
   source=[json.loads(s) for s in Path(proto['sources'][a.task]['path']).read_text().splitlines() if s.strip()]
   if a.task=='ifeval':source=[dict(r,id='ifeval:'+str(r['key']),task='ifeval') for r in source]
   if a.smoke:source=source[:4]
   records=[];cap=1280 if a.task=='ifeval' else 256
   with (out/'PREDICTIONS.jsonl').open('x') as stream,torch.inference_mode():
    for begin in range(0,len(source),4):
     batch=source[begin:begin+4];prompts=[tok.apply_chat_template([dict(role='user',content=r['prompt'])],tokenize=True,add_generation_prompt=True) for r in batch]
     width=max(map(len,prompts));assert width+cap<=net.config.max_position_embeddings
     ids=torch.tensor([[tok.pad_token_id]*(width-len(p))+p for p in prompts],device='cuda');mask=torch.tensor([[0]*(width-len(p))+[1]*len(p) for p in prompts],device='cuda')
     output=net.generate(input_ids=ids,attention_mask=mask,generation_config=gen,do_sample=False,num_beams=1,
      max_new_tokens=cap,pad_token_id=tok.pad_token_id,eos_token_id=eos,use_cache=True)[:,width:].cpu().tolist()
     for row,prompt,tokens in zip(batch,prompts,output):
      stop=next((i+1 for i,t in enumerate(tokens) if t in eos),len(tokens));tokens=tokens[:stop];text=tok.decode(tokens,skip_special_tokens=True).strip()
      value=instruction(row,text) if a.task=='ifeval' else translation(text,row['reference'])
      try:
       from langdetect import detect,DetectorFactory
       DetectorFactory.seed=0;lang=detect(text)
      except Exception:lang='und'
      record=dict(id=row['id'],task=row['task'],group_id=row.get('group_id',row['id']),model=a.model,prompt_token_ids=prompt,
       token_ids=tokens,text=text,scores=value,prompt_tokens=len(prompt),generated_tokens=len(tokens),capped=not tokens or tokens[-1] not in eos,
       output_language=lang,repeated_4grams=sum(v-1 for v in collections.Counter(tuple(text.split()[i:i+4]) for i in range(max(0,len(text.split())-3))).values()))
      records.append(record);stream.write(json.dumps(record,ensure_ascii=False)+'\n')
     stream.flush();write(out/'PROGRESS.json',dict(n=len(records),updated_unix=time.time()))
   if a.task=='ifeval':
    denominator=sum(len(r['scores']['instruction_pass']) for r in records)
    scores=dict(n=len(records),prompt_strict_ACC=100*sum(r['scores']['prompt_strict'] for r in records)/len(records),
     instruction_strict_ACC=100*sum(sum(r['scores']['instruction_pass']) for r in records)/denominator,instruction_denominator=denominator)
   else:
    from sacrebleu.metrics import CHRF
    scores={}
    for task in sorted({r['task'] for r in records}):
     here=[r for r in records if r['task']==task];stats=np.asarray([r['scores']['chrfpp_stats'] for r in here]).sum(0).tolist()
     scores[task]=dict(n=len(here),corpus_chrfpp=CHRF(char_order=6,word_order=2,beta=2)._compute_score_from_stats(stats).score)
   scores['capped']=sum(r['capped'] for r in records)
  write(out/'SCORES.json',scores);write(out/'DONE.json',dict(model=a.model,task=a.task,smoke=a.smoke,scores=scores,
   model_sha256=modelsha,peak_allocated_bytes=torch.cuda.max_memory_allocated(),finished_unix=time.time(),
   files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}))
except BaseException as e:
 if isinstance(e,SystemExit) and e.code==75:raise
 write(RUN/f'FAILED_eval_{a.model}_{stage}_{a.task}_{time.time_ns()}.json',dict(error=str(e),traceback=traceback.format_exc(),failed_unix=time.time()))
 raise
