"""One isolated evaluation task; invoked only by the unified controller."""
import argparse, importlib.util, json, os, runpy, sys
from pathlib import Path
from run_baseline_eval import R,sha,write

def install_generation(c,j,out):
 import torch,random,numpy as np
 from transformers.generation.utils import GenerationMixin
 random.seed(j['seed']);np.random.seed(j['seed']);torch.manual_seed(j['seed']);torch.cuda.manual_seed_all(j['seed'])
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 original=GenerationMixin.generate
 first=True
 params=dict(c['sampling'] if j['kind']=='lighteval' else c['general_sampling'])
 params['max_new_tokens']=128 if j['kind']=='lighteval' else c['general_max_tokens'][j['name']]
 params['num_return_sequences']=1
 def generate(self,*args,**kwargs):
  nonlocal first
  if first:
   torch.manual_seed(j['seed']);torch.cuda.manual_seed_all(j['seed']);first=False
  from transformers import GenerationConfig
  kwargs['generation_config']=GenerationConfig.from_pretrained(c['base_model'],local_files_only=True)
  kwargs.update(params)
  tensor=kwargs.get('input_ids',args[0] if args else None)
  if tensor is None:tensor=kwargs.get('inputs')
  record={'seed':j['seed'],'parameters':params,'generation_config':self.generation_config.to_dict(),'input_ids':tensor.detach().cpu().tolist() if tensor is not None else None}
  with (out/'effective_generation.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
  return original(self,*args,**kwargs)
 GenerationMixin.generate=generate

def light(c,j,model,out,smoke):
 from lighteval.models.transformers.transformers_model import TransformersModel
 for name in ['greedy_until','loglikelihood','loglikelihood_rolling']:
  f=getattr(TransformersModel,name)
  if hasattr(f,'__wrapped__'):setattr(TransformersModel,name,f.__wrapped__)
 # Export the selected legacy task through the one public custom-task registry.
 os.environ['BASELINE_EVAL_TASK_SOURCE']=j['source']
 task=j['task']
 if '{subject}' in task:
  from eval_plnd_downstream import load_subjects
  subjects=load_subjects();task=','.join(task.format(subject=s) for s in (subjects[:1] if smoke else subjects))
 sys.path.insert(0,str(R/'evaluation'));sys.path.insert(0,str(R/'evaluation/src'))
 from lighteval.__main__ import app
 sys.argv=['lighteval','accelerate',f'model_name={model},batch_size={c["batch_size"]},dtype={c["dtype"]},override_chat_template=true',task,'--custom-tasks',str(R/'evaluation/custom_multilingual_tasks.py'),'--save-details','--output-dir',str(out)]
 limit=1 if smoke else j['max_samples']
 if limit is not None:sys.argv+=['--max-samples',str(limit)]
 try:app()
 except SystemExit as exc:
  if exc.code not in (0,None):raise
 files=list(out.rglob('results_*.json'))
 if len(files)!=1:raise RuntimeError('expected one result file, got '+str(files))
 data=json.loads(files[0].read_text());results=data['results']
 metric='chrfpp_sample' if j['generative'] else 'acc'
 vals=[]
 for name,values in results.items():
  if name=='all':continue
  # Keep raw task metrics; choose the unnormalized accuracy field only.
  if metric in values:vals.append(values[metric])
 if not vals:raise RuntimeError('missing '+metric+' in '+str(results))
 score=sum(vals)/len(vals)
 if not j['generative']:score*=100
 if not list(out.rglob('*.parquet')):raise RuntimeError('missing sample details')
 return {j['name']:score}

def general(c,j,model,out,smoke):
 script=R/'evaluation/src'/(j['name']+'.py')
 sys.argv=[str(script),'--model_name_or_path',str(model),'--output_dir',str(out),'--apply_chat_template','--batch_size',str(c['batch_size']),'--max_new_tokens',str(c['general_max_tokens'][j['name']])]
 if j['name']=='ifeval':sys.argv+=['--dataset_jsonl',c['ifeval_data'],'--official_files_dir',str(R/'evaluation/src/utils')]
 else:sys.argv+=['--test_arrow',c['gsm8k_data'],'--mode','gsm8k_cot','--save_raw_prompt']
 if smoke:sys.argv+=['--max_samples','1']
 runpy.run_path(str(script),run_name='__main__')
 v=json.loads((out/'summary.json').read_text())
 keys={'prompt_level_strict_acc':'ifeval_prompt_strict','inst_level_strict_acc':'ifeval_instruction_strict'} if j['name']=='ifeval' else {'strict_match_acc':'gsm8k_strict','flexible_extract_acc':'gsm8k_flexible'}
 return {name:v[k]*100 for k,name in keys.items()}

def ppl(c,model,out,smoke):
 script=R/'analysis/lape_qwen_preexperiment.py'
 sys.argv=[str(script),'evaluate','--model',str(model),'--conditions','none','--data-root',c['ppl_data_root'],'--eval-tokens',str(512 if smoke else c['ppl_eval_tokens']),'--batch-size',str(c['batch_size']),'--output',str(out/'ppl.csv')]
 sys.path.insert(0,str(R/'analysis'));runpy.run_path(str(script),run_name='__main__')
 rows=json.loads((out/'ppl.json').read_text());result={}
 for r in rows:
  for k in ['ppl','nll']:result[r['eval_language']+'_'+k]=r[k]
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path);p.add_argument('--model',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
 c=json.loads(a.config.read_text());ident=json.loads((a.output/'job.json').read_text());j=ident['job'];smoke=ident['smoke']
 import importlib.metadata,platform
 write(a.output/'environment.json',{'python':platform.python_version(),'packages':{n:importlib.metadata.version(n) for n in ['torch','transformers','datasets','accelerate']}})
 # Every backend, including PPL, gets the same tokenizer and chat template.
 from transformers import AutoTokenizer
 original_tokenizer=AutoTokenizer.from_pretrained
 def fixed_tokenizer(path,*args,**kwargs):
  kwargs.pop('revision',None)
  kwargs['local_files_only']=True
  return original_tokenizer(c['base_model'],*args,**kwargs)
 AutoTokenizer.from_pretrained=staticmethod(fixed_tokenizer)
 write(a.output/'tokenizer_identity.json',{'effective_path':c['base_model'],'policy':'fixed Base tokenizer for all models; token ID mapping checked by controller'})
 if j['generative']:install_generation(c,j,a.output)
 if j['kind']=='lighteval':scores=light(c,j,a.model,a.output,smoke)
 elif j['kind']=='general':scores=general(c,j,a.model,a.output,smoke)
 else:scores=ppl(c,a.model,a.output,smoke)
 if j['generative'] and not (a.output/'effective_generation.jsonl').exists():raise RuntimeError('no audited generation calls')
 write(a.output/'scores.json',scores)
if __name__=='__main__':main()
