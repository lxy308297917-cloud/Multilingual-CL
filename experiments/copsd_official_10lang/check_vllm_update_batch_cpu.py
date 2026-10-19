import ast,contextlib,copy,io,json,sys,types
from pathlib import Path
import torch
from transformers import AutoTokenizer
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--model', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
cli=parser.parse_args()
from vllm_update_batch import UpdateBatch
source=cli.source/'multilingual_opsd_trainer.py'
klass=next(x for x in ast.parse(source.read_text()).body if isinstance(x,ast.ClassDef) and x.name=='OPSDTrainer')
fn=next(x for x in klass.body if isinstance(x,ast.FunctionDef) and x.name=='_generate_on_policy_outputs_vllm')
class Sampling:
 def __init__(self,**kw): self.__dict__.update(kw)
 def __repr__(self): return repr(self.__dict__)
ns={'torch':torch,'SamplingParams':Sampling,'profiling_decorator':lambda f:f}
exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),ns)
generate=ns[fn.name]
tok=AutoTokenizer.from_pretrained(cli.model,local_files_only=True)
class Engine:
 def __init__(self):
  self.llm_engine=types.SimpleNamespace(model_executor=types.SimpleNamespace(is_sleeping=False)); self.calls=[];self.sleep_calls=0
 def generate(self,prompts,**kwargs):
  self.calls.append(len(prompts))
  out=[]
  for s in prompts:
   length=[3,2048,11][sum(s.encode())%3]
   ids=tok.encode(' reasoning '+s[-8:],add_special_tokens=False)
   out.append(types.SimpleNamespace(outputs=[types.SimpleNamespace(token_ids=(ids*((length//len(ids))+1))[:length])]))
  return out
 def sleep(self,level=1): self.sleep_calls+=1;self.llm_engine.model_executor.is_sleeping=True

def trainer():
 t=types.SimpleNamespace(processing_class=tok,accelerator=types.SimpleNamespace(device='cpu'),
  args=types.SimpleNamespace(gradient_accumulation_steps=32,max_length=20000,top_p=.95),
  reason_first=False,state=types.SimpleNamespace(global_step=0),_last_vllm_sync_step=0,
  vllm_mode='colocate',vllm_enable_sleep_mode=True,vllm_guided_decoding_regex=None,
  vllm_tensor_parallel_size=1,vllm_engine=Engine())
 t.get_batch_samples=lambda it,n,device: (list(it),None)
 t._wake_vllm_if_needed=lambda: setattr(t.vllm_engine.llm_engine.model_executor,'is_sleeping',False)
 return t
cfg=types.SimpleNamespace(max_new_tokens=2048,temperature=1.1,top_k=20)
comparisons=0
for size in [32,20]:
 inputs=[{'student_prompts':tok('Problem '+str(i)+' <think>\n',return_tensors='pt').input_ids} for i in range(size)]
 baseline=trainer();batched=trainer(); c=UpdateBatch(batched); c.install()
 batched.get_batch_samples(iter(inputs),size,'cpu')
 with contextlib.redirect_stdout(io.StringIO()):
  for sample in inputs:
   baseline._wake_vllm_if_needed(); a=generate(baseline,sample,cfg,tok.pad_token_id)
   batched._wake_vllm_if_needed(); b=generate(batched,sample,cfg,tok.pad_token_id)
   for x,y in zip(a,b):
    assert torch.equal(x,y) if isinstance(x,torch.Tensor) else x==y
   comparisons+=1
 assert batched.vllm_engine.calls==[size]
 assert baseline.vllm_engine.calls==[1]*size
 assert batched.vllm_engine.sleep_calls==1
 c.assert_drained(); c.detach()
# Fail closed if an optimizer update or prompt ordering changes.
for fault in ['step','prompt']:
 t=trainer();c=UpdateBatch(t);c.install(); t.get_batch_samples(iter(inputs[:2]),2,'cpu')
 if fault=='step': t.state.global_step=1
 prompt=c.prompts[0] if fault=='step' else 'WRONG'
 try: c.generate([prompt],Sampling(n=1,max_tokens=2048),False)
 except AssertionError: pass
 else: raise AssertionError('missing guard '+fault)
result={'check':'CPU actual official generation method, deterministic fake inference engine, real Qwen3 tokenizer','matching_microbatch_outputs':comparisons,'compared':['input_ids','attention_mask','labels','prompt_texts','completion_texts'],'group_sizes':[32,20],'generation_calls_per_group':1,'engine_sleep_calls_per_group':1,'stale_step_and_wrong_prompt_rejected':True,'gpu_training_validated':False,'scope':'Packing and mask parity for fixed completions only; no stochastic-output equivalence, speed or training-effectiveness claim.'}
p=cli.output
p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
