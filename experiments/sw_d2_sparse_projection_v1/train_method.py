"""New-method runner, fail-closed until recipe selection and release are frozen.

Formal calls must be routed through scripts/train_cl.py. The route is installed;
formal execution remains gated by recipe selection and the training release.
"""
import argparse,collections,fcntl,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).parent;REPO=ROOT.parent.parent/'ssu-main'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT.parent/'sw_update_utility_v1/prototype'))
from method_optimizers import ElementwiseProtectedAdamW,ProjectedRowAdamW
from activation_bases import estimate_bases,basis_identity,edge_matrices,LAYERS
from safe_state import file_sha

def write(path,value):
 path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
def rows(path):
 with open(path) as f:return [json.loads(x) for x in f]
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
 if os.environ.get('SW_D2_METHOD_UNIFIED_ENTRY')!='1':raise RuntimeError('Use the released scripts/train_cl.py route')
 cp=Path(a.config).resolve();c=json.loads(cp.read_text());release=json.loads((ROOT/'TRAINING_RELEASE.json').read_text())
 assert release['status']=='released' and release['configs'][str(cp)]==file_sha(cp)
 for path,h in {**release['code'],**release['base_files'],**release['evidence']}.items():assert file_sha(path)==h,path
 selection=json.loads(Path(c['recipe_selection']).read_text());assert selection['selected_recipe']==c['recipe']
 out=Path(c['output_dir']);assert out.resolve().parent==Path(c['checkpoint_root']).resolve();out.mkdir(parents=True,exist_ok=True)
 lock=(out/'train.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (out/'TRAINING_DONE.json').exists():
  done=json.loads((out/'TRAINING_DONE.json').read_text());assert done['config_sha256']==file_sha(cp)
  for path,h in done['final_weights'].items():assert file_sha(path)==h
  return
 if (out/'FAILED.json').exists() and not a.resume:raise RuntimeError('Review failed attempt before explicit resume')
 assert shutil.disk_usage(out).free>=c['minimum_start_free_bytes']
 gpu_lock=(ROOT/('gpu'+str(a.gpu)+'.lock')).open('a');fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()
 busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines();assert uuid[a.gpu] not in busy
 started=time.time()
 os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
 import torch,platform,importlib.metadata as metadata
 from transformers import Trainer,TrainingArguments,TrainerCallback,AutoTokenizer,AutoModelForCausalLM,set_seed
 from torch.utils.data import SequentialSampler
 from torch.utils.tensorboard import SummaryWriter
 from torch_optimizer import TorchBlockAdamW
 from dense_optimizer import DenseFP32AdamW
 from raw_gradient_capture import RawGradientCaptureMixin
 from safe_resume import SafeResumeMixin
 from source_gradients import source_gradients,per_example_loss
 from calibration_schedule import CalibrationSchedule
 environment={'python':platform.python_version(),**{k:metadata.version(k) for k in c['training_environment'] if k!='python'}}
 assert environment==c['training_environment'];assert Path(sys.executable).resolve()==Path(c['training_python']).resolve()
 assert file_sha(c['manifest'])==c['manifest_sha256'];manifest=json.loads(Path(c['manifest']).read_text())
 for k in ['train_pool','dev_pool','indices']:assert file_sha(manifest[k])==manifest[k+'_sha256']
 pool=rows(manifest['train_pool']);dev=rows(manifest['dev_pool']);indices=json.loads(Path(manifest['indices']).read_text())
 assert len(indices)%2==0 and max(collections.Counter(indices).values())<=3
 prefix=[0];supervised=[0]
 for i in indices:
  x=pool[i];assert len(x['input_ids'])==len(x['labels'])<=c['max_length']
  prefix.append(prefix[-1]+len(x['input_ids']));supervised.append(supervised[-1]+sum(v!=-100 for v in x['labels'][1:]))
 assert prefix[-1]==manifest['budget_nonpadding_input_tokens'] and c['input_token_budget']<=prefix[-1]<c['input_token_budget']+2*c['max_length']
 assert not ({x['dedup_group'] for x in pool}&{x['dedup_group'] for x in dev})
 set_seed(c['seed']);tok=AutoTokenizer.from_pretrained(c['base_model'],local_files_only=True)
 def collate(items):
  length=max(len(x['input_ids']) for x in items)
  return {'input_ids':torch.tensor([x['input_ids']+[tok.pad_token_id]*(length-len(x['input_ids'])) for x in items]),'labels':torch.tensor([x['labels']+[-100]*(length-len(x['labels'])) for x in items]),'attention_mask':torch.tensor([[1]*len(x['input_ids'])+[0]*(length-len(x['input_ids'])) for x in items])}
 model=AutoModelForCausalLM.from_pretrained(c['base_model'],local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa');model.config.use_cache=False
 all_named=dict(model.named_parameters());candidate={n:p for n,p in all_named.items() if '.layers.' in n and p.ndim==2}
 assert len(candidate)==196 and sum(p.numel() for p in candidate.values())==1310195712
 calibration_rows=[x for x in rows(c['calibration_file']) if x['split']=='scoring']
 assert file_sha(c['calibration_file'])==c['calibration_sha256']
 pools={task:[x for x in calibration_rows if x['task']==task] for task in ['instruction','reading','math']}
 assert all(len(v)==256 for v in pools.values())
 writer=SummaryWriter(str(out/'calibration_tensorboard'));attempt=str(time.time_ns());trace=(out/('calibration_'+attempt+'.jsonl')).open('a')
 def select_window(step):
  cycle=step//200
  return [pools[k][(cycle*n+i)%256] for k,n in [('instruction',11),('reading',11),('math',10)] for i in range(n)]
 def provider(step):
  window=select_window(step)
  # One task-balanced set; source_gradients returns an overall scaled gradient.
  # Matrix-wise normalization makes the common scale immaterial.
  batches={'english':[collate([x]) for x in window]}
  grads,stats=source_gradients(model,candidate,batches,next(model.parameters()).device,torch.bfloat16)
  stats.update(step=step,ids=[x['id'] for x in window]);trace.write(json.dumps(stats)+'\n');trace.flush()
  writer.add_scalar('source/english_loss',stats['task_mean_loss']['english'],step)
  return grads,stats
 method=c['method']
 if method=='fft_fp32_matched':optimizer=DenseFP32AdamW(all_named,lr=c['learning_rate'],weight_decay=c['weight_decay'])
 elif method in ('element_protected','element_unprotected'):
  for n,p in all_named.items():p.requires_grad_(n in candidate)
  optimizer=ElementwiseProtectedAdamW(candidate,source_provider=provider if method=='element_protected' else None,
       protected=method=='element_protected',lr=c['learning_rate'],wd=c['weight_decay'],fraction=.15,
       mask_interval=20,source_interval=200)
 elif method in ('edge_projected_rows','edge_rows_no_projection'):
  edge={n:p for n,p in candidate.items() if any(f'.layers.{i}.' in n for i in LAYERS)}
  for n,p in all_named.items():p.requires_grad_(n in edge)
  model.to('cuda')
  basis_file=Path(c['basis_file'])
  if not basis_file.exists():raise FileNotFoundError('Frozen basis required before training')
  from safetensors.torch import load_file
  bases={n+'.weight':u for n,u in load_file(str(basis_file),device='cpu').items()}
  assert file_sha(basis_file)==c['basis_sha256']
  optimizer=ProjectedRowAdamW(edge,bases,lr=c['learning_rate'],wd=c['weight_decay'],
       alpha=.5 if method=='edge_projected_rows' else 0.,row_fraction=.5,mask_interval=20)
 else:raise ValueError('Unknown method: '+method)
 class MethodTrainer(RawGradientCaptureMixin,SafeResumeMixin,Trainer):
  def _get_train_sampler(self,train_dataset=None):return SequentialSampler(self.train_dataset if train_dataset is None else train_dataset)
  def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
   labels=inputs.pop('labels')
   if not bool((labels[:,1:]!=-100).any(dim=1).all()):raise ValueError('Sample has no shifted supervised token')
   outputs=model(**inputs,use_cache=False);loss=per_example_loss(outputs.logits,labels).mean()
   if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite training/development loss')
   return (loss,outputs) if return_outputs else loss
  def _save_checkpoint(self,*args,**kwargs):
   with (ROOT/'checkpoint_write.lock').open('a') as guard:
    fcntl.flock(guard,fcntl.LOCK_EX);return super()._save_checkpoint(*args,**kwargs)
 steps=len(indices)//2;prior=0
 if (out/'PROGRESS.json').exists():prior=json.loads((out/'PROGRESS.json').read_text())['total_elapsed_seconds']
 class Progress(TrainerCallback):
  def on_step_end(self,args,state,control,**kwargs):
   n=state.global_step*2;previous=prefix[max(0,n-2)]
   if any(previous<t*prefix[-1]/5<=prefix[n] for t in range(1,5)):control.should_evaluate=True;control.should_save=True
   if state.global_step%20==0:
    masks=getattr(optimizer,'masks',{})
    selected=sum(int(m.sum()) for m in masks.values())
    total=sum(m.numel() for m in masks.values())
    write(out/'PROGRESS.json',{'step':state.global_step,'total_steps':steps,'input_tokens_seen':prefix[n],
         'supervised_tokens_seen':supervised[n],'selected_units':selected,'candidate_units':total,
         'total_elapsed_seconds':prior+time.time()-started,'pid':os.getpid(),'gpu':a.gpu})
    if total:writer.add_scalar('mechanism/selected_unit_fraction',selected/total,state.global_step)
 keys=sorted({(x['task'],x['source']) for x in dev});eval_data={f'{task}_{i}':[x for x in dev if (x['task'],x['source'])==(task,source)] for i,(task,source) in enumerate(keys)}
 args=TrainingArguments(output_dir=str(out),max_steps=steps,per_device_train_batch_size=1,per_device_eval_batch_size=1,gradient_accumulation_steps=2,learning_rate=c['learning_rate'],lr_scheduler_type='cosine',warmup_ratio=.05,max_grad_norm=1.,bf16=True,gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},seed=c['seed'],data_seed=c['seed'],eval_strategy='no',save_strategy='no',save_total_limit=1,logging_steps=20,report_to=['tensorboard'],logging_dir=str(out/'tensorboard'),prediction_loss_only=True,remove_unused_columns=False,load_best_model_at_end=False,save_safetensors=True)
 callbacks=[Progress()]
 trainer=MethodTrainer(model=model,args=args,train_dataset=[pool[i] for i in indices],eval_dataset=eval_data,data_collator=collate,processing_class=tok,optimizers=(optimizer,None),callbacks=callbacks);trainer.model_accepts_loss_kwargs=False
 checkpoints=sorted(out.glob('checkpoint-*'),key=lambda x:int(x.name.split('-')[-1]))
 if checkpoints and not a.resume:raise RuntimeError('Explicit resume required')
 if a.resume and not checkpoints:raise RuntimeError('No checkpoint to resume')
 write(out/'EXPERIMENT_INFO.json',{'config':str(cp),'config_sha256':file_sha(cp),'manifest_sha256':c['manifest_sha256'],'method':method,'recipe':c['recipe'],'pid':os.getpid(),'gpu':a.gpu,'phase':'method_comparison','source_trace':str(trace.name),'source_gradients':'32 example calibration at current weights','target_scoring_gradients':'target Adam first moment','moments':'FP32','initialization':'independent Base'})
 try:
  if not a.resume:trainer.evaluate(metric_key_prefix='eval_initial')
  trainer.train(resume_from_checkpoint=str(checkpoints[-1]) if a.resume else None);trainer.evaluate(metric_key_prefix='eval_final')
  trainer._save_checkpoint(model,None);trainer.save_model(str(out/'final'));trainer.save_state()
  write(out/'TRAINING_DONE.json',{'config_sha256':file_sha(cp),'manifest_sha256':c['manifest_sha256'],'global_step':trainer.state.global_step,'input_tokens':prefix[-1],'supervised_tokens':supervised[-1],'total_gpu_hours':(prior+time.time()-started)/3600,'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated(),'final_weights':{str(p):file_sha(p) for p in (out/'final').glob('*.safetensors')}})
 except BaseException as e:
  write(out/'FAILED.json',{'error':str(e),'traceback':traceback.format_exc(),'time':time.time()});raise
 finally:trace.close();writer.close()
if __name__=='__main__':main()
