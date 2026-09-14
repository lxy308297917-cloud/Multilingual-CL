"""Construct and gate the seven-method baseline training commands."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parents[2]
def command(c,method):
 settings=c['methods'][method];output=Path(c['output_root'])/method
 entry='run_mofo_bf16.py' if method=='mofo15' else 'main_bf16.py'
 cmd=[c['training_python'],'-u',str(R/'training/src'/entry)]
 fields={'dataset_path':c['training_data'],'model_name_or_path':c['base_model'],'tokenizer_name_or_path':c['tokenizer'],'output_dir':str(output),'logging_dir':str(output/'logs'),'seed':c['seed'],'data_seed':c['data_seed'],'max_steps':c['max_steps'],'per_device_train_batch_size':c['batch_size'],'gradient_accumulation_steps':c['gradient_accumulation_steps'],'learning_rate':c['learning_rate'],'weight_decay':c['weight_decay'],'warmup_ratio':c['warmup_ratio'],'lr_scheduler_type':c['lr_scheduler_type'],'max_grad_norm':c['max_grad_norm'],'adam_beta1':c['adam_beta1'],'adam_beta2':c['adam_beta2'],'adam_epsilon':c['adam_epsilon'],'optim':c['optimizer'],'eval_strategy':'no','save_strategy':c.get('save_strategy','no'),'logging_steps':50,'report_to':'none','cl_method':'none','gradient_checkpointing':str(c['gradient_checkpointing']).lower()}
 fields['gradient_checkpointing_kwargs']=json.dumps(c.get('gradient_checkpointing_kwargs',{'use_reentrant':False}))
 for k,v in fields.items():cmd+=['--'+k,str(v)]
 cmd+=['--do_train','--bf16']
 env={'CUDA_VISIBLE_DEVICES':str(c['gpu']),'TRAINABLE_LAYER_RANGE':''}
 if method in ['ssu_en_freeze50','hft50']:
  cmd+=['--do_hft','--freeze_ratio',str(settings['freeze_ratio']),'--freeze_strategy','ssu_based' if method=='ssu_en_freeze50' else 'hft_based','--freeze_seed',str(c['seed']),'--skip_embeddings_and_head','--freeze_chat_template_tokens']
  if method=='ssu_en_freeze50':cmd+=['--calibration_dataset_path',c['english_calibration'],'--num_calibration_samples',str(c['english_calibration_requested']),'--calibration_max_length',str(c['calibration_max_length'])]
 elif method=='current_top20_all_layers':
  cmd+=['--lassu_enable','--lassu_current_score_path',str(Path(c['output_root']).parent/'calibration/ig_top20.pt'),'--lassu_old_score_paths','','--lassu_old_shared_threshold','2','--lassu_old_specific_scale','0.0','--lassu_old_shared_scale','0.0','--lassu_current_shared_scale','1.0','--lassu_current_specific_scale','1.0','--lassu_others_scale','0.0']
 elif method=='lota90':
  cmd+=['--use_lota','--lota_sparsity',str(settings['sparsity']),'--lota_calibration_steps',str(settings['calibration_steps']),'--lota_grad_accum_steps',str(settings['calibration_grad_accum_steps']),'--lota_optimizer',settings['calibration_optimizer'],'--lota_skip_embeddings_and_head','--lota_verbose']
 elif method=='mofo15':cmd+=['--mofo_update_fraction',str(settings['fraction'])]
 elif method=='layers1_5_23_27_fft':env['TRAINABLE_LAYER_RANGE']=settings['zero_based_layer_ranges']
 return cmd,env

def calibrate_top20(c,config_path):
 import fcntl
 out=Path(c['output_root']).parent/'calibration';out.mkdir(parents=True,exist_ok=True)
 with (out/'selection.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (out/'TOP20_DONE.json').exists():raise RuntimeError('Selection already exists; inspect identity before reuse')
  import torch
  from datasets import load_from_disk
  from transformers import AutoModelForCausalLM,AutoTokenizer
  sys.path.insert(0,str(R/'training/src'))
  from utils.data_utils import create_calibration_dataloader
  from utils.model_utils import collect_lassu_column_scores
  torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed'])
  dataset=load_from_disk(c['training_data']);tokenizer=AutoTokenizer.from_pretrained(c['tokenizer'],local_files_only=True)
  loader=create_calibration_dataloader(c['training_data'],c['target_calibration_requested'],dataset,tokenizer)
  indices=list(loader.dataset.indices)
  samples=[{'index':int(i),'tokens':len(dataset[int(i)]['input_ids']),'input_ids_sha256':hashlib.sha256(json.dumps(dataset[int(i)]['input_ids']).encode()).hexdigest()} for i in indices]
  (out/'samples.json').write_text(json.dumps(samples,indent=2));(out/'config_at_selection.json').write_bytes(config_path.read_bytes())
  model=AutoModelForCausalLM.from_pretrained(c['base_model'],torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2',local_files_only=True).to('cuda')
  calls=[]
  hook=model.register_forward_hook(lambda *args:calls.append(1))
  scores=collect_lassu_column_scores(model,loader,num_calibration_samples=len(indices),top_ratio=c['methods']['current_top20_all_layers']['top_ratio'],save_path=str(out/'ig_top20.pt'),skip_embeddings_and_head=True)
  hook.remove()
  if len(calls)!=len(indices):raise RuntimeError('Calibration did not complete every selected sample')
  score_hash=hashlib.sha256((out/'ig_top20.pt').read_bytes()).hexdigest()
  record={'status':'complete','actual_samples':len(indices),'actual_forward_calls':len(calls),'effective_input_token_positions':sum(x['tokens'] for x in samples),'score_sha256':score_hash,'config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),'selection_code_sha256':hashlib.sha256((R/'training/src/utils/model_utils.py').read_bytes()).hexdigest(),'training_started':False}
  (out/'TOP20_DONE.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)

def file_sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()

def source_identity():
 paths=list((R/'training/src').rglob('*.py'))+[R/'scripts/src/train_cl.py',Path(__file__).resolve()]
 return {str(p):file_sha(p) for p in sorted(paths)}

def execute_training(c,a):
 import fcntl,shutil,time
 from copy import deepcopy
 actual=deepcopy(c);actual['gpu']=a.gpu if a.gpu is not None else c['gpu']
 if actual['gpu'] not in c['training_gpu_pool']:raise ValueError('GPU outside training pool')
 root=Path(c['output_root']);suite=root.parent
 if a.smoke:
  actual['max_steps']=3;actual['output_root']=str(suite/'training_smoke_nonreentrant');actual['save_strategy']='no'
 else:
  release=suite/'training_releases'/(a.method+'.json')
  if not release.exists():raise RuntimeError('Missing audited training release: '+str(release))
  evidence=json.loads(release.read_text())
  if evidence['config_sha256']!=file_sha(a.config) or evidence['code']!=source_identity():raise RuntimeError('Training identity changed after release')
  if not evidence.get('passed'):raise RuntimeError('Training release did not pass')
  for path,expected in evidence['inputs'].items():
   if file_sha(path)!=expected:raise RuntimeError('Training input changed: '+path)
 output=Path(actual['output_root'])/a.method;output.parent.mkdir(parents=True,exist_ok=True)
 locks=suite/'training_locks';locks.mkdir(exist_ok=True)
 with (locks/(a.method+'.lock')).open('a') as lock,(locks/('gpu'+str(actual['gpu'])+'.lock')).open('a') as gpu_lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if output.exists() and any(output.iterdir()):raise RuntimeError('Output already contains artifacts; inspect before any rerun: '+str(output))
  # Reserve every possible concurrent unfinished final save, plus 1 GiB.
  model_bytes=sum(p.stat().st_size for p in Path(c['base_model']).glob('*.safetensors'))
  unfinished=[name for name in c['methods'] if not (root/name/'TRAINING_DONE.json').exists()]
  reserved_models=len(c['training_gpu_pool']) if a.smoke else min(len(c['training_gpu_pool']),max(1,len(unfinished)))
  required=reserved_models*(model_bytes+128*(1<<20))+(1<<30)
  free=shutil.disk_usage(output.parent).free
  if free<required:raise RuntimeError(f'Insufficient final-save capacity: free={free}, required={required}; nothing deleted')
  output.mkdir(exist_ok=True);cmd,extra=command(actual,a.method)
  env=dict(os.environ,**extra,HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',AUTO_RESUME_FROM_CHECKPOINT='0',EXACT_SAVE_STEPS='',BASELINE_MASK_AUDIT='1')
  info={'method':a.method,'smoke':a.smoke,'config_sha256':file_sha(a.config),'code':source_identity(),'command':cmd,'environment':extra,'gpu':actual['gpu'],'steps':actual['max_steps'],'save_policy':'final weights only; interrupted training must restart from Base','storage_free_bytes':free,'storage_required_bytes':required,'storage_reserved_models':reserved_models,'started_unix':time.time()}
  (output/'EXPERIMENT_INFO.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n')
  with (output/'train.log').open('w') as log:
   child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,pass_fds=(lock.fileno(),gpu_lock.fileno()))
   (output/'RUNNING.json').write_text(json.dumps({'pid':os.getpid(),'child_pid':child.pid,'gpu':actual['gpu']}))
   rc=child.wait()
  if rc:
   (output/'FAILED.json').write_text(json.dumps({'returncode':rc}));raise RuntimeError('Training failed: '+str(output/'train.log'))
  weights=list(output.glob('*.safetensors'))
  if not weights:raise RuntimeError('Training exited without saved weights')
  result={'steps':actual['max_steps'],'smoke':a.smoke,'completed_unix':time.time(),'weights_sha256':{p.name:file_sha(p) for p in weights}}
  (output/'TRAINING_DONE.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--method',required=True);p.add_argument('--plan',action='store_true');p.add_argument('--smoke',action='store_true');p.add_argument('--gpu',type=int);p.add_argument('--calibrate-top20',action='store_true');a=p.parse_args();c=json.loads(a.config.read_text())
 if a.method not in c['methods']:raise ValueError('Method is outside the frozen seven-method suite')
 cmd,env=command(c,a.method)
 if a.calibrate_top20:
  if a.method!='current_top20_all_layers':raise ValueError('Calibration only supports target Top20 selection')
  os.environ['CUDA_VISIBLE_DEVICES']=str(c['gpu'])
  calibrate_top20(c,a.config);return
 if a.plan:
  print(json.dumps({'method':a.method,'command':cmd,'environment':env,'protocol_status':c['protocol_status'],'note':'Plan only; save policy and release gates must be verified before training'},ensure_ascii=False,indent=2));return
 execute_training(c,a)
if __name__=='__main__':main()
