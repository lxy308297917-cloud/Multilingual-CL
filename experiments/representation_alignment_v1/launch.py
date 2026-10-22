"""One-shot entrypoint with frozen release identity, no-overwrite and bounded GPU budget."""
import argparse,subprocess,sys,time,fcntl,os
from common import *
def check_gpu():
 pids=subprocess.check_output(['nvidia-smi','--id='+os.environ['CUDA_VISIBLE_DEVICES'],'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines()
 if any(p.strip() and int(p)!=os.getpid() for p in pids):raise RuntimeError('GPU has another compute process; refuse launch')
def ledger(update):
 OUT.mkdir(parents=True,exist_ok=True)
 with (OUT/'budget.lock').open('a+') as f:
  fcntl.flock(f,fcntl.LOCK_EX);p=OUT/'BUDGET.json';d=json.loads(p.read_text()) if p.exists() else {'entries':[]};update(d);write(p,d)
def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['diagnostics','technical','aux_preflight','train','eval']);p.add_argument('--method',choices=list('ABCD'));a=p.parse_args();release=json.loads((ROOT/'RELEASE.json').read_text())
 import importlib.metadata
 for package,expected in release['software'].items():
  if importlib.metadata.version(package)!=expected:raise RuntimeError('Frozen software version changed: '+package)
 for name,expected in release['hashes'].items():
  if sha(ROOT/name)!=expected:raise RuntimeError('Frozen runtime/data changed: '+name)
 check_gpu();protocol=json.loads((ROOT/'protocol.json').read_text());cap={'diagnostics':2,'technical':1,'aux_preflight':.5,'train':3,'eval':7.5}[a.stage];key=a.stage+('_'+a.method if a.method else '')+'_'+str(int(time.time()));start=time.monotonic()
 def reserve(d):
  used=sum(e.get('gpu_hours',e['reserved_gpu_hours']) for e in d['entries'])
  if used+cap>protocol['total_gpu_hours_ceiling']:raise RuntimeError('Total preregistered GPUh exhausted/reserved')
  d['entries'].append({'id':key,'stage':a.stage,'method':a.method,'reserved_gpu_hours':cap,'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES'],'started_unix':time.time(),'status':'running'})
 ledger(reserve);ok=False
 try:
  def run(script,*args):subprocess.run([str(ROOT.parents[3]/'work/representation_alignment/plot-env/bin/python') if script=='plot.py' else sys.executable,str(ROOT/script),*map(str,args)],check=True,timeout=max(1,cap*3600-(time.monotonic()-start)))
  if a.stage=='diagnostics':
   run('diagnose.py');run('diagnose.py','--name','ExistingFullDense','--model',HOME/'models/checkpoint/sw_recipe_fft_20261010/aya80_mt20/final');run('plot.py')
  elif a.stage=='technical':
   for method in 'ABCD':
    extra=[] if method=='A' else ['--smoke-no-checkpoint']
    run('train.py','--method',method,'--smoke','--output',TECH/method,*extra)
   # Same main trajectory before the first auxiliary event must be bitwise reproducible.
   trace={m:rows(TECH/m/'STEPS.jsonl') for m in 'ABCD'}
   for m in 'BCD':
    assert [v['main_ce'] for v in trace[m]]==[v['main_ce'] for v in trace['A']]
    assert [v['grad_norm'] for v in trace[m][:3]]==[v['grad_norm'] for v in trace['A'][:3]]
   # A live optimizer/RNG restore followed by new steps, compared with uninterrupted8-step run.
   run('train.py','--method','A','--smoke','--smoke-steps',8,'--resume',TECH/'A/checkpoint_000004','--output',TECH/'A_restored')
   run('train.py','--method','A','--smoke','--smoke-steps',8,'--output',TECH/'A_continuous')
   from safetensors.torch import load_file
   x=load_file(str(TECH/'A_restored/checkpoint_000008/model/model.safetensors'));y=load_file(str(TECH/'A_continuous/checkpoint_000008/model/model.safetensors'));assert x.keys()==y.keys() and all(torch.equal(x[k],y[k]) for k in x);del x,y
   x=torch.load(TECH/'A_restored/checkpoint_000008/state.pt',map_location='cpu',mmap=True,weights_only=False);y=torch.load(TECH/'A_continuous/checkpoint_000008/state.pt',map_location='cpu',mmap=True,weights_only=False)
   assert x['scheduler']==y['scheduler']
   for k in x['optimizer']['state']:
    for name in ('step','exp_avg','exp_avg_sq'):assert torch.equal(x['optimizer']['state'][k][name],y['optimizer']['state'][k][name])
   write(TECH/'RESTORE_VERIFIED.json',{'model_bitwise_equal':True,'optimizer_bitwise_equal':True,'scheduler_equal':True,'continued_steps':4})
  elif a.stage=='aux_preflight':
   assert (TECH/'RESTORE_VERIFIED.json').exists()
   folder=OUT/'technical_batched_aux_v3';report={}
   stress=rows(TECH/'A/STEPS.jsonl');profile=rows(OUT/'comparison/A/STEPS.jsonl');baseline=float(np.mean([v['seconds'] for v in profile[1:]]))
   for method in 'BCD':
    run('train.py','--method',method,'--smoke','--smoke-no-checkpoint','--output',folder/method)
    trace=rows(folder/method/'STEPS.jsonl');assert [v['main_ce'] for v in trace]==[v['main_ce'] for v in stress]
    projected=float((baseline+max(0.,trace[3]['seconds']-stress[3]['seconds'])/4)*4293);report[method]={'projected_seconds':projected,'passed':bool(projected<10800*.9),'aux_batch':4,'real_long_sample_test':True}
    assert report[method]['passed'],'Auxiliary batch4 still fails original per-model cost ceiling'
   write(folder/'PREFLIGHT_DONE.json',report)
  elif a.stage=='train':
   assert (TECH/'RESTORE_VERIFIED.json').exists() and (OUT/'technical_batched_aux_v3/PREFLIGHT_DONE.json').exists();run('train.py','--method',a.method,'--output',OUT/'comparison_v3'/a.method)
  else:
   for name in ['Base']+list('ABCD'):
    model=BASE if name=='Base' else Path(json.loads((OUT/'comparison_v3'/name/'TRAINING_DONE.json').read_text())['final_model'])
    if (OUT/'development'/name/'DONE.json').exists():continue
    run('evaluate.py','--model',model,'--name',name)
    if name!='Base':run('diagnose.py','--model',model,'--name','align-'+name)
   run('plot.py');run('compare.py')
  ok=True
 finally:
  def finish(d):
   e=next(e for e in d['entries'] if e['id']==key);e['gpu_hours']=(time.monotonic()-start)/3600;e['status']='complete' if ok else 'failed';e['finished_unix']=time.time()
  ledger(finish)
if __name__=='__main__':main()
