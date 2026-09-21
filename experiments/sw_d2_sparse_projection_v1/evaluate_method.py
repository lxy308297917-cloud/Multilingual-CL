"""Released new-method finals through unchanged SW task implementations."""
import argparse,fcntl,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).parent;REPO=ROOT.parent.parent/'ssu-main'
sys.path.insert(0,str(ROOT.parent/'sw_update_utility_v1/prototype'));sys.path.insert(0,str(ROOT.parent/'reproduction'))
from safe_state import file_sha
from verified_cache import verify

def write(path,value):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));tmp.replace(path)
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--gpu',required=True,type=int);a=p.parse_args()
 if os.environ.get('SW_D2_METHOD_UNIFIED_EVAL_ENTRY')!='1':raise RuntimeError('Use scripts/evaluate_cl.py after the new-method route is released')
 cp=Path(a.config).resolve();c=json.loads(cp.read_text());release=json.loads((ROOT/'TRAINING_RELEASE.json').read_text())
 assert release['status']=='released' and release['configs'][str(cp)]==file_sha(cp)
 for name,h in {**release['code'],**release['base_files'],**release['evidence']}.items():assert file_sha(name)==h,name
 out=Path(c['output_dir']);done_path=out/'TRAINING_DONE.json';done=json.loads(done_path.read_text())
 assert done['config_sha256']==file_sha(cp) and done['manifest_sha256']==c['manifest_sha256']
 assert done['final_weights'],'No final weights recorded'
 model=(out/'final').resolve()
 for name,h in done['final_weights'].items():
  assert Path(name).resolve().parent==model and file_sha(name)==h,name
 root=ROOT/'evaluation'/c['method'];root.mkdir(parents=True,exist_ok=True)
 with (ROOT/'evaluation.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  evidence=verify('sw',str(model));write(root/'verified_before.json',evidence)
  if evidence['complete']:
   write(root/'DONE.json',{'model':str(model),'config_sha256':file_sha(cp),'training_done_sha256':file_sha(done_path),'verified_evidence':evidence,'time':time.time()});return
  gpu_lock=(ROOT/f'gpu{a.gpu}.lock').open('a');fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()
  busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines();assert uuids[a.gpu] not in busy,'GPU occupied'
  missing=set(evidence['missing']);commands=[]
  if missing-{'ppl','gsm8k'}:commands.append(['bash',str(REPO/'scripts/sw_eval.sh'),str(model),'--gpu',str(a.gpu)])
  if 'ppl' in missing:commands.append([c['training_python'],str(ROOT.parent/'sw_fineweb2_aya_v1/diagnostics/evaluate_cl.py'),str(model)])
  if 'gsm8k' in missing:commands.append(['bash',str(REPO/'scripts/gsm8k_eval.sh'),str(model)])
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),GSM8K_GPU=str(a.gpu))
  attempt=str(time.time_ns());write(root/('ATTEMPT_'+attempt+'.json'),{'config':str(cp),'config_sha256':file_sha(cp),'model':str(model),'gpu':a.gpu,'missing':sorted(missing),'commands':commands,'pid':os.getpid()})
  try:
   for command in commands:
    subprocess.run(command,env=env,check=True,pass_fds=(gpu_lock.fileno(),lock.fileno()))
   evidence=verify('sw',str(model));write(root/'verified_after.json',evidence);assert evidence['complete'],evidence['missing']
   write(root/'DONE.json',{'model':str(model),'config_sha256':file_sha(cp),'training_done_sha256':file_sha(done_path),'verified_evidence':evidence,'attempt':attempt,'time':time.time()})
  except Exception as e:
   write(root/('FAILED_'+attempt+'.json'),{'error':str(e),'traceback':traceback.format_exc(),'time':time.time()});raise
if __name__=='__main__':main()
