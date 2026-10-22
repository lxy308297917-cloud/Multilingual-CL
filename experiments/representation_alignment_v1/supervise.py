"""Manually started, one-shot dependency supervisor; no retries, no startup service."""
import subprocess,sys,time,os
from common import *
QUEUE=HOME/'gpu-queue/gpu_queue.py';PREFIX=HOME/'ssu-migration/envs/cl';CONDA=HOME/'.local/miniforge3/bin/conda'
def start(name,stage,method=None):
 cmd=['python3',str(QUEUE),'--queue',name,'start','--script',str(ROOT/'run_stage.sh'),'--conda-prefix',str(PREFIX),'--conda-exe',str(CONDA),'--gpus','1','--interval-minutes','5','--log',str(HOME/'logs/representation_alignment_20261021'/(name+'.log')),'--',stage]
 if method:cmd+=['--method',method]
 subprocess.run(cmd,check=True)
def wait_for(paths,label):
 while True:
  state={'stage':label,'time':time.time(),'pid':os.getpid(),'waiting':[str(p) for p in paths if not p.exists()]};write(OUT/'SUPERVISOR_V3.json',state)
  if not state['waiting']:return
  # Queue failure is terminal for this coordinator; authorized diagnosis/recovery is a later explicit agent action.
  for d in (HOME/'gpu-queue/queues').glob('sw-align-*-20261021/state.json'):
   if d.parent.name in json.loads((ROOT/'RECOVERY_V2.json').read_text())['superseded_queues']:continue
   q=json.loads(d.read_text())
   if q.get('phase') in ('failed','error') or (q.get('job') or {}).get('exit_code') not in (None,0):raise RuntimeError('Alignment queue failed: '+str(d))
  if list((OUT/'technical_batched_aux_v3').glob('*/FAILED.json')) or list((TECH).glob('*/FAILED.json')) or list((OUT/'comparison_v3').glob('*/FAILED.json')):raise RuntimeError('Alignment worker failed; preserve outputs and review')
  if time.time()>__import__('datetime').datetime.fromisoformat('2026-10-22T20:00:00+08:00').timestamp():raise RuntimeError('Reporting deadline reached; preserve partial matching without efficacy claim')
  time.sleep(60)
def main():
 try:
  wait_for([OUT/'diagnostics/Base/DONE.json',OUT/'diagnostics/ExistingFullDense/DONE.json'],'diagnostics')
  if not (TECH/'RESTORE_VERIFIED.json').exists():raise RuntimeError('Requires completed deterministic-v2 real recovery gate; do not repeat valid technical checkpoints')
  start('sw-align-v3-auxcheck-20261021','aux_preflight');wait_for([OUT/'technical_batched_aux_v3/PREFLIGHT_DONE.json'],'batched_aux_preflight')
  for m in 'ABCD':start('sw-align-v3-'+m.lower()+'-20261021','train',m)
  wait_for([OUT/'comparison_v3'/m/'TRAINING_DONE.json' for m in 'ABCD'],'matched_training')
  start('sw-align-v3-dev-20261021','eval');wait_for([OUT/'COMPARISON.json'],'development')
  write(OUT/'SUPERVISOR_V3_DONE.json',{'status':'complete','time':time.time(),'comparison':str(OUT/'COMPARISON.json')})
 except BaseException as e:write(OUT/'SUPERVISOR_V3_FAILED.json',{'error':str(e),'time':time.time()});raise
if __name__=='__main__':main()
