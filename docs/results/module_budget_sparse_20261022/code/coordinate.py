"""One manually started CPU coordinator; use verified idle cards, never reserve empty cards or retry failures."""
from settings import *
import subprocess,time
sys.path.insert(0,str(H/'gpu-queue'));import gpu_queue as q
start=time.time();pending=['S-Attn','S-MLP'];live={};streams={};failed=False
while True:
 assert time.time()-start<18*3600,'Frozen18h walltime waiting/series bound'
 for group,info in list(live.items()):
  code=info['process'].poll()
  if code is not None:
   streams[group].close();del streams[group];del live[group]
   if code or not (RUN/('GROUP_DONE_'+group+'.json')).exists():failed=True;print('GROUP_FAILURE',group,code,flush=True)
 if (RUN/'GROUP_FAILED_S-All.json').exists():failed=True
 if not failed:
  for group in list(pending):
   if (RUN/('GROUP_DONE_'+group+'.json')).exists():pending.remove(group);continue
   assert not (OUT/group).exists(),'Existing partial scientific run requires explicit diagnosed recovery'
   snapshot=q.snapshot();reserved=q.other_queue_gpus();assigned=q.user_gpu_assignments(snapshot,reserved)
   if len(assigned)>=8:continue
   occupied={v['uuid'] for v in live.values()}
   # D1 owns GPU3 through its downstream endpoint, including its CPU handoff gap.
   if not (ROOT/'work/infonce_strength_v1/ANALYSIS_DONE.json').exists():occupied.add('GPU-6f39cd66-8deb-a8bb-1b2e-8b96432fa573')
   if not (RUN/'GROUP_DONE_S-All.json').exists():occupied.add('GPU-4d65516e-eaa6-0d50-ff59-a1c8d4a75df8')
   candidates=[uuid for uuid,g in snapshot.items() if uuid not in reserved|occupied and q.idle(g,{'max_idle_memory_mib':512,'max_idle_util_percent':5})]
   if not candidates:continue
   uuid=candidates[0];env=dict(os.environ,MODULE_BUDGET_GROUP=group);log=H/'logs/module_budget_sparse_20261022'/(group+'.log');streams[group]=log.open('w');process=subprocess.Popen([sys.executable,str(REPORT/'code/worker.py'),'--uuid',uuid,'--script','group_pipeline.py'],stdout=streams[group],stderr=subprocess.STDOUT,env=env);live[group]={'process':process,'uuid':uuid};pending.remove(group);write(RUN/'claims'/(group+'.json'),{'pid':process.pid,'group':group,'uuid':uuid,'started_unix':time.time(),'account_assigned_before':sorted(assigned),'manual_one_shot':True});print('LAUNCH',group,uuid,process.pid,flush=True)
 done=[g for g in ('S-All','S-Attn','S-MLP') if (RUN/('GROUP_DONE_'+g+'.json')).exists()];write(RUN/'SERIES_PROGRESS.json',{'queued':pending,'running':{g:{'pid':v['process'].pid,'uuid':v['uuid']} for g,v in live.items()},'done':done,'failed':failed,'updated_unix':time.time(),'temporary_account_cap':8,'no_hold_or_other_process_interruption':True})
 if failed:
  if not live:raise RuntimeError('A group failed; retain logs, diagnose before authorized recovery. No automatic scientific retry.')
 elif len(done)==3:break
 time.sleep(30)
subprocess.run([sys.executable,str(REPORT/'code/analyze.py')],check=True);write(RUN/'ANALYSIS_DONE.json',{'finished_unix':time.time()})
