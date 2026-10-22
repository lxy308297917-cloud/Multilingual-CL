"""One manual CPU supervisor for the authorized endpoint only; no restart/retry services."""
from settings import *
import subprocess,time
start=time.time();log=H/'logs/infonce_strength_20261022/endpoint.log'
while not (RUN/'PIPELINE_DONE.json').exists():
 if (RUN/'PIPELINE_FAILED.json').exists():raise RuntimeError('Training pipeline failed; inspect preserved failure before any recovery')
 if time.time()-start>4*3600:raise RuntimeError('Bounded endpoint supervisor timeout')
 time.sleep(30)
# Parent training worker must exit before acquisition; no GPU held while waiting.
sys.path.insert(0,str(H/'gpu-queue'));import gpu_queue as q
uuid='GPU-6f39cd66-8deb-a8bb-1b2e-8b96432fa573'
for _ in range(20):
 snapshot=q.snapshot();reserved=q.other_queue_gpus();assigned=q.user_gpu_assignments(snapshot,reserved)
 if uuid not in reserved and len(assigned)<4 and q.idle(snapshot[uuid],{'max_idle_memory_mib':512,'max_idle_util_percent':5}):break
 time.sleep(15)
else:raise RuntimeError('Endpoint GPU unavailable; no force or silent resource change')
with log.open('w') as stream:subprocess.run([sys.executable,str(REPORT/'code/worker.py'),'--uuid',uuid,'--script','endpoint_pipeline.py'],stdout=stream,stderr=subprocess.STDOUT,check=True)
subprocess.run([sys.executable,str(REPORT/'code/analyze.py')],check=True)
write(RUN/'ANALYSIS_DONE.json',{'finished_unix':time.time()})
