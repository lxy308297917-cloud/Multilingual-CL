"""CPU-only physical-memory sampling; exits after this one endpoint pipeline."""
from settings import *
import subprocess,time
out=RUN/'RESOURCE_SAMPLES.jsonl';start=time.time()
with out.open('a') as f:
 while time.time()-start<6*3600:
  if (RUN/'ANALYSIS_DONE.json').exists() or (RUN/'PIPELINE_FAILED.json').exists():break
  value=subprocess.check_output(['nvidia-smi','--id=GPU-6f39cd66-8deb-a8bb-1b2e-8b96432fa573','--query-gpu=memory.used,utilization.gpu,power.draw','--format=csv,noheader,nounits'],text=True).strip().split(',')
  phase='endpoint' if (RUN/'PIPELINE_DONE.json').exists() else 'training'
  f.write(json.dumps({'unix':time.time(),'phase':phase,'gpu_uuid':'GPU-6f39cd66-8deb-a8bb-1b2e-8b96432fa573','physical_memory_mib':int(value[0]),'util_percent':int(value[1]),'power_watt':float(value[2])})+'\n');f.flush();time.sleep(30)
