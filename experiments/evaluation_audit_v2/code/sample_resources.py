"""CPU-only bounded physical-GPU sampler for the remaining evaluation block."""
import subprocess,time,os
from settings import *
launch=json.loads((RUN/'GENERATION_LAUNCH.json').read_text());uuids={x['gpu_uuid'] for x in launch};workers={x['pid'] for x in launch};plan=json.loads((RUN/'PROTOCOL_FROZEN.json').read_text());count=0
with (RUN/'physical_gpu_samples.jsonl').open('x') as f:
    while time.time()<plan['hard_finish_unix']:
        try:
            text=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
            samples=[]
            for line in text.splitlines():
                uuid,mem,util=[s.strip() for s in line.split(',')]
                if uuid in uuids:samples.append({'uuid':uuid,'memory_used_mib':int(mem),'utilization_percent':int(util)})
            f.write(json.dumps({'unix':time.time(),'gpus':samples})+'\n');f.flush();count+=1
        except Exception as e:f.write(json.dumps({'unix':time.time(),'error':str(e)})+'\n');f.flush()
        alive=False
        for pid in workers:
            try:os.kill(pid,0);alive=True
            except ProcessLookupError:pass
        if not alive:break
        time.sleep(5)
write(REPORT/'PHYSICAL_SAMPLER_DONE.json',{'samples':count,'file_sha256':sha(RUN/'physical_gpu_samples.jsonl'),'sampling_seconds':5,'coverage':'Remaining generation workers only, started after initial evaluation began; does not reconstruct earlier mechanism peaks','GPU_context_created':False})
