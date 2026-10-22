"""Manual one-shot worker; respect shared queue lock, account cap and owner."""
import argparse,os,sys,time,subprocess,fcntl,runpy,gc,traceback
from settings import *
def main():
    p=argparse.ArgumentParser();p.add_argument('--uuid',required=True);a=p.parse_args()
    sys.path.insert(0,str(H/'gpu-queue'));import gpu_queue as q
    # Acquisition uses the same lock as existing named queues. Never kill any process.
    with (H/'gpu-queue/reservation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for _ in range(2):
            gpus=q.snapshot();reserved=q.other_queue_gpus();assigned=q.user_gpu_assignments(gpus,reserved)
            assert len(assigned)<4,(assigned,'No account slot')
            assert a.uuid not in reserved and q.idle(gpus[a.uuid],{'max_idle_memory_mib':512,'max_idle_util_percent':5}),'Assigned GPU no longer idle'
            time.sleep(1)
        os.environ['CUDA_VISIBLE_DEVICES']=a.uuid
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        import torch
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.cuda.init();probe=torch.empty(1,device='cuda');torch.cuda.synchronize();del probe
        write(RUN/('WORKER_'+a.uuid+'.json'),{'pid':os.getpid(),'gpu_uuid':a.uuid,'started_unix':time.time(),'account_assigned_before':sorted(assigned),'account_cap':4})
    plan=json.loads((RUN/'PROTOCOL_FROZEN.json').read_text())
    jobs=[('evaluate_v2.py',n) for n in ['Base','B','C','D','A']]+[('mechanism_v2.py',n) for n in ['Base','B','C','D','A','B-mid','C-mid','D-mid']]+[('teacher_identity_probe.py','Base')]
    while True:
        if time.time()>plan['stop_new_jobs_unix']:break
        chosen=None
        with (RUN/'dispatch.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            for script,name in jobs:
                if script=='evaluate_v2.py' and not (RUN/'ACCEPTANCE_FROZEN.json').exists():continue
                receipt=RUN/'claims'/(script+'_'+name+'.json')
                if not receipt.exists():
                    write(receipt,{'script':script,'name':name,'pid':os.getpid(),'gpu_uuid':a.uuid,'started_unix':time.time()});chosen=script,name;break
        if chosen is None:
            if not (RUN/'ACCEPTANCE_FROZEN.json').exists():
                # All independent probes claimed; no idle GPU reservation while waiting.
                break
            break
        script,name=chosen;sys.argv=[script,'--name',name];print('START',script,name,time.time(),flush=True)
        try:
            runpy.run_path(str(pathlib.Path(__file__).parent/script),run_name='__main__')
            write(RUN/'job_receipts'/(script+'_'+name+'.json'),{'status':'done','finished_unix':time.time(),'gpu_uuid':a.uuid})
        except BaseException as e:
            write(RUN/'job_receipts'/(script+'_'+name+'.json'),{'status':'failed','error':str(e),'traceback':traceback.format_exc(),'finished_unix':time.time(),'gpu_uuid':a.uuid});traceback.print_exc()
            # No blind retry; continue independent frozen jobs. Failures require diagnosis.
        gc.collect();torch.cuda.empty_cache()
    write(RUN/('WORKER_DONE_'+a.uuid+'.json'),{'pid':os.getpid(),'finished_unix':time.time()})
if __name__=='__main__':main()
