"""Fixed four-endpoint development block after the parallel mandatory runs."""
import subprocess,sys,json,time,traceback
from pathlib import Path
from calibration import ROOT,write,sha
from train import BASE
RUN=Path('/data0/users/lixiaoyang/models/checkpoint/paired_capability_sparse_20261021/comparison/matched_v1')
def main():
    start=time.monotonic();completed=[]
    try:
        for label,model in [('Base',BASE)]+[(m+'2048',RUN/m/'final') for m in ('A','D','R')]:
            subprocess.run([sys.executable,str(ROOT/'evaluate_dev.py'),'--model',str(model),'--label',label,'--output',str(RUN/('dev_'+label))],check=True);completed.append(label)
        command=[sys.executable,str(ROOT/'compare_dev.py'),'--output',str(RUN/'MANDATORY_DECISION.json')]
        for label in completed:command+=['--endpoint',label+'='+str(RUN/('dev_'+label))]
        subprocess.run(command,check=True)
        write(RUN/'DEV_BLOCK_DONE.json',{'completed':completed,'protocol_sha256':sha(ROOT/'comparison_protocol.v1.json'),'wall_seconds':time.monotonic()-start,'report':str(RUN/'MANDATORY_DECISION.json'),'supersedes_original_serial_coordinator':True,'long_training_released':False})
    except BaseException as error:write(RUN/'DEV_BLOCK_FAILED.json',{'error':str(error),'traceback':traceback.format_exc(),'completed':completed,'wall_seconds':time.monotonic()-start});raise
if __name__=='__main__':main()
