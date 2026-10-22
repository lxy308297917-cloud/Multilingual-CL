"""Manually launched one-shot CPU supervisor; wait for three fixed prefixes.

No reboot startup, no retry, and no method/parameter choices from dev scores.
"""
import json,subprocess,sys,time,traceback
from pathlib import Path
from calibration import ROOT,write,sha
RUN=Path('/data0/users/lixiaoyang/models/checkpoint/paired_capability_sparse_20261021/comparison/matched_v1')
QUEUE=Path('/data0/users/lixiaoyang/gpu-queue/gpu_queue.py')
STATE=RUN/'DEADLINE_SUPERVISOR.json'

def main():
    if STATE.exists():raise FileExistsError('Supervisor already has a record; no automatic rerun')
    start=time.time();write(STATE,{'phase':'waiting_training','started_unix':start,'pid':__import__('os').getpid(),'target':'2026-10-22 20:00 CST','interval_seconds':60})
    try:
        rows=[json.loads(s) for s in (ROOT/'data/calibration_pairs.v1.jsonl').read_text().splitlines()]
        while True:
            phases={}
            d=RUN/'D';gate=RUN/'DEADLINE_PAIRING_GATE_AT1024.json'
            if not gate.exists() and (d/'RECOVERY_TRANSPORT_001024.json').exists():
                state=json.loads((d/'cache_001024/MASTERY.json').read_text())['records'];joint=sum(state[r['id']+':en']['known'] and state[r['id']+':sw']['known'] for r in rows);enonly=sum(state[r['id']+':en']['known'] and not state[r['id']+':sw']['known'] for r in rows)
                write(gate,{'joint_known':joint,'EN_only_known':enonly,'adequate':joint>=8 and enonly>=8,'cache_sha256':sha(d/'cache_001024/MASTERY.json'),'fixed_checkpoint':str(d/'checkpoint-001024.pt'),'transport_checked':True,'no_dev_score_used':True,'not_a_pairing_efficacy_result':True})
            for method in ('D','A','R'):
                folder=RUN/method
                if (folder/'FAILED.json').exists():raise RuntimeError(method+' training failed; inspect and recover only after diagnosis')
                if (folder/'TRAINING_DONE.json').exists():
                    receipt=json.loads((folder/'TRAINING_DONE.json').read_text())
                    if receipt['global_step']!=2048 or receipt['method']!=method or receipt['protocol_sha256']!=sha(ROOT/'comparison_protocol.v1.json'):raise RuntimeError('Training receipt mismatch')
                    phases[method]='done2048'
                elif (folder/'PROGRESS.json').exists():phases[method]=json.loads((folder/'PROGRESS.json').read_text())['step']
                else:phases[method]='waiting_start'
            write(STATE,{'phase':'waiting_training','started_unix':start,'updated_unix':time.time(),'training':phases,'target':'2026-10-22 20:00 CST','parent_serial_status':'superseded after D; A/R already launched independently'})
            if all(value=='done2048' for value in phases.values()):break
            time.sleep(60)
        command=['/usr/bin/python3',str(QUEUE),'--queue','paired-cap-dev-20261021','start','--script',str(ROOT/'run_dev_block.sh'),'--conda-prefix','/data0/users/lixiaoyang/ssu-migration/envs/cl','--conda-exe','/data0/users/lixiaoyang/.local/miniforge3/bin/conda','--gpus','1','--interval-minutes','5','--log','/data0/users/lixiaoyang/logs/paired_capability_sparse_20261021/development_block_v1.log']
        result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,check=True)
        write(STATE,{'phase':'development_manually_queued_once','started_unix':start,'updated_unix':time.time(),'training':phases,'queue':'paired-cap-dev-20261021','command':command,'result':result.stdout,'pid_exits_after_dispatch':True,'no_restart_or_retry':True})
    except BaseException as error:
        write(STATE,{'phase':'failed','error':str(error),'traceback':traceback.format_exc(),'started_unix':start,'updated_unix':time.time(),'no_retry':True});raise
if __name__=='__main__':main()
