"""Own one evaluation GPU and an independent CPU-only controller."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--case',choices=['Base','BN20'],required=True);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    from evaluate_canonical_merged_bn4b import prepare
    f,spec,rows,output,validation=prepare(a.root,a.case);code=Path(__file__).parent
    if a.prepare_only:print('CPU_PREFLIGHT_CANONICAL_BN4B400_FULL8192_NO_GPU',flush=True);return
    assert os.environ['GPU_QUEUE_GPU_COUNT']=='1';visible=os.environ['CUDA_VISIBLE_DEVICES'];assert visible.startswith('GPU-') and ',' not in visible
    from benchmark_eval_concurrency8192 import wait_for_empty_GPU
    wait_for_empty_GPU(visible)
    logs=a.root/'logs/copsd_official_10lang';stem=spec['run_id'];role_path=logs/(stem+'.roles.PRIVATE.json');assert not role_path.exists()
    initialization=logs/(stem+'.canonical_init.PRIVATE.json');audit=logs/(stem+'.canonical_weights.PRIVATE.jsonl');seeds=logs/(stem+'.question_seeds.PRIVATE.jsonl')
    assert not any(x.exists() for x in [initialization,audit,seeds])
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    env=dict(os.environ,COPSD_SERVER_HTTP_PORT=str(port),COPSD_SERVER_GPU_UUID=visible,
        COPSD_DIRECT_ADAPTER_PATH=str(a.root/spec['adapter_relative']) if spec['adapter_relative'] else '',COPSD_DIRECT_REFERENCE_RECEIPTS=str(a.root/spec['canonical_reference_relative']),
        COPSD_DIRECT_WEIGHT_AUDIT=str(audit),COPSD_DIRECT_INIT_RECEIPT=str(initialization),COPSD_EVAL_SEED_RECEIPTS=str(seeds))
    proof_path=logs/(stem+'.parent_model_validation.PRIVATE.json');assert not proof_path.exists();proof_path.write_text(json.dumps(validation,indent=2)+'\n')
    from evaluate_canonical_merged_bn4b import sha
    env['COPSD_PARENT_MODEL_VALIDATION']=str(proof_path);env['COPSD_PARENT_MODEL_VALIDATION_SHA256']=sha(proof_path)
    command=[sys.executable,str(code/'run_trl_eval_direct_server32.py'),'--model',str(a.root/'models/Qwen3-4B'),'--host','127.0.0.1','--port',str(port),'--tensor_parallel_size','1','--data_parallel_size','1','--dtype','bfloat16','--gpu_memory_utilization','.85','--max_model_len','40960']
    server=controller=monitor=None;streams=[];status={'roles':{},'controller_exit_code':None,'server_forced_kill':False}
    try:
        stream=(logs/(stem+'.gpu.csv')).open('x');streams.append(stream)
        monitor=subprocess.Popen(['nvidia-smi','-i',visible,'--query-gpu=timestamp,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits','-l','1'],stdout=stream,stderr=subprocess.STDOUT)
        stream=(logs/(stem+'.server.log')).open('x');streams.append(stream)
        server=subprocess.Popen(command,env=dict(env,CUDA_VISIBLE_DEVICES=visible),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        status['roles']['server']={'pid':server.pid,'GPU_UUID':visible,'command':command};role_path.write_text(json.dumps(status,indent=2)+'\n')
        deadline=time.monotonic()+240
        while True:
            if server.poll() is not None:raise RuntimeError('Canonical server initialization failed; inspect preserved server log: '+str(server.returncode))
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/health/',timeout=2) as response:
                    if response.status==200:break
            except (OSError,urllib.error.URLError):pass
            if time.monotonic()>deadline:raise TimeoutError('Canonical server health initialization exceeded240seconds')
            time.sleep(1)
        receipt=json.loads(initialization.read_text());assert receipt['all398_reference_weights_match'] and receipt['all_weights_loaded']==398
        command=[sys.executable,str(code/'evaluate_canonical_merged_bn4b.py'),'--root',str(a.root),'--case',a.case]
        controller=subprocess.Popen(command,env=dict(env,CUDA_VISIBLE_DEVICES=''))
        status['roles']['CPU_controller']={'pid':controller.pid,'CUDA_VISIBLE_DEVICES':'','command':command};role_path.write_text(json.dumps(status,indent=2)+'\n')
        deadline=time.monotonic()+4*3600
        while controller.poll() is None:
            if server.poll() is not None:raise RuntimeError('Canonical server exited during evaluation')
            if time.monotonic()>deadline:raise TimeoutError('Frozen400 development evaluation exceeded4hours')
            time.sleep(1)
        status['controller_exit_code']=controller.returncode
        if controller.returncode:raise RuntimeError('CPU controller failed: '+str(controller.returncode))
        assert json.loads((output/'DONE.json').read_text())['items']==400
        assert json.loads((output/'PHASE.json').read_text())['phase']=='done'
    finally:
        if controller is not None and controller.poll() is None:
            controller.terminate()
            try:controller.wait(timeout=20)
            except subprocess.TimeoutExpired:controller.kill();controller.wait()
        if server is not None and server.poll() is None:
            os.killpg(server.pid,signal.SIGTERM)
            try:server.wait(timeout=30)
            except subprocess.TimeoutExpired:status['server_forced_kill']=True;os.killpg(server.pid,signal.SIGKILL);server.wait()
        if server is not None:status['server_exit_code']=server.returncode
        if monitor is not None:
            monitor.terminate()
            try:monitor.wait(timeout=5)
            except subprocess.TimeoutExpired:monitor.kill();monitor.wait()
        for stream in streams:stream.close()
        role_path.write_text(json.dumps(status,indent=2)+'\n')
    assert not status['server_forced_kill']
    (output/'SERVER_PARENT_DONE.json').write_text(json.dumps({'controller_exit_code':0,'server_cleanup_completed':True,'server_exit_code':status['server_exit_code'],'single_GPU_job_completed':True,'CPU_controller_only':True})+'\n')
    print('CANONICAL_MERGED_BN4B_EVAL_PARENT_DONE',flush=True)


if __name__=='__main__':main()
