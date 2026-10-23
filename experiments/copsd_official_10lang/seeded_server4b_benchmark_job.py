"""Own and clean up exactly one two-GPU server/training resource job."""
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
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    from benchmark_seeded_server4b import prepare
    f,rows,data,output=prepare(a.root);code=Path(__file__).parent
    if a.prepare_only:
        print(json.dumps({'CPU_preflight':True,'roles':2,'completion_budget':8192,'GPU_started':False}));return
    assert json.loads((code/'sw4b_server_resource_v2_completed_20261022.PUBLIC.json').read_text())['engineering_completed']
    assert os.environ['GPU_QUEUE_GPU_COUNT']=='2'
    devices=os.environ['CUDA_VISIBLE_DEVICES'].split(',');assert len(devices)==len(set(devices))==2 and all(x.startswith('GPU-') for x in devices)
    from benchmark_eval_concurrency8192 import wait_for_empty_GPU
    for device in devices:wait_for_empty_GPU(device)
    logs=a.root/'logs/copsd_official_10lang';stem=f['run_id']
    receipts=logs/(stem+'.remote_weights.PRIVATE.jsonl');assert not receipts.exists()
    role_info=logs/(stem+'.roles.PRIVATE.json');assert not role_info.exists()
    ports=[]
    for _ in range(2):
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));ports.append(sock.getsockname()[1])
    assert ports[0]!=ports[1]
    base=dict(os.environ,COPSD_SERVER_HTTP_PORT=str(ports[0]),COPSD_SERVER_GROUP_PORT=str(ports[1]),COPSD_SERVER_RECEIPTS=str(receipts),COPSD_SERVER_GPU_UUID=devices[1],COPSD_EVAL_SEED_RECEIPTS=str(logs/(stem+'.question_seeds.PRIVATE.jsonl')))
    model=a.root/'models/Qwen3-4B'
    command=[sys.executable,str(code/'run_trl_eval_seeded_server32.py'),'--model',str(model),'--host','127.0.0.1','--port',str(ports[0]),'--tensor_parallel_size','1','--data_parallel_size','1','--dtype','bfloat16','--gpu_memory_utilization','.85','--max_model_len','40960']
    server=train=None;monitors=[];streams=[];status={'roles':{},'diagnostic_exit_code':None,'server_forced_kill':False}
    try:
        for i,device in enumerate(devices):
            stream=(logs/(stem+f'.gpu_role{i}.csv')).open('x');streams.append(stream)
            monitors.append(subprocess.Popen(['nvidia-smi','-i',device,'--query-gpu=timestamp,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits','-l','1'],stdout=stream,stderr=subprocess.STDOUT))
        stream=(logs/(stem+'.server.log')).open('x');streams.append(stream)
        server=subprocess.Popen(command,env=dict(base,CUDA_VISIBLE_DEVICES=devices[1]),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        status['roles']['server']={'pid':server.pid,'GPU_UUID':devices[1],'command':command}
        role_info.write_text(json.dumps(status,indent=2)+'\n')
        deadline=time.monotonic()+240
        while True:
            if server.poll() is not None:raise RuntimeError('Server exited during initialization: '+str(server.returncode))
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{ports[0]}/health/',timeout=2) as response:
                    if response.status==200:break
            except (OSError,urllib.error.URLError):pass
            if time.monotonic()>deadline:raise TimeoutError('Server health initialization timed out; preserve logs')
            time.sleep(1)
        command=[sys.executable,str(code/'benchmark_seeded_server4b.py'),'--root',str(a.root)]
        train=subprocess.Popen(command,env=dict(base,CUDA_VISIBLE_DEVICES=devices[0]))
        status['roles']['checker']={'pid':train.pid,'GPU_UUID':devices[0],'command':command}
        role_info.write_text(json.dumps(status,indent=2)+'\n')
        deadline=time.monotonic()+1800
        while train.poll() is None:
            if time.monotonic()>deadline:raise TimeoutError('Diagnostic exceeded30minutes; no retry')
            if server.poll() is not None:raise RuntimeError('Server exited while trainer was active')
            time.sleep(1)
        status['diagnostic_exit_code']=train.returncode
        if train.returncode:raise RuntimeError('Training role failed: '+str(train.returncode))
        assert (output/'BENCHMARK_DONE.json').exists()
        assert json.loads((output/'SERVER_CLIENT_SHUTDOWN_DONE.json').read_text())['communicator_closed']
    finally:
        if train is not None and train.poll() is None:
            # This exact Popen is ours. Never derive kill targets from GPU process lists.
            train.terminate()
            try:train.wait(timeout=20)
            except subprocess.TimeoutExpired:train.kill();train.wait()
        if server is not None and server.poll() is None:
            os.killpg(server.pid,signal.SIGTERM)
            try:server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                status['server_forced_kill']=True;os.killpg(server.pid,signal.SIGKILL);server.wait()
        if server is not None:status['server_exit_code']=server.returncode
        for monitor in monitors:
            monitor.terminate()
            try:monitor.wait(timeout=5)
            except subprocess.TimeoutExpired:monitor.kill();monitor.wait()
        for stream in streams:stream.close()
        role_info.write_text(json.dumps(status,indent=2)+'\n')
    assert not status['server_forced_kill'],'Server did not shut down within grace period'
    (output/'SERVER_PARENT_DONE.json').write_text(json.dumps({'diagnostic_exit_code':0,'server_cleanup_completed':True,'server_exit_code':status['server_exit_code'],'two_role_job_completed':True})+'\n')
    print('TWO_GPU_SERVER_RESOURCE_PARENT_DONE',flush=True)


if __name__=='__main__':main()
