"""Check installed server sampler/loader compatibility without CUDA/NCCL."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import torch
import trl.scripts.vllm_serve as released
from vllm import SamplingParams
from trl_server_audited_worker import AuditedWeightSync,ORIGINAL_WORKER_SOURCE


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    assert ORIGINAL_WORKER_SOURCE==inspect.getsource(released.llm_worker)
    assert ORIGINAL_WORKER_SOURCE.count('trl.scripts.vllm_serve.WeightSyncWorkerExtension')==1
    params=SamplingParams(n=1,temperature=1.1,top_p=.95,top_k=20,min_p=0.,max_tokens=2048,repetition_penalty=1.,presence_penalty=0.,guided_decoding=None,logprobs=0)
    assert params.max_tokens==2048 and params.top_k==20 and params.presence_penalty==0.
    arguments=released.ScriptArguments(model='/not_loaded',host='127.0.0.1',tensor_parallel_size=1,data_parallel_size=1,gpu_memory_utilization=.9,dtype='bfloat16',max_model_len=20000)
    class Model:
        def __init__(self):self.loaded=[]
        def load_weights(self,weights):
            self.loaded.extend((n,v.clone()) for n,v in weights)
            return {n for n,v in weights}
    model=Model();worker=AuditedWeightSync();worker.device=torch.device('cpu');worker.client_rank=1
    payload=torch.arange(12,dtype=torch.bfloat16).reshape(3,4)
    worker.communicator=SimpleNamespace(broadcast=lambda tensor,src:tensor.copy_(payload),group=SimpleNamespace(barrier=lambda:None))
    worker.model_runner=SimpleNamespace(model=model)
    with tempfile.TemporaryDirectory() as folder:
        receipt=Path(folder)/'receipts.jsonl';os.environ['COPSD_SERVER_RECEIPTS']=str(receipt)
        worker.update_named_param('test.weight','torch.bfloat16',(3,4))
        rows=[json.loads(x) for x in receipt.read_text().splitlines()]
        assert rows==[{'name':'test.weight','shape':[3,4],'dtype':'torch.bfloat16','sha256':hashlib.sha256(payload.view(torch.uint8).numpy().tobytes()).hexdigest(),'original_loader_returned':True}]
        assert len(model.loaded)==1 and torch.equal(model.loaded[0][1],payload)
    assert not torch.cuda.is_initialized()
    result={'all_passed':True,'GPU_started':False,'released_SamplingParams_accepts_full_budget':True,'original_worker_loop_preserved_except_extension_class':True,'original_CPU_loader_and_received_weight_receipt_match':True,'scope':'OriginalTRLSampler construction and CPU fake broadcast/loader only; not actual NCCL or remote CUDA weight synchronization.',
        'installed_server_source_sha256':hashlib.sha256(Path(released.__file__).read_bytes()).hexdigest(),
        'audited_worker_sha256':hashlib.sha256(Path(__file__).with_name('trl_server_audited_worker.py').read_bytes()).hexdigest()}
    with a.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
