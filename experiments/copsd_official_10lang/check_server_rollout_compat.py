"""CPU-only schema, sampling and current-policy cache checks before server trial."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from server_rollout_compat import compatible_client, ServerUpdateBatch


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();assert not a.output.exists()
    os.environ['COPSD_SERVER_GROUP_PORT']='51987'
    class FakeClient:
        def __init__(self,**kwargs):self.kwargs=kwargs;self.calls=[]
        def generate(self,prompts,**kwargs):
            self.calls.append((prompts,kwargs))
            return {'completion_ids':[[i+1,151645] for i in range(len(prompts))],'prompt_ids':[[1]]*len(prompts),'logprobs':[[]]*len(prompts)}
    client=compatible_client(FakeClient)(host='127.0.0.1')
    assert client.kwargs['group_port']==51987
    raw=client.generate(['p'],n=1,max_tokens=2048,presence_penalty=.2)
    assert raw==[[1,151645]] and client.calls[-1][1]['generation_kwargs']=={'presence_penalty':.2}
    t=SimpleNamespace(vllm_client=client,args=SimpleNamespace(gradient_accumulation_steps=32),state=SimpleNamespace(global_step=0),_last_vllm_sync_step=0,
        processing_class=SimpleNamespace(pad_token='<pad>',batch_decode=lambda x,**k:x),
        get_batch_samples=lambda *a,**k:([{'student_prompts':['<pad>p'+str(i)]} for i in range(32)],64))
    rejected=[]
    def must_reject(name,call):
        try:call()
        except AssertionError:rejected.append(name)
        else:raise RuntimeError('Missing rejection: '+name)
    with tempfile.TemporaryDirectory() as folder:
        b=ServerUpdateBatch(t,Path(folder));b.install();t.get_batch_samples()
        params={'n':1,'max_tokens':2048,'temperature':1.1,'top_p':.95,'top_k':20,'presence_penalty':0.,'guided_decoding_regex':None}
        assert client.generate(['p0'],**params)==[[1,151645]]
        must_reject('unconsumed_batch',lambda:t.get_batch_samples())
        must_reject('wrong_prompt_order',lambda:client.generate(['p2'],**params))
        t.state.global_step=1
        must_reject('cross_update_stale',lambda:client.generate(['p1'],**params));t.state.global_step=0
        t._last_vllm_sync_step=-1
        must_reject('unsynchronized_policy',lambda:client.generate(['p1'],**params));t._last_vllm_sync_step=0
        must_reject('changed_sampling',lambda:client.generate(['p1'],**{**params,'presence_penalty':.1}))
        for i in range(1,32):assert client.generate(['p'+str(i)],**params)==[[i+1,151645]]
        b.assert_drained();assert len(b.records)==1 and b.records[0]['rollouts']==32
        t.state.global_step=t._last_vllm_sync_step=1;t.get_batch_samples()
        for i in range(32):client.generate(['p'+str(i)],**params)
        assert len(b.records)==2 and len(client.calls)==3  # one schema call + two full batches
        assert len((Path(folder)/'SERVER_ROLLOUTS.PRIVATE.jsonl').read_text().splitlines())==64
        b.detach()
    result={'all_passed':True,'GPU_started':False,'client_return_schema_adapted':True,'presence_penalty_transported_via_generation_kwargs':True,'rollouts_per_update':32,'groups':2,'guard_rejections':rejected,'scope':'CPU fake transport and policy-cache invariants only. No NCCL, remote weights, CUDA memory or actual model-generation claim.',
        'adapter_sha256':hashlib.sha256(Path(__file__).with_name('server_rollout_compat.py').read_bytes()).hexdigest()}
    with a.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
