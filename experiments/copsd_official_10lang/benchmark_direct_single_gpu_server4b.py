"""Full8192 fixed64 SW/EN engineering benchmark of memory-only merged serving.

No training or model checkpoint writes. Native32 measurements are historical
references, not a controlled same-time comparison or a quality equivalence claim.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(4*2**20),b''):h.update(block)
    return h.hexdigest()


def prepare(root):
    code=Path(__file__).parent;f=read(code/'direct_single_gpu_server4b_frozen_20261023.PUBLIC.json')
    for name,digest in f['code_sha256'].items():assert sha(code/name)==digest,name
    for name,digest in f['installed_source_sha256'].items():assert sha(root/name)==digest,name
    lock_path=root/f['model_lock_relative'];assert sha(lock_path)==f['model_lock_sha256'];lock=read(lock_path)
    for name,info in lock['files'].items():assert sha(root/lock['model_relative']/name)==info['sha256'],name
    from benchmark_eval_concurrency4b import prepare as native_prepare
    native,reference,rows=native_prepare(root,code)
    assert native['model_revision']==f['model_revision'] and native['adapter_sha256']==f['adapter_sha256']
    assert sha(root/f['native32_done_relative'])==f['native32_done_sha256']
    assert sha(root/f['native32_items_relative'])==f['native32_items_sha256']==read(root/f['native32_done_relative'])['items_sha256']
    assert sha(root/f['canonical_reference_relative'])==f['canonical_reference_sha256']
    from safetensors import safe_open
    with (root/f['canonical_reference_relative']).open() as stream:canonical=[json.loads(x) for x in stream]
    assert len(canonical)==398
    weights={x['name']:x for x in canonical}
    base_index=read(root/'models/Qwen3-4B/model.safetensors.index.json')['weight_map']
    assert set(weights)==set(base_index) and len(base_index)==398
    cfg=read(Path(reference['adapter'])/'adapter_config.json')
    assert cfg['r']==64 and cfg['lora_alpha']==128 and not any(cfg.get(k) for k in ['use_rslora','use_dora','lora_bias','fan_in_fan_out','rank_pattern','alpha_pattern','modules_to_save'])
    targets=set(cfg['target_modules']);assert targets=={'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'}
    selected=[k for k in weights if k.removesuffix('.weight').split('.')[-1] in targets];assert len(selected)==252
    with safe_open(str(Path(reference['adapter'])/'adapter_model.safetensors'),framework='pt',device='cpu') as adapter:
        keys={'base_model.model.'+name.removesuffix('.weight')+'.lora_'+side+'.weight' for name in selected for side in ['A','B']}
        assert set(adapter.keys())==keys
        for name in selected:
            prefix='base_model.model.'+name.removesuffix('.weight')
            ashape=adapter.get_slice(prefix+'.lora_A.weight').get_shape();bshape=adapter.get_slice(prefix+'.lora_B.weight').get_shape()
            assert ashape[0]==bshape[1]==64 and weights[name]['shape']==[bshape[0],ashape[1]]
    assert sha(root/f['two_gpu_items_relative'])==f['two_gpu_items_sha256']
    output=root/f['output_relative'];assert not output.exists()
    return f,rows,reference,output


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--prepare-only',action='store_true');a=parser.parse_args()
    f,rows,reference,output=prepare(a.root)
    if a.prepare_only:print('CPU_PREFLIGHT_SEEDED64_FULL8192_NO_GPU',flush=True);return
    assert os.environ['GPU_QUEUE_GPU_COUNT']=='1' and os.environ['CUDA_VISIBLE_DEVICES']==''
    output.mkdir(parents=True)
    from transformers import AutoTokenizer
    from trl.extras.vllm_client import VLLMClient
    from evaluate_vllm import official_api,question_seed
    from trl_server_eval_seeded_worker32 import SEED_KEY
    tick=time.monotonic()
    tokenizer=AutoTokenizer.from_pretrained(reference['model']);api=official_api(Path(reference['source']))
    prompts=[api['build_qwen3_prompt'](tokenizer,row['question'],row['language'].upper(),True) for row in rows]
    seeds=[question_seed(42,row['language'],row['problem_id']) for row in rows]
    prompt_ids=[tokenizer.encode(x,add_special_tokens=False) for x in prompts]
    assert len(prompts)==64 and max(map(len,prompt_ids))+8192<=40960
    client=VLLMClient(host='127.0.0.1',server_port=int(os.environ['COPSD_SERVER_HTTP_PORT']),connection_timeout=120)
    controller_prepare_seconds=time.monotonic()-tick
    try:
        initialization=read(Path(os.environ['COPSD_DIRECT_INIT_RECEIPT']))
        assert initialization['all398_reference_weights_match'] and initialization['all_weights_loaded']==398 and initialization['adapted_weights_loaded']==252
        tick=time.monotonic()
        response=client.generate(prompts=prompts,n=1,temperature=1.,top_p=.95,top_k=-1,min_p=0.,max_tokens=8192,generation_kwargs={'presence_penalty':0.,'extra_args':{SEED_KEY:seeds}})
        seconds=time.monotonic()-tick
        assert response['prompt_ids']==prompt_ids and len(response['completion_ids'])==64
        with Path(os.environ['COPSD_EVAL_SEED_RECEIPTS']).open() as stream:receipts=[json.loads(x) for x in stream]
        assert len(receipts)==1 and receipts[0]['per_question_params_reached_LLM_generate'];metadata=receipts[0]['records'];assert len(metadata)==64
        records=[]
        for row,prompt,seed,tokens,meta in zip(rows,prompts,seeds,response['completion_ids'],metadata,strict=True):
            assert meta['prompt_sha256']==hashlib.sha256(prompt.encode()).hexdigest() and meta['seed']==seed
            assert meta['token_ids_sha256']==hashlib.sha256(json.dumps(tokens).encode()).hexdigest() and len(tokens)==meta['generated_tokens']
            assert 0<len(tokens)<=8192 and meta['finish_reason'] in ['stop','length']
            records.append({'id':row['benchmark_id'],'source_id':row['problem_id'],'language':row['language'],'question':row['question'],'answer':row['Answer'],'prompt':prompt,'seed':seed,'token_ids':tokens,'finish_reason':meta['finish_reason'],'raw_output':tokenizer.decode(tokens,skip_special_tokens=False)})
        with (output/'ITEMS.jsonl').open('x') as stream:
            for row in records:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
        native=read(a.root/f['native32_done_relative']);tokens=sum(len(x['token_ids']) for x in records)
        with (a.root/f['native32_items_relative']).open() as stream:prior=[json.loads(x) for x in stream]
        assert len(prior)==64
        for current,old in zip(records,prior,strict=True):assert all(current[k]==old[k] for k in ['id','source_id','language','question','answer','prompt','seed'])
        with (a.root/f['two_gpu_items_relative']).open() as stream:two_gpu=[json.loads(x) for x in stream]
        assert len(two_gpu)==64
        for current,old in zip(records,two_gpu,strict=True):assert all(current[k]==old[k] for k in ['id','source_id','language','question','answer','prompt','seed'])
        result={'items':64,'generated_tokens':tokens,'generation_HTTP_wall_seconds':seconds,'tokens_per_generation_second':tokens/seconds,'CPU_controller_prepare_seconds':controller_prepare_seconds,'canonical_initialization':initialization,'per_question_seed_receipts_verified':True,'all398_tensor_receipts_verified':True,'normal_terminations':sum(x['finish_reason']=='stop' for x in records),'truncated':sum(x['finish_reason']=='length' for x in records),'exact_token_output_matches_two_gpu_merged':sum(x['token_ids']==y['token_ids'] for x,y in zip(records,two_gpu)),'exact_token_output_matches_native32':sum(x['token_ids']==y['token_ids'] for x,y in zip(records,prior)),'historical_native32':native,'generation_wall_reference_speedup':native['generation_seconds']/seconds,'token_throughput_reference_ratio':(tokens/seconds)/native['tokens_per_generation_second'],'items_sha256':sha(output/'ITEMS.jsonl'),'scope':f['scope']}
        (output/'BENCHMARK_DONE.json').write_text(json.dumps(result,indent=2)+'\n');print('DIRECT_SINGLE_GPU_SERVER64_DONE',json.dumps({k:v for k,v in result.items() if k not in ['historical_native32','scope']}),flush=True)
    finally:
        client.session.close()
        (output/'SERVER_CLIENT_SHUTDOWN_DONE.json').write_text(json.dumps({'communicator_created':False,'CPU_controller_only':True})+'\n')


if __name__=='__main__':main()
