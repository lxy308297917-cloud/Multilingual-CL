"""Fixed-token engineering diagnostic; never scores mathematical answers."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prepare(root):
    code=Path(__file__).parent;freeze=code/'engine_probability_probe_frozen_20261022.PUBLIC.json';f=read(freeze)
    for name,digest in f['code_sha256'].items():assert sha(code/name)==digest,name
    path=root/f['input_relative'];assert sha(path)==f['input_sha256'];inputs=read(path)
    assert len(inputs)==len({r['key'] for r in inputs})==32
    assert len({r['id'] for r in inputs})==4 and {r['language'] for r in inputs}=={'sw','en'}
    assert {r['response_position'] for r in inputs}=={0,32,64,128}
    adapter=root/f['adapter_relative'];assert sha(adapter/'adapter_model.safetensors')==f['adapter_sha256']
    assert read(adapter/'trainer_state.json')['global_step']==20
    export=root/f['dense_export_relative'];manifest=export/'EXPORT.MANIFEST.PRIVATE.json'
    assert sha(manifest)==f['dense_export_manifest_sha256']
    m=read(manifest);assert m['completed'] and m['source_adapter_sha256']==f['adapter_sha256']
    for r in inputs:assert len(r['input_ids'])>=2 and r['input_ids'][-1]==r['target_token_id']
    return f,inputs,adapter,export


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--stage',choices=['hf','native','dense'],required=True);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    f,inputs,adapter,export=prepare(a.root);output=a.root/f['output_relative'];destination=output/(a.stage+'_DONE.PRIVATE.json');assert not destination.exists()
    if a.prepare_only:
        print(json.dumps({'stage':a.stage,'prefixes':32,'distinct_original_questions':4,'GPU_used':False}));return
    assert os.environ.get('GPU_QUEUE_GPU_COUNT')=='1'
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','');assert visible.startswith('GPU-') and ',' not in visible
    from benchmark_eval_concurrency8192 import wait_for_empty_GPU
    wait_for_empty_GPU(visible)
    if a.stage=='dense':
        for name,info in read(export/'EXPORT.MANIFEST.PRIVATE.json')['files'].items():assert sha(export/name)==info['sha256'],name
    output.mkdir(parents=True,exist_ok=True);tick=time.monotonic();records={};details={}
    if a.stage=='hf':
        import torch
        from transformers import AutoModelForCausalLM
        from peft import PeftModel
        base=AutoModelForCausalLM.from_pretrained(a.root/'models/Qwen3-1.7B',local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda:0')
        model=PeftModel.from_pretrained(base,adapter,is_trainable=False);model.eval()
        details={'base_dtype':str(next(base.parameters()).dtype),'LoRA_dtypes':sorted({str(p.dtype) for n,p in model.named_parameters() if 'lora_' in n}),'attention':'sdpa','adapter_merge_performed':False}
        def measure():
            result=[]
            with torch.inference_mode():
                for row in inputs:
                    tokens=torch.tensor([row['input_ids']],dtype=torch.long,device='cuda:0')
                    logits=model(input_ids=tokens,attention_mask=torch.ones_like(tokens),use_cache=False).logits[0,-2].float()
                    probs=torch.log_softmax(logits,dim=-1);argmax=int(probs.argmax().item())
                    result.append({'key':row['key'],'target_logprob':float(probs[row['target_token_id']].item()),'argmax_token_id':argmax,'argmax_logprob':float(probs[argmax].item())})
            return result
        with model.disable_adapter():records['Base']=measure()
        records['LoRA']=measure()
        torch.cuda.synchronize()
    else:
        from vllm import LLM,SamplingParams
        from vllm.lora.request import LoRARequest
        from evaluate_vllm import official_api
        def configured(**kwargs):return LLM(**kwargs,max_num_seqs=16)
        api=official_api(a.root/'COPSD-official-20261015/polymath_eval/evaluate_math.py',{'LLM':configured,'SamplingParams':SamplingParams})
        model_path=export if a.stage=='dense' else a.root/'models/Qwen3-1.7B'
        llm,_=api['load_vllm_model'](str(model_path),str(adapter) if a.stage=='native' else None,gpu_memory_utilization=.9,tensor_parallel_size=1,max_model_len=40960,enable_thinking=True)
        assert llm.llm_engine.vllm_config.scheduler_config.max_num_seqs==16
        details={'actual_max_num_seqs':16,'model_dtype':str(llm.llm_engine.vllm_config.model_config.dtype),'prompt_logprobs_k':20,'dummy_max_new_tokens':1}
        for condition in (['Base','LoRA'] if a.stage=='native' else ['Dense']):
            params=[SamplingParams(n=1,temperature=1.,top_p=1.,top_k=-1,max_tokens=1,prompt_logprobs=20,seed=42) for _ in inputs]
            request=LoRARequest('sw20_fixed',1,str(adapter)) if condition=='LoRA' else None
            generated=llm.generate([{'prompt_token_ids':row['input_ids']} for row in inputs],params,lora_request=request,use_tqdm=True)
            assert len(generated)==32;result=[]
            for row,out in zip(inputs,generated):
                assert out.prompt_token_ids==row['input_ids'] and len(out.prompt_logprobs)==len(row['input_ids'])
                last=out.prompt_logprobs[-1];assert row['target_token_id'] in last
                argmax=max(last,key=lambda x:last[x].logprob)
                result.append({'key':row['key'],'target_logprob':float(last[row['target_token_id']].logprob),'argmax_token_id':argmax,'argmax_logprob':float(last[argmax].logprob),'dummy_generated_tokens':len(out.outputs[0].token_ids)})
            records[condition]=result
    result={'protocol':f['protocol'],'stage':a.stage,'input_sha256':f['input_sha256'],'records':records,'details':details,'elapsed_seconds_including_load':time.monotonic()-tick,
        'packages':{n:importlib.metadata.version(n) for n in ['torch','transformers','peft','vllm']},'scope':f['scope']}
    with destination.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({'stage':a.stage,'conditions':list(records),'prefixes_each':32,'elapsed_seconds':result['elapsed_seconds_including_load']}),flush=True)


if __name__=='__main__':main()
