"""Read-only Base -> savedJA20 -> Base actual server prefix-probability check.

Diagnostic only: fixed32 prompts/greedy8 tokens, not training or a benchmark score.
Uses the existing audited TRL server and original named-weight loader.
"""
import argparse
import atexit
from contextlib import nullcontext
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
    code=Path(__file__).parent;f=read(code/'actual_server_student4b_frozen_20261023.PUBLIC.json')
    for k,v in f['code_sha256'].items():assert sha(code/k)==v,k
    lock=read(root/f['model_lock_relative']);assert sha(root/f['model_lock_relative'])==f['model_lock_sha256']
    assert lock['revision']==f['model_revision']
    for name,meta in lock['files'].items():assert sha(root/lock['model_relative']/name)==meta['sha256'],name
    for k,v in f['installed_source_sha256'].items():assert sha(root/k)==v,k
    adapter=root/f['adapter_relative'];assert sha(adapter/'adapter_model.safetensors')==f['adapter_sha256']
    rows=[]
    for lang,meta in f['inputs'].items():
        path=root/meta['relative'];assert sha(path)==meta['sha256'];source=read(path)
        selected=source[:16];assert [str(x['problem_id']) for x in selected]==meta['selected_IDs']
        for x in selected:rows.append({'id':str(x['problem_id']),'language':lang,'question':x['problem' if lang=='en' else 'problem_ja']})
    assert len(rows)==32
    assert sha(root/'COPSD-official-20261015/polymath_eval/evaluate_math.py')==f['official_evaluation_source_sha256']
    output=root/f['output_relative'];assert not output.exists()
    return f,rows,None,output


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    f,rows,_,output=prepare(a.root)
    if a.prepare_only:print('CPU_PREFLIGHT_32_FIXED_PROMPTS_NO_GPU_WORK',flush=True);return
    assert os.environ['GPU_QUEUE_GPU_COUNT']=='2'
    assert os.environ['CUDA_VISIBLE_DEVICES'].startswith('GPU-') and ',' not in os.environ['CUDA_VISIBLE_DEVICES']
    assert os.environ['CUDA_VISIBLE_DEVICES']!=os.environ['COPSD_SERVER_GPU_UUID']
    output.mkdir(parents=True)
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel
    from trl.extras.vllm_client import VLLMClient
    from lora_sync_weights import iter_merged_weights
    from evaluate_vllm import official_api
    model_path=a.root/'models/Qwen3-4B';tick=time.monotonic()
    base=AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':0})
    model=PeftModel.from_pretrained(base,a.root/f['adapter_relative'],is_trainable=False).eval()
    model.requires_grad_(False);assert all(x.device.type=='cuda' for x in model.parameters())
    tokenizer=AutoTokenizer.from_pretrained(model_path)
    api=official_api(a.root/'COPSD-official-20261015/polymath_eval/evaluate_math.py')
    prompts=[api['build_qwen3_prompt'](tokenizer,x['question'],x['language'].upper(),True) for x in rows]
    prompt_ids=[tokenizer.encode(x,add_special_tokens=False) for x in prompts];assert all(0<len(x)<5000 for x in prompt_ids)
    def fingerprint():
        h=hashlib.sha256()
        for name,value in model.named_parameters():
            if 'lora_' not in name:h.update(name.encode());h.update(value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
        return h.hexdigest()
    original_hash=fingerprint()
    client=VLLMClient(host='127.0.0.1',server_port=int(os.environ['COPSD_SERVER_HTTP_PORT']),group_port=int(os.environ['COPSD_SERVER_GROUP_PORT']),connection_timeout=120)
    client.init_communicator(device=0)
    receipt_path=Path(os.environ['COPSD_SERVER_RECEIPTS']);consumed=0;sync_records=[];results={};private={}
    def sync(which):
        nonlocal consumed
        start=time.monotonic();expected=[]
        if which=='JA20':iterator=iter_merged_weights(model)
        else:
            iterator=((name.removeprefix('base_model.model.').replace('.base_layer',''),v.detach()) for name,v in model.named_parameters() if 'lora_' not in name)
        for name,value in iterator:
            digest=hashlib.sha256(value.contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
            expected.append({'name':name,'shape':list(value.shape),'dtype':str(value.dtype),'sha256':digest,'original_loader_returned':True})
            client.update_named_param(name,value)
        client.reset_prefix_cache()
        with receipt_path.open() as stream:actual=[json.loads(x) for x in stream]
        assert len(expected)==398 and actual[consumed:]==expected;consumed=len(actual)
        assert fingerprint()==original_hash
        sync_records.append({'condition':which,'tensors':len(expected),'seconds':time.monotonic()-start})
    @torch.no_grad()
    def score(ids,tokens,active):
        data=torch.tensor([ids+tokens],device='cuda:0')
        with (nullcontext() if active else model.disable_adapter()):logits=model(input_ids=data,attention_mask=torch.ones_like(data),use_cache=False).logits
        selected=logits[0,len(ids)-1:len(ids)+len(tokens)-1].float()
        targets=torch.tensor(tokens,device=selected.device)
        values=selected.log_softmax(-1).gather(-1,targets[:,None]).squeeze(-1).cpu().tolist()
        best=selected.argmax(-1).cpu().tolist()
        del logits,selected,data
        return values,best
    try:
        for condition in ['Base_initial','JA20','Base_restored']:
            if condition=='JA20':sync('JA20')
            elif condition=='Base_restored':sync('Base')
            response=client.generate(prompts=prompts,n=1,temperature=0.,top_p=1.,top_k=-1,min_p=0.,max_tokens=8,generation_kwargs={'seed':42,'ignore_eos':True,'presence_penalty':0.})
            assert response['prompt_ids']==prompt_ids and len(response['completion_ids'])==32
            errors=[];wrong=[];agree=0;total=0;details=[]
            for row,ids,tokens,logps in zip(rows,prompt_ids,response['completion_ids'],response['logprobs'],strict=True):
                assert len(tokens)==len(logps)==8
                correct,hf_top=score(ids,tokens,condition=='JA20')
                other,_=score(ids,tokens,condition!='JA20')
                assert all(math_isfinite(x) for x in logps+correct+other)
                errors.extend(abs(x-y) for x,y in zip(logps,correct));wrong.extend(abs(x-y) for x,y in zip(logps,other))
                agree+=sum(x==y for x,y in zip(tokens,hf_top));total+=8
                details.append({'id':row['id'],'language':row['language'],'prompt_ids':ids,'completion_ids':tokens,'server_logprobs':logps,'matching_HF_logprobs':correct,'other_HF_logprobs':other,'matching_HF_top1':hf_top})
            ordered=sorted(errors);p99=ordered[int(.99*(len(ordered)-1))]
            results[condition]={'positions':total,'HF_top1_agree':agree,'HF_top1_agreement':agree/total,'matching_HF_logprob_MAE':sum(errors)/total,'other_HF_logprob_MAE':sum(wrong)/total,'matching_HF_logprob_p99_abs_error':p99,'within_declared_numeric_gate':agree/total>=.95 and sum(errors)/total<=.1 and p99<=.3}
            private[condition]=details
            (output/'PREFIX_PROBABILITIES.PRIVATE.json').write_text(json.dumps(private,ensure_ascii=False)+'\n')
            print('ACTUAL_SERVER_CONDITION',json.dumps({condition:results[condition]}),flush=True)
        initial=private['Base_initial'];restored=private['Base_restored'];adapter=private['JA20']
        exact=sum(x['completion_ids']==y['completion_ids'] for x,y in zip(initial,restored));maxrestore=max(abs(v-w) for x,y in zip(initial,restored) for v,w in zip(x['server_logprobs'],y['server_logprobs']))
        changed=sum(x['completion_ids']!=y['completion_ids'] for x,y in zip(initial,adapter));unchanged=fingerprint()==original_hash
        report={'conditions':results,'Base_restored_exact_prompt_completions':exact,'Base_restore_max_logprob_difference':maxrestore,'JA_adapter_changed_greedy_prompt_completions':changed,'base_HF_unchanged_bitwise':unchanged,'weight_syncs':sync_records,'seconds':time.monotonic()-tick,'numeric_gates_all_passed':all(x['within_declared_numeric_gate'] for x in results.values()),'Base_restore_gate_passed':exact==32 and maxrestore<=1e-5,'adapter_response_observed':changed>0,'scope':f['scope']}
        (output/'DIAGNOSTIC_DONE.json').write_text(json.dumps(report,indent=2)+'\n')
    finally:
        client.close_communicator();atexit.unregister(client.close_communicator)
        (output/'SERVER_CLIENT_SHUTDOWN_DONE.json').write_text(json.dumps({'communicator_closed':True})+'\n')


def math_isfinite(x):
    import math
    return math.isfinite(x)


if __name__=='__main__':main()
