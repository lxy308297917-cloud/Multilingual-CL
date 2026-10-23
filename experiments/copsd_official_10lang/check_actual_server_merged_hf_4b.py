"""Post-hoc localization of the original native-LoRA probability-tail gate failure.

No new generations. Score all256 saved JA20 service tokens using a separate HF
clone merged by original PEFT semantics. Original gates/results remain unchanged.
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


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--prepare-only',action='store_true');a=parser.parse_args()
    code=Path(__file__).parent;f=read(code/'actual_server_mergedHF4b_frozen_20261023.PUBLIC.json')
    for name,digest in f['code_sha256'].items():assert sha(code/name)==digest,name
    original_path=code/'actual_server_student4b_frozen_20261023.PUBLIC.json';assert sha(original_path)==f['primary_freeze_sha256'];original=read(original_path)
    lock_path=a.root/original['model_lock_relative'];assert sha(lock_path)==original['model_lock_sha256'];lock=read(lock_path)
    for name,info in lock['files'].items():assert sha(a.root/lock['model_relative']/name)==info['sha256'],name
    adapter=a.root/original['adapter_relative'];assert sha(adapter/'adapter_model.safetensors')==original['adapter_sha256']
    source=a.root/original['output_relative'];raw_path=source/'PREFIX_PROBABILITIES.PRIVATE.json';assert sha(raw_path)==f['prefix_probabilities_sha256'];raw=read(raw_path)['JA20'];assert len(raw)==32 and sum(len(x['completion_ids']) for x in raw)==256
    diagnostic=read(source/'DIAGNOSTIC_DONE.json');assert diagnostic['numeric_gates_all_passed'] is False and diagnostic['Base_restore_gate_passed']
    state=read(a.root/'gpu-queue/queues/copsd-actual-server4b-v1/state.json');assert state['phase']=='completed' and state['job']['exit_code']==0
    receipt_path=a.root/'logs/copsd_official_10lang'/ (original['run_id']+'.remote_weights.PRIVATE.jsonl');assert sha(receipt_path)==f['receipt_sha256']
    with receipt_path.open() as stream:receipts=[json.loads(x) for x in stream]
    assert len(receipts)==796;expected={x['name']:x for x in receipts[:398]};assert len(expected)==398
    output=a.root/f['output_relative'];assert not output.exists()
    if a.prepare_only:print('CPU_MERGED_HF_LOCALIZATION_PREFLIGHT_NO_GPU',flush=True);return
    assert os.environ['GPU_QUEUE_GPU_COUNT']=='1';visible=os.environ['CUDA_VISIBLE_DEVICES'];assert visible.startswith('GPU-') and ',' not in visible
    from benchmark_eval_concurrency8192 import wait_for_empty_GPU
    wait_for_empty_GPU(visible)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    tick=time.monotonic();model=AutoModelForCausalLM.from_pretrained(a.root/'models/Qwen3-4B',torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':0})
    model=PeftModel.from_pretrained(model,adapter,is_trainable=False).eval()
    model=model.merge_and_unload(safe_merge=False).eval();model.requires_grad_(False)
    actual={}
    for name,value in model.named_parameters():actual[name]={'name':name,'shape':list(value.shape),'dtype':str(value.dtype),'sha256':hashlib.sha256(value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest(),'original_loader_returned':True}
    weight_mismatches=[k for k in set(actual)|set(expected) if actual.get(k)!=expected.get(k)]
    output.mkdir(parents=True);(output/'WEIGHT_COMPARISON.PRIVATE.json').write_text(json.dumps({'expected':expected,'actual':actual},indent=2)+'\n')
    errors=[];agreements=0;details=[]
    with torch.no_grad():
        for row in raw:
            ids,tokens=row['prompt_ids'],row['completion_ids'];data=torch.tensor([ids+tokens],device='cuda:0')
            logits=model(input_ids=data,attention_mask=torch.ones_like(data),use_cache=False).logits[0,len(ids)-1:len(ids)+len(tokens)-1].float()
            values=logits.log_softmax(-1).gather(-1,torch.tensor(tokens,device='cuda:0')[:,None]).squeeze(-1).cpu().tolist();top=logits.argmax(-1).cpu().tolist()
            errors.extend(abs(x-y) for x,y in zip(values,row['server_logprobs']));agreements+=sum(x==y for x,y in zip(top,tokens))
            details.append({'id':row['id'],'language':row['language'],'merged_HF_logprobs':values,'server_logprobs':row['server_logprobs'],'native_HF_logprobs':row['matching_HF_logprobs'],'merged_HF_top1':top,'server_tokens':tokens})
    assert len(errors)==256;ordered=sorted(errors);mae=sum(errors)/256;p99=ordered[int(.99*255)]
    report={'posthoc_localization':True,'saved_positions':256,'new_generated_tokens':0,'merged_HF_weight_parameters':len(actual),'parameters_exactly_match_JA20_received_tensors':not weight_mismatches,'weight_mismatch_names':weight_mismatches,'server_vs_merged_HF_top1_agree':agreements,'server_vs_merged_HF_top1_agreement':agreements/256,'server_vs_merged_HF_logprob_MAE':mae,'server_vs_merged_HF_logprob_p99_abs_error':p99,'same_original_numeric_gate_passed':not weight_mismatches and agreements/256>=.95 and mae<=.1 and p99<=.3,'original_native_HF_gate_passed':False,'original_native_HF_logprob_MAE':diagnostic['conditions']['JA20']['matching_HF_logprob_MAE'],'original_native_HF_logprob_p99_abs_error':diagnostic['conditions']['JA20']['matching_HF_logprob_p99_abs_error'],'seconds':time.monotonic()-tick,'scope':f['scope']}
    (output/'SCORING.PRIVATE.json').write_text(json.dumps(details)+'\n');(output/'LOCALIZATION_DONE.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
