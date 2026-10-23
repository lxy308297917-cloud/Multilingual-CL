"""Single-GPU evaluation initialization, canonical BF16 LoRA merge on that GPU.

No CPU teacher/model forward, separate HF GPU, full merged checkpoint, or changes
to training. All398 tensors must match an existing canonical export receipt.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import time
from trl_server_audited_worker import AuditedWeightSync
from trl_server_worker32 import worker_source
from trl_server_eval_seeded_worker32 import per_question_params


def read(p):return json.loads(Path(p).read_text())
def tensor_sha(value):
    import torch
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


class DirectMergedWeightSync(AuditedWeightSync):
    def initialize_canonical_adapter(self,base_path,adapter_path,reference_path,audit_path):
        assert not getattr(self,'_canonical_initialized',False)
        import torch
        from safetensors import safe_open
        tick=time.monotonic();base=Path(base_path);index=read(base/'model.safetensors.index.json')['weight_map'];cfg=read(base/'config.json')
        assert len(index)==398 and cfg['hidden_size']==2560 and cfg['num_hidden_layers']==36 and cfg['tie_word_embeddings']
        with Path(reference_path).open() as stream:reference=[json.loads(x) for x in stream]
        assert len(reference)>=398 and len(reference)%398==0
        expected={x['name']:x for x in reference[-398:]};assert set(expected)==set(index)
        model=self.model_runner.model;device=next(model.parameters()).device;assert device.type=='cuda'
        adapter=None;scaling=None;adapted=[]
        targets={'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'}
        if adapter_path:
            directory=Path(adapter_path);acfg=read(directory/'adapter_config.json')
            assert acfg['r']==64 and acfg['lora_alpha']==128 and set(acfg['target_modules'])==targets
            assert acfg['bias']=='none' and not any(acfg.get(k) for k in ['use_dora','use_rslora','lora_bias','fan_in_fan_out','rank_pattern','alpha_pattern','modules_to_save'])
            adapter=safe_open(str(directory/'adapter_model.safetensors'),framework='pt',device='cpu');scaling=2.
            adapted=[k for k in index if k.removesuffix('.weight').split('.')[-1] in targets]
            assert len(adapted)==252
            keys={'base_model.model.'+name.removesuffix('.weight')+'.lora_'+side+'.weight' for name in adapted for side in ['A','B']}
            assert set(adapter.keys())==keys and len(keys)==504
        previous_tf32=torch.backends.cuda.matmul.allow_tf32
        previous_precision=torch.get_float32_matmul_precision()
        torch.backends.cuda.matmul.allow_tf32=False
        records=[];loaded=0;audit=Path(audit_path);assert not audit.exists()
        try:
            with torch.no_grad(),audit.open('x') as stream:
                for name in sorted(index):
                    with safe_open(str(base/index[name]),framework='pt',device='cpu') as shard:value=shard.get_tensor(name).to(torch.bfloat16)
                    changed=name in adapted
                    value=value.to(device)
                    if changed:
                        stem='base_model.model.'+name.removesuffix('.weight')
                        a=adapter.get_tensor(stem+'.lora_A.weight').to(device=device,dtype=torch.float32)
                        b=adapter.get_tensor(stem+'.lora_B.weight').to(device=device,dtype=torch.float32)
                        assert a.shape[0]==b.shape[1]==64 and tuple(value.shape)==(b.shape[0],a.shape[1])
                        value.add_((b@a)*scaling)
                    row={'name':name,'shape':list(value.shape),'dtype':str(value.dtype),'sha256':tensor_sha(value)}
                    matches=all(row[k]==expected[name][k] for k in row)
                    stream.write(json.dumps({**row,'reference_matches':matches,'has_LoRA_delta':changed})+'\n');stream.flush()
                    assert matches,'Canonical merged weight mismatch: '+name
                    model.load_weights(weights=[(name,value)])
                    loaded+=1
                    records.append(row)
                    del value
            torch.cuda.synchronize(device);self._canonical_initialized=True
        finally:
            torch.set_float32_matmul_precision(previous_precision)
            assert torch.backends.cuda.matmul.allow_tf32==previous_tf32
        return {'all398_reference_weights_match':len(records)==398,'all_weights_loaded':loaded,'adapted_weights_loaded':len(adapted),
                'LoRA_tensors':504 if adapter_path else 0,'fresh_Base_no_adapter':not bool(adapter_path),
                'merge_TF32_enabled':False,'previous_float32_matmul_precision':previous_precision,'previous_TF32_restored':torch.backends.cuda.matmul.allow_tf32==previous_tf32,
                'merge_seconds':time.monotonic()-tick,'no_CPU_model_forward':True,'no_full_model_checkpoint_write':True}


def llm_worker(script_args,data_parallel_rank,master_port,connection):
    import trl.scripts.vllm_serve as released
    from vllm import LLM,SamplingParams
    class DirectSeededLLM(LLM):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            result=self.collective_rpc(method='initialize_canonical_adapter',kwargs={
                'base_path':script_args.model,'adapter_path':os.environ.get('COPSD_DIRECT_ADAPTER_PATH') or None,
                'reference_path':os.environ['COPSD_DIRECT_REFERENCE_RECEIPTS'],
                'audit_path':os.environ['COPSD_DIRECT_WEIGHT_AUDIT']})
            assert len(result)==1 and result[0]['all398_reference_weights_match']
            self.reset_prefix_cache()
            with Path(os.environ['COPSD_DIRECT_INIT_RECEIPT']).open('x') as stream:stream.write(json.dumps(result[0],indent=2)+'\n')
        def generate(self,prompts=None,sampling_params=None,*args,**kwargs):
            assert not args and prompts is not None
            params=per_question_params(sampling_params,prompts,SamplingParams)
            outputs=super().generate(prompts=prompts,sampling_params=params,**kwargs);assert len(outputs)==len(prompts)
            records=[]
            for prompt,param,result in zip(prompts,params,outputs,strict=True):
                text=prompt['prompt'] if isinstance(prompt,dict) else prompt;assert result.prompt==text and len(result.outputs)==1
                item=result.outputs[0];ids=list(item.token_ids)
                records.append({'prompt_sha256':hashlib.sha256(text.encode()).hexdigest(),'seed':param.seed,
                    'token_ids_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest(),'generated_tokens':len(ids),'finish_reason':item.finish_reason})
            with Path(os.environ['COPSD_EVAL_SEED_RECEIPTS']).open('a') as stream:stream.write(json.dumps({'per_question_params_reached_LLM_generate':True,'records':records})+'\n')
            return outputs
    source=worker_source();needle='trl_server_audited_worker.AuditedWeightSync';assert source.count(needle)==1
    source=source.replace(needle,'trl_server_eval_direct_lora_worker32.DirectMergedWeightSync')
    namespace=dict(vars(released));namespace['LLM']=DirectSeededLLM
    exec(compile(ast.parse(source),'installed_TRL_singleGPU_canonical_merged_eval_worker','exec'),namespace)
    namespace['llm_worker'](script_args,data_parallel_rank,master_port,connection)
