"""Evaluation-only worker: original audited loader plus independent per-prompt seeds.

The installed TRL HTTP schema accepts a single SamplingParams; this wrapper
expands a declared seed vector into the same fresh per-question SamplingParams
used by the frozen native evaluation. No training worker is modified.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
from trl_server_worker32 import worker_source


SEED_KEY='copsd_eval_question_seeds'


def per_question_params(template,prompts,params_class):
    extra=dict(template.extra_args or {});assert set(extra)=={SEED_KEY}
    seeds=extra[SEED_KEY]
    assert len(seeds)==len(prompts) and all(type(x) is int and 0<=x<2**32 for x in seeds)
    assert template.n==1 and template.temperature==1. and template.top_p==.95 and template.top_k==-1
    assert template.min_p==template.presence_penalty==template.frequency_penalty==0.
    assert template.repetition_penalty==1. and template.max_tokens==8192 and not template.ignore_eos
    assert not template.stop and not template.stop_token_ids and template.logprobs==0
    return [params_class(n=1,temperature=1.,top_p=.95,top_k=-1,min_p=0.,presence_penalty=0.,
                         max_tokens=8192,seed=seed,logprobs=0) for seed in seeds]


def llm_worker(script_args,data_parallel_rank,master_port,connection):
    import trl.scripts.vllm_serve as released
    from vllm import LLM,SamplingParams
    class SeededEvaluationLLM(LLM):
        def generate(self,prompts=None,sampling_params=None,*args,**kwargs):
            assert not args and prompts is not None
            params=per_question_params(sampling_params,prompts,SamplingParams)
            outputs=super().generate(prompts=prompts,sampling_params=params,**kwargs)
            assert len(outputs)==len(prompts)
            records=[]
            for prompt,param,result in zip(prompts,params,outputs,strict=True):
                text=prompt['prompt'] if isinstance(prompt,dict) else prompt
                assert result.prompt==text and len(result.outputs)==1
                output=result.outputs[0];ids=list(output.token_ids)
                records.append({'prompt_sha256':hashlib.sha256(text.encode()).hexdigest(),'seed':param.seed,
                    'token_ids_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                    'generated_tokens':len(ids),'finish_reason':output.finish_reason})
            path=Path(os.environ['COPSD_EVAL_SEED_RECEIPTS'])
            with path.open('a') as stream:stream.write(json.dumps({'per_question_params_reached_LLM_generate':True,'records':records})+'\n')
            return outputs
    namespace=dict(vars(released));namespace['LLM']=SeededEvaluationLLM
    exec(compile(ast.parse(worker_source()),'installed_TRL_eval_worker_audited_seqs32_question_seeds','exec'),namespace)
    namespace['llm_worker'](script_args,data_parallel_rank,master_port,connection)
