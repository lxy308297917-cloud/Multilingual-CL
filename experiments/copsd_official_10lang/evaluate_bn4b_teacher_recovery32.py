"""Frozen4B privileged continuation of first32 heldout BN histories, no training.

Not replay of training-token KL: histories come from native16 development outputs.
Same256-token prefix budget across three student conditions; clean has no history.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
    with p.open() as f: return [json.loads(line) for line in f]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args(); root=a.root; code=Path(__file__).parent
    frozen_path=code/'bn4b_teacher_recovery32_frozen_20261023.PUBLIC.json'
    frozen=read(frozen_path)
    for name,digest in frozen['runtime_sha256'].items(): assert sha(code/name)==digest,name
    for name,digest in frozen['official_source_sha256'].items(): assert sha(root/'COPSD-official-20261015'/name)==digest,name
    f=read(code/'bn4b_unclip20_eval_frozen_20261023.PUBLIC.json')
    data=root/f['inputs']['bn']['relative']; assert sha(data)==f['inputs']['bn']['sha256']
    selected=read(data); assert [r['problem_id'] for r in selected]==f['selected_IDs']
    selected=selected[:32]
    source=root/'COPSD-official-20261015';sys.path.insert(0,str(source))
    from transformers import AutoTokenizer
    from multilingual_data_collator import MultilingualSelfDistillationDataCollator
    model=root/'models/Qwen3-4B'
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
    collator=MultilingualSelfDistillationDataCollator(tokenizer,max_length=20000,reason_first=False,student_enable_thinking=True,include_problem_en=True,include_reference_solution_en=True)
    prompts=[]
    for r in selected:
        encoded=collator([{**r,'target_lang':'BN'}])
        ids=encoded['teacher_prompts'][0].tolist()
        assert len(ids)<20000 and r['solution'].strip()
        assert all(x==1 for x in encoded['teacher_prompt_attention_mask'][0].tolist())
        prompts.append(ids)
    files={'Base':root/f['cached_Base_BN']['relative']/'ITEMS.jsonl',
           'Clip20':root/'logs/copsd_official_10lang/bn4b_followup_development200/bn4b_followup200_BN20_bn_20261023/ITEMS.jsonl',
           'Unclip20':root/'logs/copsd_official_10lang/bn4b_unclip20_followup_development200/bn4b_unclip20_followup200_Unclip20_bn_20261023/ITEMS.jsonl'}
    accepted=read(code/'bn4b_unclip20_completed_20261023.PUBLIC.json')
    histories={}
    for name,path in files.items():
        assert sha(path)==accepted['languages']['bn']['conditions'][name]['source']['items_sha256']
        rr=rows(path);assert len(rr)==200
        assert [r['id'] for r in rr]==f['selected_IDs']
        assert [r['answer'] for r in rr[:32]]==[r['Answer'] for r in selected]
        assert all(len(r['samples'][0]['token_ids'])>=256 for r in rr[:32])
        histories[name]=[r['samples'][0]['token_ids'][:256] for r in rr[:32]]
    histories['Clean']=[[] for _ in selected]
    lock=root/f['model_lock_relative'];assert sha(lock)==f['model_lock_sha256']
    identity=read(lock);assert identity['revision']==f['model_revision']
    for name,info in identity['files'].items(): assert sha(model/name)==info['sha256'],name
    output=root/'logs/copsd_official_10lang/bn4b_teacher_recovery32_8192_20261023_v1'
    config={'protocol':'project_BN4B_privileged_recovery_first32_8192_v1','revision':f['model_revision'],
        'selection':'first32 frozen development IDs, no score-based selection','items':32,'conditions':list(histories),
        'history_tokens':256,'new_tokens':8192,'samples':1,'temperature':1.,'top_p':.95,'top_k':-1,'seed':42,
        'source_file_sha256':{n:sha(source/n) for n in ['multilingual_data_collator.py','language_config.py','polymath_eval/evaluate_math.py']},
        'history_file_sha256':{n:sha(path) for n,path in files.items()},'data_sha256':sha(data),'script_sha256':sha(Path(__file__)), 'frozen_sha256':sha(frozen_path),
        'scope':'Conditional generation with English reference, not fair benchmark, SFT labels, online training trajectory replay or token-gradient correctness proof. Clean has256 fewer supplied tokens. Three history conditions matched on supplied length; no history filtering.'}
    assert not output.exists()
    if a.prepare_only:
        print(json.dumps({'preflight':True,'conditions':4,'planned_completions':128,'all_histories_have256tokens':True,'no_GPU_loaded':True}));return
    assert os.environ.get('GPU_QUEUE_GPU_COUNT')=='1'
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','');assert visible.startswith('GPU-') and ',' not in visible
    import subprocess
    check=subprocess.run(['nvidia-smi','-i',visible,'--query-compute-apps=pid','--format=csv,noheader,nounits'],check=True,text=True,capture_output=True,timeout=15)
    assert all(s.strip().isdigit() and int(s)==os.getpid() for s in check.stdout.splitlines() if s.strip()),'Foreign compute process on assigned GPU'
    from evaluate_vllm import official_api,question_seed,write_json
    from vllm import LLM,SamplingParams
    def limited(**kwargs):return LLM(**kwargs,max_num_seqs=16)
    api=official_api(source/'polymath_eval/evaluate_math.py',{'LLM':limited,'SamplingParams':SamplingParams})
    output.mkdir(parents=True);write_json(output/'CONFIG.PUBLIC.json',config)
    write_json(output/'PHASE.json',{'phase':'model_load'})
    tick=time.monotonic()
    llm,_=api['load_vllm_model'](str(model),None,gpu_memory_utilization=.9,tensor_parallel_size=1,max_model_len=40960,enable_thinking=True)
    elapsed=time.monotonic()-tick; summaries={}
    for name,prefixes in histories.items():
        inputs=[{'prompt_token_ids':q+h} for q,h in zip(prompts,prefixes,strict=True)]
        params=[SamplingParams(n=1,temperature=1.,top_p=.95,top_k=-1,max_tokens=8192,seed=question_seed(42,'bn',r['problem_id'])) for r in selected]
        write_json(output/'PHASE.json',{'phase':name});tick=time.monotonic()
        generated=llm.generate(inputs,params,use_tqdm=True);assert len(generated)==32
        counts={'items':32,'correct':0,'truncated':0,'extraction_failures':0,'generated_tokens':0,'supplied_history_tokens':sum(map(len,prefixes)),'prefix_already_boxed':0}
        with (output/(name+'_ITEMS.PRIVATE.jsonl')).open('x') as stream:
            for row,prefix,inp,response in zip(selected,prefixes,inputs,generated,strict=True):
                assert response.prompt_token_ids==inp['prompt_token_ids'] and len(response.outputs)==1
                sample=response.outputs[0];combined=tokenizer.decode(prefix,skip_special_tokens=False)+sample.text
                answer=api['extract_boxed_answer'](combined);correct=bool(api['grade_answer'](answer,row['Answer']))
                counts['correct']+=correct;counts['truncated']+=sample.finish_reason=='length';counts['extraction_failures']+=answer is None
                counts['generated_tokens']+=len(sample.token_ids);counts['prefix_already_boxed']+=api['extract_boxed_answer'](tokenizer.decode(prefix)) is not None
                stream.write(json.dumps({'id':row['problem_id'],'answer':row['Answer'],'correct':correct,'boxed_answer':answer,'finish_reason':sample.finish_reason,'prefix_token_ids':prefix,'new_token_ids':list(sample.token_ids),'combined_text':combined},ensure_ascii=False)+'\n')
        counts['generation_seconds']=time.monotonic()-tick;summaries[name]=counts;write_json(output/(name+'_DONE.json'),counts)
    write_json(output/'DONE.PUBLIC.json',{'protocol':config['protocol'],'conditions':summaries,'model_load_seconds':elapsed,'scope':config['scope'],'source_script_sha256':config['script_sha256']})
    write_json(output/'PHASE.json',{'phase':'done'});print(json.dumps(summaries),flush=True)


if __name__=='__main__':main()
