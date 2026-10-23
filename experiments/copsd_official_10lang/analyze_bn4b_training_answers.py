"""CPU score/termination coverage of saved BN4B640+640 on-policy trajectories.

Training outputs are descriptive diagnostics, never heldout accuracy or proof
quality. Official extractor/grader retained; missing boxes are not relabeled.
"""
import argparse
from collections import Counter
import datetime
import hashlib
import json
from pathlib import Path
import sys
from evaluate_vllm import official_api


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
    with p.open() as stream:return [json.loads(s) for s in stream]


def summarize(records):
    return {'trajectories':len(records),'boxed_answer_present':sum(r['boxed_answer_present'] for r in records),'official_correct':sum(r['correct'] for r in records),'thinking_closed':sum(r['thinking_closed'] for r in records),'EOS_token_present':sum(r['EOS_token_present'] for r in records),'length2048':sum(r['length2048'] for r in records),'length2048_without_box':sum(r['length2048'] and not r['boxed_answer_present'] for r in records),'boxed_but_incorrect':sum(r['boxed_answer_present'] and not r['correct'] for r in records),'valid_response_tokens':sum(r['valid_tokens'] for r in records),'correct_without_thinking_closed':sum(r['correct'] and not r['thinking_closed'] for r in records)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=a.root;code=Path(__file__).parent;source=root/'COPSD-official-20261015'
    baseline=read(code/'bn4b_server20_frozen_20261023.PUBLIC.json');manifest=root/baseline['training_manifest_relative'];assert sha(manifest)==baseline['training_manifest_sha256'];m=read(manifest);data=root/m['data_relative'];assert sha(data)==m['data_sha256']
    prepared=read(code/'bn4b_initial_teacher32_replay_prepared_20261023.PUBLIC.json')
    for name,digest in prepared['official_source_sha256'].items():assert sha(source/name)==digest
    cases={}
    for name,frozen in [('Clip20',baseline),('Unclip20',read(code/'bn4b_unclip20_frozen_20261023.PUBLIC.json'))]:
        folder=root/frozen['checkpoint_relative'];rp=folder/'SERVER_ROLLOUTS.PRIVATE.jsonl';ep=folder/'EXPOSURE_PROMPTS.PRIVATE.json';rr=rows(rp);exposure=read(ep)
        assert len(rr)==len(exposure)==640 and [r['global_step'] for r in rr]==[i//32 for i in range(640)]
        cases[name]={'rows':rr,'exposure':exposure,'rollout_sha256':sha(rp),'exposure_sha256':sha(ep),'proof':read(folder/'SERVER_PILOT20_UPDATES_DONE.json')}
    assert cases['Clip20']['rows'][:32]==cases['Unclip20']['rows'][:32]
    assert cases['Clip20']['exposure']==cases['Unclip20']['exposure']
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(root/'models/Qwen3-4B',local_files_only=True)
    sys.path.insert(0,str(source))
    from multilingual_data_collator import MultilingualSelfDistillationDataCollator
    collator=MultilingualSelfDistillationDataCollator(tokenizer,max_length=20000,reason_first=False,student_enable_thinking=True,include_problem_en=True,include_reference_solution_en=True)
    targets={r['student_prompt_sha256'] for case in cases.values() for r in case['rows']};mapping={}
    for original in read(data):
        batch=collator([{**original,'target_lang':'BN'}]);text=tokenizer.decode(batch['student_prompts'][0],skip_special_tokens=False)
        if tokenizer.pad_token:text=text.replace(tokenizer.pad_token,'')
        digest=hashlib.sha256(text.encode()).hexdigest()
        if digest in targets:
            mapping.setdefault(digest,[]).append({'id':str(original['problem_id']),'answer':original['Answer'],'student_prompts_sha256':hashlib.sha256(batch['student_prompts'].numpy().tobytes()).hexdigest(),'teacher_prompts_sha256':hashlib.sha256(batch['teacher_prompts'].numpy().tobytes()).hexdigest()})
    assert set(mapping)==targets
    api=official_api(source/'polymath_eval/evaluate_math.py')
    private=root/'logs/copsd_official_10lang/bn4b_training_answer_coverage_20261023_v1';private.mkdir(exist_ok=False)
    result={};all_records={}
    for name,case in cases.items():
        scored=[]
        with (private/(name+'_ITEMS.PRIVATE.jsonl')).open('x') as stream:
            for i,r in enumerate(case['rows']):
                matches=[row for row in mapping[r['student_prompt_sha256']] if all(row[k]==case['exposure'][i][k] for k in ['student_prompts_sha256','teacher_prompts_sha256'])]
                assert len(matches)==1,'Ambiguous or mismatched original question'
                original=matches[0];tokens=r['completion_ids'];assert 0<len(tokens)<=2048
                text=tokenizer.decode(tokens,skip_special_tokens=False);answer=api['extract_boxed_answer'](text)
                row={'index':i,'global_step':r['global_step'],'id':original['id'],'boxed_answer_present':answer is not None,'correct':bool(api['grade_answer'](answer,original['answer'])),'thinking_closed':'</think>' in text,'EOS_token_present':tokenizer.eos_token_id in tokens,'length2048':len(tokens)==2048,'valid_tokens':sum(t!=tokenizer.pad_token_id for t in tokens)}
                stream.write(json.dumps(row)+'\n');scored.append(row)
        totals=summarize(scored);assert totals['valid_response_tokens']==case['proof']['cost']['valid_response_loss_positions']
        result[name]={'totals':totals,'updates':[{'global_step':i,**summarize(scored[i*32:(i+1)*32])} for i in range(20)],'five_update_blocks':[{'updates':[i+1,i+5],**summarize(scored[i*32:(i+5)*32])} for i in [0,5,10,15]],'unique_questions':len({r['id'] for r in scored}),'rollout_sha256':case['rollout_sha256'],'exposure_sha256':case['exposure_sha256'],'scored_items_sha256':sha(private/(name+'_ITEMS.PRIVATE.jsonl'))}
        all_records[name]=scored;print(name,json.dumps(totals),flush=True)
    assert [r['id'] for r in all_records['Clip20']]==[r['id'] for r in all_records['Unclip20']]
    report={'completed_at':datetime.datetime.now().isoformat(),'protocol':'BN4B_actual_training640_answer_coverage_v1','conditions':result,'same640_question_exposure_order_and_both_prompt_tensor_hashes_verified':True,'first32_rollout_tokens_identical':True,'source_data_sha256':sha(data),'official_scoring_source_sha256':sha(source/'polymath_eval/evaluate_math.py'),'analysis_sha256':sha(Path(__file__)),'scope':'Training-distribution descriptive scores using existing online trajectories, not development/test accuracy or proof quality. Boxedanswers withinthinking may score correct withoutfinalsection; EOSpresence andlengthbudget are separate from missinganswer. First32identical; latertrajectories reflect evolvingdifferentpolicies and are not fixedtrajectory causal comparisons. No generation, GPU or parameter update.'}
    with a.output.open('x') as stream:stream.write(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
