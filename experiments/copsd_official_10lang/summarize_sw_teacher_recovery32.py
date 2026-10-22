"""Audit real completed SW teacher continuations and paired same-trajectory budgets."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
from evaluate_vllm import official_api, question_seed


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def pair(a,b,field='correct'):
    assert [r['id'] for r in a]==[r['id'] for r in b]
    aa=[r[field] for r in a];bb=[r[field] for r in b]
    return {'reference_correct':sum(aa),'comparison_correct':sum(bb),'gained':sum(not x and y for x,y in zip(aa,bb)),'lost':sum(x and not y for x,y in zip(aa,bb))}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists();root=a.root;code=Path(__file__).parent
    freeze=code/'sw_teacher_recovery32_frozen_20261022.PUBLIC.json';f=read(freeze)
    state=read(root/'gpu-queue/queues/copsd-sw-teacher-recovery32-v1/state.json')
    assert state['phase']=='completed' and state['job']['exit_code']==0
    job=state['job'];assert len(job['gpu_uuids'])==1 and job['config']['max_user_gpus']==4
    assert job['script_sha256']==f['code_sha256']['run_sw_teacher_recovery32.sh']
    for name,digest in f['code_sha256'].items():assert sha(code/name)==digest,name
    folder=root/f['output_relative'];cfg=read(folder/'CONFIG.PRIVATE.json');done=read(folder/'DONE.json')
    assert cfg['frozen_sha256']==sha(freeze) and cfg['teacher_adapter'] is None and cfg['actual_max_num_seqs']==64
    assert done['training_updates']==0 and done['protocol']==f['protocol']
    prepared=root/f['prepared_relative']
    for name,digest in f['prepared_sha256'].items():assert sha(prepared/name)==digest
    context=read(prepared/'CONFIG.PRIVATE.json');original=read(prepared/'PROMPTS.PRIVATE.json')[:32]
    source=Path(context['source'])/'polymath_eval/evaluate_math.py';assert sha(source)==f['official_evaluation_sha256']
    api=official_api(source)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(context['model'],local_files_only=True)
    records={};sources={}
    for condition in f['conditions']:
        prefix=[[] for _ in original]
        if condition!='No_history':
            spec=f['history_inputs'][condition];path=root/spec['relative'];assert sha(path)==spec['sha256']
            prefix=[r['samples'][0]['token_ids'][:256] for r in map(json.loads,path.read_text().splitlines()[:32])]
        path=folder/(condition+'_ITEMS.PRIVATE.jsonl');rows=[json.loads(s) for s in path.read_text().splitlines()]
        assert len(rows)==32 and [r['id'] for r in rows]==f['selected_IDs']
        for row,ref,pre in zip(rows,original,prefix):
            assert row['answer']==ref['answer'] and row['prefix_token_ids']==pre
            assert row['seed']==question_seed(42,'sw',row['id']) and 0<len(row['new_token_ids'])<=8192
            for length,answer_key,correct_key in [(8192,'boxed_answer','correct'),(2048,'censored2048_box','censored2048_correct')]:
                answer=api['extract_boxed_answer'](tokenizer.decode(pre+row['new_token_ids'][:length],skip_special_tokens=False))
                assert answer==row[answer_key] and bool(api['grade_answer'](answer,row['answer']))==row[correct_key]
        expected={'items':32,'correct':sum(r['correct'] for r in rows),'censored2048_correct':sum(r['censored2048_correct'] for r in rows),
            'normal_terminations':sum(r['finish_reason']=='stop' for r in rows),'truncated':sum(r['finish_reason']=='length' for r in rows),
            'answer_extraction_failures':sum(r['boxed_answer'] is None for r in rows),'generated_tokens':sum(len(r['new_token_ids']) for r in rows),
            'supplied_history_tokens':sum(map(len,prefix)),'teacher_input_tokens':sum(len(r['teacher_prompt_token_ids'])+len(pre) for r,pre in zip(original,prefix))}
        for key,val in expected.items():assert done['conditions'][condition][key]==val,(condition,key)
        assert read(folder/(condition+'_DONE.json'))==done['conditions'][condition]
        records[condition]=rows;sources[condition]={'items_sha256':sha(path),'config_sha256':sha(folder/'CONFIG.PRIVATE.json')}
    hours=(datetime.datetime.fromisoformat(job['finished_at'])-datetime.datetime.fromisoformat(job['started_at'])).total_seconds()/3600
    length_pairs={}
    for condition,rows in records.items():
        length_pairs[condition]={'censored2048_correct':sum(r['censored2048_correct'] for r in rows),'full8192_correct':sum(r['correct'] for r in rows),
            'gained_after2048':sum(not r['censored2048_correct'] and r['correct'] for r in rows),'lost_after2048':sum(r['censored2048_correct'] and not r['correct'] for r in rows)}
    from torch.utils.tensorboard import SummaryWriter
    tb=root/'logs/copsd_official_10lang/tensorboard/sw_teacher_recovery32_20261022_v1';assert not tb.exists()
    with SummaryWriter(str(tb)) as writer:
        for condition,m in done['conditions'].items():
            writer.add_scalar(condition+'/accuracy_full8192',m['correct']/32,0)
            writer.add_scalar(condition+'/accuracy_same_trajectory_censored2048',m['censored2048_correct']/32,0)
            writer.add_scalar(condition+'/generated_tokens',m['generated_tokens'],0)
        writer.add_scalar('cost/unique_parent_GPU_hours',hours,0)
    result={**done,'observed_at':datetime.datetime.now().isoformat(),'frozen_sha256':sha(freeze),'parent_completed_exit0':True,
        'all128_raw_conditions_prefix_ID_seed_token_budget_and_official_regrades_verified':True,
        'Base_history_vs_SW_history':{case:pair(records['Base_history'],records[case]) for case in ['SW42_history','SW43_history']},
        'same_trajectory_length_pairs':length_pairs,'source_records':sources,
        'cost':{'new_responses':128,'generated_tokens':sum(m['generated_tokens'] for m in done['conditions'].values()),'unique_parent_GPU_hours_counted_once':hours,'generation_seconds':sum(m['generation_seconds'] for m in done['conditions'].values()),'scope':'Parent allocation once, includes load/CPU/inference, excludes waiting/hold. No new training; input history was generated previously.'},
        'TensorBoard_relative':str(tb.relative_to(root)),'analysis_source_sha256':sha(Path(__file__))}
    with a.output.open('x') as stream:json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps({'conditions':done['conditions'],'paired':result['Base_history_vs_SW_history'],'length':length_pairs,'cost':result['cost']}))


if __name__=='__main__':main()
