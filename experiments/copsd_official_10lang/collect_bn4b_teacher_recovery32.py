"""CPU acceptance of the frozen4B teacher conditional-recovery diagnostic."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
from evaluate_vllm import official_api


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
    with p.open() as f:return [json.loads(s) for s in f]


def pair(a,b):
    assert [r['id'] for r in a]==[r['id'] for r in b]
    aa=[r['correct'] for r in a];bb=[r['correct'] for r in b]
    return {'reference_correct':sum(aa),'comparison_correct':sum(bb),'gained':sum(not x and y for x,y in zip(aa,bb,strict=True)),'lost':sum(x and not y for x,y in zip(aa,bb,strict=True))}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=a.root;code=Path(__file__).parent;freeze=code/'bn4b_teacher_recovery32_frozen_20261023.PUBLIC.json';f=read(freeze)
    for name,digest in f['runtime_sha256'].items():assert sha(code/name)==digest,name
    for name,digest in f['official_source_sha256'].items():assert sha(root/'COPSD-official-20261015'/name)==digest,name
    queue=read(root/'gpu-queue/queues/copsd-bn4b-teacher-recovery32-v1/state.json')
    assert queue['phase']=='completed' and queue['job']['exit_code']==0
    job=queue['job'];assert len(job['gpu_uuids'])==1 and job['config']['max_user_gpus']==4
    assert job['script_sha256']==f['runtime_sha256']['run_bn4b_teacher_recovery32.sh']
    folder=root/'logs/copsd_official_10lang/bn4b_teacher_recovery32_8192_20261023_v1'
    cfg=read(folder/'CONFIG.PUBLIC.json');done=read(folder/'DONE.PUBLIC.json')
    assert cfg['protocol']==done['protocol']==f['protocol'] and cfg['frozen_sha256']==sha(freeze)
    assert cfg['script_sha256']==f['runtime_sha256']['evaluate_bn4b_teacher_recovery32.py']
    assert cfg['revision']==f['model_revision'] and cfg['conditions']==f['conditions']
    for key in ['new_tokens','samples','temperature','top_p','top_k','seed']:assert cfg[key]==f[key]
    evaluation=read(code/'bn4b_unclip20_eval_frozen_20261023.PUBLIC.json');data=root/evaluation['inputs']['bn']['relative'];assert sha(data)==cfg['data_sha256']==evaluation['inputs']['bn']['sha256']
    selected=read(data)[:32];ids=[r['problem_id'] for r in selected];assert ids==evaluation['selected_IDs'][:32]
    api=official_api(root/'COPSD-official-20261015/polymath_eval/evaluate_math.py')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(root/'models/Qwen3-4B',local_files_only=True)
    files={'Base':root/evaluation['cached_Base_BN']['relative']/'ITEMS.jsonl','Clip20':root/'logs/copsd_official_10lang/bn4b_followup_development200/bn4b_followup200_BN20_bn_20261023/ITEMS.jsonl','Unclip20':root/'logs/copsd_official_10lang/bn4b_unclip20_followup_development200/bn4b_unclip20_followup200_Unclip20_bn_20261023/ITEMS.jsonl'}
    records={};sources={};lengths={}
    for name in f['conditions']:
        prefix=[[] for _ in selected]
        if name!='Clean':
            assert sha(files[name])==cfg['history_file_sha256'][name]
            history=rows(files[name])[:32];assert [r['id'] for r in history]==ids
            prefix=[r['samples'][0]['token_ids'][:256] for r in history]
        path=folder/(name+'_ITEMS.PRIVATE.jsonl');rr=rows(path);assert len(rr)==32 and [r['id'] for r in rr]==ids
        censored=[]
        for row,original,pre in zip(rr,selected,prefix,strict=True):
            assert row['answer']==original['Answer'] and row['prefix_token_ids']==pre
            assert 0<len(row['new_token_ids'])<=8192 and row['finish_reason'] in ['stop','length']
            if row['finish_reason']=='length':assert len(row['new_token_ids'])==8192
            answer=api['extract_boxed_answer'](row['combined_text'])
            assert answer==row['boxed_answer'] and bool(api['grade_answer'](answer,row['answer']))==row['correct']
            token_answer=api['extract_boxed_answer'](tokenizer.decode(pre+row['new_token_ids'],skip_special_tokens=False))
            assert token_answer==answer,'Stored text and decoded full token answer differ'
            short=api['extract_boxed_answer'](tokenizer.decode(pre+row['new_token_ids'][:2048],skip_special_tokens=False))
            censored.append(bool(api['grade_answer'](short,row['answer'])))
        totals={'items':32,'correct':sum(r['correct'] for r in rr),'truncated':sum(r['finish_reason']=='length' for r in rr),'extraction_failures':sum(r['boxed_answer'] is None for r in rr),'generated_tokens':sum(len(r['new_token_ids']) for r in rr),'supplied_history_tokens':sum(map(len,prefix)),'prefix_already_boxed':sum(api['extract_boxed_answer'](tokenizer.decode(pre)) is not None for pre in prefix)}
        assert all(done['conditions'][name][k]==v for k,v in totals.items()),name
        assert read(folder/(name+'_DONE.json'))==done['conditions'][name]
        records[name]=rr;sources[name]={'items_sha256':sha(path)}
        lengths[name]={'same_trajectory_censored2048_correct':sum(censored),'full8192_correct':totals['correct'],'gained_after2048':sum(not x and r['correct'] for x,r in zip(censored,rr,strict=True)),'lost_after2048':sum(x and not r['correct'] for x,r in zip(censored,rr,strict=True))}
    hours=(datetime.datetime.fromisoformat(job['finished_at'])-datetime.datetime.fromisoformat(job['started_at'])).total_seconds()/3600
    from torch.utils.tensorboard import SummaryWriter
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    tb=root/'logs/copsd_official_10lang/tensorboard/bn4b_teacher_recovery32_20261023';expected={'cost/parent_GPU_hours':hours}
    for name,m in done['conditions'].items():expected[name+'/conditional_accuracy']=m['correct']/32
    with SummaryWriter(str(tb)) as writer:
        for key,value in expected.items():writer.add_scalar(key,value,0)
    accumulator=EventAccumulator(str(tb)).Reload();assert all(abs(accumulator.Scalars(k)[-1].value-v)<1e-6 for k,v in expected.items())
    report={**done,'accepted_at':datetime.datetime.now().isoformat(),'all128_rows_ID_answer_prefix_budget_and_official_score_verified':True,'parent_exit0':True,'frozen_sha256':sha(freeze),'source_records':sources,'Base_history_comparisons':{name:pair(records['Base'],records[name]) for name in ['Clip20','Unclip20','Clean']},'same_trajectory_length_summaries':lengths,'parent_GPU_hours_counted_once':hours,'generated_tokens':sum(m['generated_tokens'] for m in done['conditions'].values()),'TensorBoard_five_scalars_read_back':True,'collector_sha256':sha(Path(__file__)),'interpretation_limits':'32fixed development questions, one generation seed. Privileged reference available; not benchmark accuracy or train-token supervision proof. Censored2048 regrades same8192trajectory, not fresh2048generation. Clean has256fewer suppliedtokens. No selected language or changed trainingprotocol on this diagnostic alone.'}
    with a.output.open('x') as stream:stream.write(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'conditions':done['conditions'],'paired':report['Base_history_comparisons'],'GPU_hours':hours}))


if __name__=='__main__':main()
