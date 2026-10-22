"""Accept the two fixed SW4B PolyMath500/native32 formal reports; no test-driven selection."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

from evaluate_transformers import load_items
from evaluate_vllm import question_seed


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.root;code=Path(__file__).parent
    freeze_path=code/'sw4b_PolyMath_c32_frozen_20261023.PUBLIC.json';f=read(freeze_path)
    for name,digest in f['runtime_code_sha256'].items():assert sha(code/name)==digest,name
    source=root/'COPSD-official-20261015/polymath_eval/evaluate_math.py'
    locked=read(root/'logs/copsd_official_10lang/eval_vllm_runtime_20261019.CONFIG.json')
    assert sha(source)==locked['source_sha256']['polymath_eval/evaluate_math.py']
    cases,costs,all_rows={},[],{}
    for condition,spec in f['cases'].items():
        for benchmark,run_id in spec['run_ids'].items():
            tag='mgsm' if benchmark=='mgsm_rev2' else 'gsm'
            queue=f'copsd-sw4b-poly32-{condition.lower()}-v1'
            state=read(root/'gpu-queue/queues'/queue/'state.json');job=state['job']
            assert state['phase']=='completed' and job['exit_code']==0
            assert job['script_sha256']==f['runtime_code_sha256']['run_sw4b_polymath8192_c32.sh']
            assert len(job['gpu_uuids'])==1 and job['config']['max_user_gpus']==4
            assert job['config']['job']['args']==[condition]
            hours=(datetime.fromisoformat(job['finished_at'])-datetime.fromisoformat(job['started_at'])).total_seconds()/3600
            costs.append({'queue':queue,'parent_GPU_hours':hours,'training_updates':0})
            folder=root/'logs/copsd_official_10lang/evaluation_vllm8192_4b_poly_c32_v1'/run_id
            cfg,done=read(folder/'CONFIG.json'),read(folder/'DONE.json')
            data=root/f['inputs'][benchmark]['relative'];count=f['inputs'][benchmark]['items'];lang=f['inputs'][benchmark]['language']
            assert cfg['case']==condition and cfg['benchmark']==benchmark and cfg['language']==lang
            assert cfg['protocol']==done['protocol']==f['protocol']
            assert cfg['model_revision']==f['model_revision'] and cfg['cohort']=='independent_4B'
            assert cfg['frozen_stage_sha256']==sha(freeze_path)
            assert cfg['input_sha256']==f['inputs'][benchmark]['sha256']==sha(data)
            assert cfg['evaluator_sha256']==f['runtime_code_sha256']['evaluate_sw4b_polymath8192_c32.py']
            assert cfg['official_source_sha256']==sha(source)
            assert cfg['model']==str(root/'models/Qwen3-4B')
            assert cfg['adapter']==(str(root/spec['adapter_relative']) if spec['adapter_relative'] else None)
            assert cfg['adapter_sha256']==spec['adapter_sha256']
            assert cfg['checkpoint_step']==(20 if condition=='SW20' else None)
            if condition=='SW20':assert sha(root/spec['adapter_relative']/'adapter_model.safetensors')==spec['adapter_sha256']
            for key in ['samples','max_new_tokens','max_num_seqs','temperature','top_p','top_k','seed']:assert cfg[key]==f[key],key
            rows=read_jsonl(folder/'ITEMS.jsonl')
            originals=list(load_items(data,'mgsm_rev2' if benchmark=='mgsm_rev2' else 'polymath_project',lang))
            assert len(rows)==len(originals)==len({str(r['id']) for r in rows})==count
            assert done['formal_full_test'] and done['n']==1 and done['generation_count']==done['items']==count
            for row,original in zip(rows,originals):
                assert all(row[k]==original[k] for k in ['id','question','answer'])
                assert len(row['samples'])==1
                sample=row['samples'][0]
                assert sample['seed']==question_seed(42,lang,row['id'])
                assert sample['tokens']==len(sample['token_ids'])<=8192
                assert type(sample['correct']) is bool
            samples=[r['samples'][0] for r in rows]
            actual={'single_correct':sum(s['correct'] for s in samples),'generated_tokens':sum(s['tokens'] for s in samples),
                'truncated':sum(s['finish_reason']=='length' for s in samples),'normal_terminations':sum(s['finish_reason']=='stop' for s in samples),
                'answer_extraction_failures':sum(s['boxed_answer'] is None for s in samples)}
            assert all(done[k]==v for k,v in actual.items())
            assert done['single_accuracy']==done['single_correct']/count
            assert done['actual_max_num_seqs']==32 and benchmark=='polymath'
            expected_tiers={}
            for row in rows:
                tier=expected_tiers.setdefault(row['split'],{'items':0,'single_correct':0})
                tier['items']+=1;tier['single_correct']+=row['samples'][0]['correct']
            assert len(expected_tiers)==4 and all(v['items']==125 for v in expected_tiers.values())
            assert set(expected_tiers)==set(done['scores_by_difficulty'])
            for tier,values in expected_tiers.items():
                assert done['scores_by_difficulty'][tier]=={**values,'single_accuracy':values['single_correct']/125}
            cases.setdefault(benchmark,{})[condition]={'metrics':done,'ITEMS_sha256':sha(folder/'ITEMS.jsonl'),'CONFIG_sha256':sha(folder/'CONFIG.json')}
            all_rows[(benchmark,condition)]=(cfg,rows)
    paired={}
    for benchmark in f['inputs']:
        bc,b=all_rows[(benchmark,'Base')];tc,t=all_rows[(benchmark,'SW20')]
        for key in ['input_sha256','model','model_revision','official_source_sha256','evaluator_sha256','packages','max_new_tokens','max_num_seqs','samples','temperature','top_p','top_k','seed']:assert bc[key]==tc[key],key
        assert all(all(x[k]==y[k] for k in ['id','question','answer','prompt']) and x['samples'][0]['seed']==y['samples'][0]['seed'] for x,y in zip(b,t))
        gains=[str(x['id']) for x,y in zip(b,t) if not x['samples'][0]['correct'] and y['samples'][0]['correct']]
        losses=[str(x['id']) for x,y in zip(b,t) if x['samples'][0]['correct'] and not y['samples'][0]['correct']]
        paired[benchmark]={'gained_IDs':gains,'lost_IDs':losses,'delta_correct':len(gains)-len(losses)}
    hours=sum(r['parent_GPU_hours'] for r in costs);tokens=sum(case['metrics']['generated_tokens'] for c in cases.values() for case in c.values())
    from torch.utils.tensorboard import SummaryWriter
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    tb=root/'logs/copsd_official_10lang/tensorboard/sw4b_PolyMath_c32_20261023';writer=SummaryWriter(str(tb))
    for benchmark,conditions in cases.items():
        for condition,data in conditions.items():writer.add_scalar(benchmark+'/'+condition+'_accuracy',data['metrics']['single_accuracy'],20)
    writer.add_scalar('cost/unique_parent_GPU_hours',hours,20);writer.add_scalar('cost/generated_tokens',tokens,20);writer.close()
    acc=EventAccumulator(str(tb)).Reload();assert len(acc.Tags()['scalars'])==4
    for benchmark,conditions in cases.items():
        for condition,data in conditions.items():assert abs(acc.Scalars(benchmark+'/'+condition+'_accuracy')[-1].value-data['metrics']['single_accuracy'])<1e-7
    result={'protocol':f['protocol'],'completed_at':datetime.now().isoformat(),'model_revision':f['model_revision'],
        'cases':cases,'paired':paired,'unique_parent_costs':costs,'parent_GPU_hours':hours,'new_generated_tokens':tokens,
        'TensorBoard_two_accuracies_two_costs_read_back':True,
        'scope':'FixedBase/checkpoint20 fullformal PolyMathSW500/n1/8192/native32, fourtiers125each. Bothmodels share32, no averagingwithMGSM/GSMnative16. No officialpass12, trainingseedconfirmation or3-language selection. Test scores usedforreporting only, not config/checkpoint/language selection.',
        'summarizer_sha256':sha(Path(__file__))}
    with a.output.open('x') as stream:stream.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:{c:v['metrics']['single_correct'] for c,v in conditions.items()} for k,conditions in cases.items()}))


if __name__=='__main__':main()
