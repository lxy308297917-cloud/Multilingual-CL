"""Check an existing Base dev result before reuse; never generates responses."""
import argparse,hashlib,importlib.metadata,json
from pathlib import Path
from evaluate_vllm import question_seed


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--language',required=True)
    parser.add_argument('--run-id',required=True)
    a=parser.parse_args();lang=a.language.lower()
    assert lang in 'sw te th bn zh ja ru de fr es en'.split()
    assert all(c.isalnum() or c in '_-' for c in a.run_id)
    path=a.root/'logs/copsd_official_10lang/development_vllm'/a.run_id
    cfg=json.loads((path/'CONFIG.json').read_text());done=json.loads((path/'DONE.json').read_text())
    assert cfg['protocol']=='project_dev_single_vllm_v1' and cfg['adapter'] is None
    expected={'language':lang,'samples':1,'max_new_tokens':2048,'temperature':1.,'top_p':.95,'top_k':-1,'seed':42,'enable_thinking':True}
    for k,v in expected.items():assert cfg[k]==v,(k,cfg[k],v)
    assert Path(cfg['model']).resolve()==(a.root/'models/Qwen3-1.7B').resolve()
    data=a.root/f'data/copsd_official_20261015/adapted_v2_ordered/project_dev_100_{lang if lang!="en" else "sw"}.json'
    source=a.root/'COPSD-official-20261015/polymath_eval/evaluate_math.py'
    for key,file in [('input',data),('source',source)]:
        assert Path(cfg[key]).resolve()==file.resolve()
        hash_key='input_sha256' if key=='input' else 'official_source_sha256'
        assert cfg[hash_key]==hashlib.sha256(file.read_bytes()).hexdigest()
    evaluator=Path(__file__).with_name('evaluate_dev_vllm.py')
    assert cfg['evaluator_sha256']==hashlib.sha256(evaluator.read_bytes()).hexdigest()
    for package,version in cfg['packages'].items():assert importlib.metadata.version(package)==version,package
    raw=json.loads(data.read_text());rows=[json.loads(s) for s in (path/'ITEMS.jsonl').read_text().splitlines()]
    assert len(rows)==len(raw)==done['items']==100 and done['n']==1 and not done['formal_test']
    for row,original in zip(rows,raw):
        assert (row['id'],row['question'],row['answer'])==(original['problem_id'],original['problem'] if lang=='en' else original['problem_'+lang],original['Answer'])
        assert len(row['samples'])==1 and row['samples'][0]['seed']==question_seed(42,lang,row['id'])
    assert sum(r['samples'][0]['correct'] for r in rows)==done['single_correct']
    assert sum(r['samples'][0]['tokens'] for r in rows)==done['generated_tokens']
    print(json.dumps({'cached_Base_development_verified':True,'language':lang,'items':100,'single_correct':done['single_correct'],'inference_repeated':False}))


if __name__=='__main__':main()
