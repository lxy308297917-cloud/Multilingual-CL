"""Compare two accepted ZH4B development pairs on the same 200 questions.

No generation or model/queue mutation. Bootstrap resamples questions, jointly
for both fixed trained models; it is not uncertainty over a seed population.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import numpy as np


def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();code=Path(__file__).parent
    paths=[code/'zh4b_followup200_completed_20261023.PUBLIC.json',code/'zh4b_seed43_followup200_completed_20261023.PUBLIC.json']
    reports=[read(x) for x in paths]
    freezes=[read(code/'zh4b_eval_frozen_20261023.PUBLIC.json'),read(code/'zh4b_seed43_eval_frozen_20261023.PUBLIC.json')]
    assert reports[1]['training_seed']==reports[1]['data_seed']==43 and reports[1]['evaluation_question_seed']==42
    assert freezes[0]['selected_IDs']==freezes[1]['selected_IDs'] and len(freezes[0]['selected_IDs'])==200
    output={}
    for lang in ['zh','en']:
        records=[r['languages'][lang] for r in reports]
        bases=[r['conditions']['Base'] for r in records]
        assert bases[0]['source']==bases[1]['source'] and bases[0]['metrics']==bases[1]['metrics']
        rows=[];configs=[]
        for i,case in [(0,'Base'),(0,'ZH20'),(1,'ZH43')]:
            cell=records[i]['conditions'][case];source=cell['source']
            if case=='Base':
                folder=a.root/freezes[1]['cached_Base_ZH' if lang=='zh' else 'cached_Base_English']['relative']
            else:
                parent='zh4b_followup_development200' if i==0 else 'zh4b_seed43_followup_development200'
                folder=a.root/'logs/copsd_official_10lang'/parent/source['run_id']
            assert sha(folder/'ITEMS.jsonl')==source['items_sha256'] and sha(folder/'CONFIG.json')==source['config_sha256']
            cfg=read(folder/'CONFIG.json');configs.append(cfg)
            raw=[json.loads(x) for x in (folder/'ITEMS.jsonl').read_text().splitlines()];rows.append(raw)
            assert [x['id'] for x in raw]==freezes[0]['selected_IDs']
            assert sum(x['samples'][0]['correct'] for x in raw)==cell['metrics']['single_correct']
        for cfg in configs[1:]:
            for key in ['language','model_revision','items','samples','max_new_tokens','max_num_seqs','temperature','top_p','top_k','seed','input_sha256','official_source_sha256','packages']:
                assert cfg[key]==configs[0][key],key
        for b,x,y in zip(*rows):
            assert all(b[k]==x[k]==y[k] for k in ['id','question','answer','prompt'])
            assert b['samples'][0]['seed']==x['samples'][0]['seed']==y['samples'][0]['seed']
        correct=np.array([[x['samples'][0]['correct'] for x in raw] for raw in rows],dtype=int)
        deltas=correct[1:]-correct[0];mean=deltas.mean(0)
        rng=np.random.default_rng(20261023);boot=mean[rng.integers(0,200,size=(10000,200))].mean(1)*100
        ci=np.quantile(boot,[.025,.975]).tolist()
        patterns=Counter(''.join(str(int(x)) for x in item) for item in correct.T)
        per_seed={}
        for index,seed in [(1,42),(2,43)]:
            changed={name:[i for i in range(200) if condition(correct[0,i],correct[index,i])] for name,condition in [('gained',lambda b,m:not b and m),('lost',lambda b,m:b and not m)]}
            per_seed[str(seed)]={name:{'count':len(indices),'Base_finish':dict(Counter(rows[0][i]['samples'][0]['finish_reason'] for i in indices)),'trained_finish':dict(Counter(rows[index][i]['samples'][0]['finish_reason'] for i in indices))} for name,indices in changed.items()}
        output[lang]={'correct_of200':dict(zip(['Base','seed42','seed43'],correct.sum(1).tolist())),
            'both_training_seed_deltas_positive':bool((deltas.sum(1)>0).all()),
            'two_fixed_model_mean_delta_pp':float(mean.mean()*100),
            'two_fixed_model_mean_paired_question_bootstrap95_CI_pp':ci,
            'mean_question_CI_lower_strictly_positive':bool(ci[0]>0),
            'Base_seed42_seed43_correctness_patterns':dict(patterns),
            'both_seeds_gain_same_questions':patterns['011'],'both_seeds_lose_same_questions':patterns['100'],
            'per_seed_termination_changes':per_seed}
    result={'completed_at':datetime.now().isoformat(),'items_each':200,'languages':output,
        'accepted_report_sha256':{p.name:sha(p) for p in paths},'summarizer_sha256':sha(Path(__file__)),
        'scope':'Two fixed trainedmodels, shared questions and identical cachedBase. Jointquestion bootstrap; not trainingseedpopulation, blindtest, multiplicityadjusted inference or automatic language selection. Review languagekeeping, Englishtradeoff, termination, teacher signal and cost before promotion. No newGPU work or cost recount; original scores unchanged.'}
    with a.output.open('x') as stream:stream.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(output,ensure_ascii=False))


if __name__=='__main__':main()
