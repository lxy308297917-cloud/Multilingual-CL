"""Aggregate four saved SW4B followup200 language analyses and paired termination."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();code=Path(__file__).parent
    math=read(code/'sw4b_followup200_completed_20261022.PUBLIC.json')
    cases,rows={},{}
    for lang in ['sw','en']:
        target='__label__swh_Latn' if lang=='sw' else '__label__eng_Latn'
        for condition in ['Base','SW20']:
            path=a.root/'logs/copsd_official_10lang/sw4b_followup200_language_20261022'/f'{condition}_{lang}.PRIVATE.json'
            data=read(path);assert data['target_label']==target and len(data['runs'])==1
            run=next(iter(data['runs'].values()))
            expected=math['languages'][lang]['conditions'][condition]
            assert run['items_sha256']==expected['source']['items_sha256']==sha(Path(run['items']))
            predictions=run['predictions'];assert len(predictions)==len({str(r['id']) for r in predictions})==200
            assert sum(r['correct'] for r in predictions)==expected['metrics']['single_correct']
            high=lambda r,part:r[part].get('label')==target and r[part].get('prob',0)>=.8
            key=f'{condition}_{lang}'
            cases[key]={'language':run['summary'],'math_correct_and_target_thinking_highconf':sum(r['correct'] and high(r,'thinking') for r in predictions),
                'math_correct_and_target_final_highconf':sum(r['correct'] and high(r,'final') for r in predictions),
                'math_correct_and_both_target_highconf':sum(r['correct'] and high(r,'thinking') and high(r,'final') for r in predictions),
                'analysis_sha256':sha(path),'items_sha256':run['items_sha256']}
            rows[key]={str(r['id']):r for r in map(json.loads,Path(run['items']).read_text().splitlines())}
    paired={}
    for lang in ['sw','en']:
        base,trained=rows['Base_'+lang],rows['SW20_'+lang]
        assert base.keys()==trained.keys()
        gains=[k for k in base if not base[k]['samples'][0]['correct'] and trained[k]['samples'][0]['correct']]
        losses=[k for k in base if base[k]['samples'][0]['correct'] and not trained[k]['samples'][0]['correct']]
        paired[lang]={'gained_IDs':gains,'lost_IDs':losses,
            'Base_finish_of_gains':dict(Counter(base[k]['samples'][0]['finish_reason'] for k in gains)),
            'Base_extraction_failures_of_gains':sum(base[k]['samples'][0]['boxed_answer'] is None for k in gains),
            'trained_finish_of_losses':dict(Counter(trained[k]['samples'][0]['finish_reason'] for k in losses)),
            'trained_extraction_failures_of_losses':sum(trained[k]['samples'][0]['boxed_answer'] is None for k in losses)}
        expected=math['languages'][lang]['paired'];assert len(gains)==expected['gained'] and len(losses)==expected['lost']
    # Same original problem IDs in SW and English permit an access/transfer diagnostic.
    assert rows['Base_sw'].keys()==rows['Base_en'].keys()==rows['SW20_sw'].keys()==rows['SW20_en'].keys()
    sw_gains=paired['sw']['gained_IDs']
    cross_language={'SW_gain_items':len(sw_gains),
        'SW_gains_already_correct_in_Base_English':sum(rows['Base_en'][k]['samples'][0]['correct'] for k in sw_gains),
        'SW_gains_correct_in_trained_English':sum(rows['SW20_en'][k]['samples'][0]['correct'] for k in sw_gains),
        'Base_correct_in_both_languages':sum(rows['Base_sw'][k]['samples'][0]['correct'] and rows['Base_en'][k]['samples'][0]['correct'] for k in rows['Base_sw']),
        'trained_correct_in_both_languages':sum(rows['SW20_sw'][k]['samples'][0]['correct'] and rows['SW20_en'][k]['samples'][0]['correct'] for k in rows['Base_sw']),
        'scope':'One sampledanswer perlanguage andfixedmodel; not proof of latent capability, complete knowledge or a new selection metric.'}
    from torch.utils.tensorboard import SummaryWriter
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    tb=a.root/'logs/copsd_official_10lang/tensorboard/sw4b_followup200_20261022/language'
    writer=SummaryWriter(str(tb))
    for condition,case in cases.items():
        for part in ['thinking','final']:writer.add_scalar(condition+'/'+part+'_target_highconf',case['language'][part]['target_top1_probability_ge_0_8']/200,20)
        writer.add_scalar(condition+'/correct_and_both_target_highconf',case['math_correct_and_both_target_highconf']/200,20)
    writer.close();acc=EventAccumulator(str(tb)).Reload();assert len(acc.Tags()['scalars'])==12
    result={'protocol':'project_SW4B_followup_dev200_language_supplement_v1','completed_at':datetime.now().isoformat(),
        'cases':cases,'paired_termination':paired,'same_question_cross_language':cross_language,'TensorBoard_twelve_language_series_read_back':True,
        'scope':'All denominators200. GlotLID whole-part labels are not calibrated fluency, tokenpurity or proofquality. Joint label/correctness counts are exploratory supplements, never rescoring or revisedselection. No new generation.',
        'summarizer_sha256':sha(Path(__file__))}
    with a.output.open('x') as stream:stream.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:{part:v['language'][part]['target_top1_probability_ge_0_8'] for part in ['thinking','final']} for k,v in cases.items()}))


if __name__=='__main__':main()
