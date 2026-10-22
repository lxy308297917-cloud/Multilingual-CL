"""Combine saved GlotLID outputs for the fixed SW4B 2x2; never alter math scores."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    code = Path(__file__).parent
    math = json.loads((code/'sw4b_prompt_interaction_completed_20261022.PUBLIC.json').read_text())
    names = {'Base_SWprefix': 'Base_sw', 'Base_ENprefix': 'Base_ENprefix_sw',
             'SW20_SWprefix': 'SW20_sw', 'SW20_ENprefix': 'SW20_ENprefix_sw'}
    cases = {}
    for condition, name in names.items():
        path = a.root/'logs/copsd_official_10lang/sw4b_server_language_20261022'/f'{name}.PRIVATE.json'
        data = json.loads(path.read_text())
        assert data['target_label'] == '__label__swh_Latn' and len(data['runs']) == 1
        run = next(iter(data['runs'].values()))
        assert run['items_sha256'] == math['cases'][condition]['ITEMS_sha256']
        predictions = run['predictions']
        assert len(predictions) == len({str(r['id']) for r in predictions}) == 100
        assert sum(r['correct'] for r in predictions) == math['cases'][condition]['single_correct']
        def sw_highconf(r, part):
            item = r[part]
            return item.get('label') == '__label__swh_Latn' and item.get('prob', 0) >= .8
        cases[condition] = {'language': run['summary'],
            'math_correct': sum(r['correct'] for r in predictions),
            'math_correct_and_thinking_SW_highconf': sum(r['correct'] and sw_highconf(r, 'thinking') for r in predictions),
            'math_correct_and_final_SW_highconf': sum(r['correct'] and sw_highconf(r, 'final') for r in predictions),
            'math_correct_and_both_parts_SW_highconf': sum(r['correct'] and sw_highconf(r, 'thinking') and sw_highconf(r, 'final') for r in predictions),
            'analysis_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'source_items_sha256': run['items_sha256']}
    from torch.utils.tensorboard import SummaryWriter
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    tb = a.root/'logs/copsd_official_10lang/tensorboard/sw4b_prompt_interaction_20261022/language'
    writer = SummaryWriter(str(tb))
    for condition, case in cases.items():
        for part in ['thinking', 'final']:
            writer.add_scalar(condition+'/'+part+'_SW_highconf', case['language'][part]['target_top1_probability_ge_0_8']/100, 20)
        writer.add_scalar(condition+'/correct_and_both_SW_highconf', case['math_correct_and_both_parts_SW_highconf']/100, 20)
    writer.close()
    accumulator = EventAccumulator(str(tb)).Reload()
    assert len(accumulator.Tags()['scalars']) == 12
    result = {'protocol': 'SW4B_fixed2x2_language_supplement_v1', 'completed_at': datetime.now().isoformat(),
        'cases': cases, 'TensorBoard_twelve_language_series_read_back': True,
        'scope': 'All denominators100; GlotLID is not calibrated fluency, proof quality or per-token purity. Joint correct/highconf counts are exploratory diagnostics, not replacements for official boxed grading or a revised selection rule. No new generations.',
        'summarizer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with a.output.open('x') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:{name:v[name] for name in ['math_correct','math_correct_and_thinking_SW_highconf','math_correct_and_final_SW_highconf','math_correct_and_both_parts_SW_highconf']} for k,v in cases.items()}))


if __name__ == '__main__':
    main()
