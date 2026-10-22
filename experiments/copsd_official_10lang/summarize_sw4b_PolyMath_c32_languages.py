"""Summarize accepted SW4B PolyMath500 native32 language and termination evidence."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
from collect_sw4b_polymath8192_c32 import read_jsonl


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    code = Path(__file__).parent
    accepted_path = code / 'sw4b_PolyMath_c32_completed_20261023.PUBLIC.json'
    accepted = read(accepted_path)
    cases, paired = {}, {}
    for benchmark, conditions in accepted['cases'].items():
        assert benchmark == 'polymath'
        target = '__label__swh_Latn'
        rows = {}
        for condition, expected in conditions.items():
            path = args.root / 'logs/copsd_official_10lang/sw4b_PolyMath_c32_language_20261023' / f'{condition}_sw.PRIVATE.json'
            analysis = read(path)
            assert analysis['target_label'] == target and len(analysis['runs']) == 1
            run = next(iter(analysis['runs'].values()))
            assert run['items_sha256'] == expected['ITEMS_sha256'] == sha(Path(run['items']))
            count = expected['metrics']['items']
            predictions = run['predictions']
            assert len(predictions) == len({str(x['id']) for x in predictions}) == count
            assert sum(x['correct'] for x in predictions) == expected['metrics']['single_correct']
            high = lambda row, part: row[part].get('label') == target and row[part].get('prob', 0) >= .8
            cases[f'{condition}_{benchmark}'] = {
                'items': count, 'language': run['summary'],
                'math_correct_and_target_thinking_highconf': sum(x['correct'] and high(x, 'thinking') for x in predictions),
                'math_correct_and_target_final_highconf': sum(x['correct'] and high(x, 'final') for x in predictions),
                'math_correct_and_both_target_highconf': sum(x['correct'] and high(x, 'thinking') and high(x, 'final') for x in predictions),
                'analysis_sha256': sha(path), 'items_sha256': run['items_sha256']}
            rows[condition] = {str(x['id']): x for x in read_jsonl(Path(run['items']))}
        base, trained = rows['Base'], rows['SW20']
        assert base.keys() == trained.keys()
        gains = [k for k in base if not base[k]['samples'][0]['correct'] and trained[k]['samples'][0]['correct']]
        losses = [k for k in base if base[k]['samples'][0]['correct'] and not trained[k]['samples'][0]['correct']]
        assert gains == accepted['paired'][benchmark]['gained_IDs']
        assert losses == accepted['paired'][benchmark]['lost_IDs']
        paired[benchmark] = {
            'gained_IDs': gains, 'lost_IDs': losses,
            'Base_finish_of_gains': dict(Counter(base[k]['samples'][0]['finish_reason'] for k in gains)),
            'Base_extraction_failures_of_gains': sum(base[k]['samples'][0]['boxed_answer'] is None for k in gains),
            'trained_finish_of_losses': dict(Counter(trained[k]['samples'][0]['finish_reason'] for k in losses)),
            'trained_extraction_failures_of_losses': sum(trained[k]['samples'][0]['boxed_answer'] is None for k in losses)}
    from torch.utils.tensorboard import SummaryWriter
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    tb = args.root / 'logs/copsd_official_10lang/tensorboard/sw4b_PolyMath_c32_20261023/language'
    writer = SummaryWriter(str(tb))
    values = {}
    for key, case in cases.items():
        for part in ['thinking', 'final']:
            values[key + '/' + part + '_target_highconf'] = case['language'][part]['target_top1_probability_ge_0_8'] / case['items']
        values[key + '/correct_and_both_target_highconf'] = case['math_correct_and_both_target_highconf'] / case['items']
    for key, value in values.items():
        writer.add_scalar(key, value, 20)
    writer.close()
    accumulator = EventAccumulator(str(tb)).Reload()
    assert len(accumulator.Tags()['scalars']) == 6
    for key, value in values.items():
        assert abs(accumulator.Scalars(key)[-1].value - value) < 1e-7
    result = {
        'completed_at': datetime.now().isoformat(), 'protocol': accepted['protocol'],
        'cases': cases, 'paired_termination': paired,
        'TensorBoard_six_language_series_read_back': True,
        'accepted_formal_report_sha256': sha(accepted_path), 'summarizer_sha256': sha(Path(__file__)),
        'scope': 'Descriptive analysis of unchanged formal scores. SW PolyMath500 only, one fixed4BBase/seed42modelpair; no seed-population inference or comparison with native16 MGSM/GSM protocol averages. GlotLID whole-part labels are not calibrated fluency, token purity or proof quality. Short/numeric final answers may lack a reliable language label; missing labels are not language failures. No new generations, no rescoring, no test-driven checkpoint/language selection.'}
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'accepted_cases': list(cases), 'new_generations': 0}))


if __name__ == '__main__':
    main()
