"""Summarize accepted SW4B two-seed language and paired termination evidence."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path


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
    math_path = code / 'sw4b_seed43_followup200_completed_20261023.PUBLIC.json'
    math = read(math_path)
    old = read(code / 'sw4b_followup200_language_20261022.PUBLIC.json')
    old_math = read(code / 'sw4b_followup200_completed_20261022.PUBLIC.json')
    cases, paired = {}, {}
    for lang in ['sw', 'en']:
        target = '__label__swh_Latn' if lang == 'sw' else '__label__eng_Latn'
        for case, old_case in [('Base', 'Base'), ('seed42', 'SW20')]:
            value = old['cases'][old_case + '_' + lang]
            assert value['items_sha256'] == old_math['languages'][lang]['conditions'][old_case]['source']['items_sha256']
            cases[case + '_' + lang] = value
        path = args.root / 'logs/copsd_official_10lang/sw4b_seed43_followup200_language_20261023' / f'SW43_{lang}.PRIVATE.json'
        analysis = read(path)
        assert analysis['target_label'] == target and len(analysis['runs']) == 1
        run = next(iter(analysis['runs'].values()))
        assert run['items_sha256'] == math['languages'][lang]['seed43_source']['items_sha256'] == sha(Path(run['items']))
        rows = run['predictions']
        assert len(rows) == len({str(x['id']) for x in rows}) == 200
        assert sum(x['correct'] for x in rows) == math['languages'][lang]['seed43_correct']
        high = lambda x, part: x[part].get('label') == target and x[part].get('prob', 0) >= .8
        cases['seed43_' + lang] = {
            'language': run['summary'],
            'math_correct_and_target_thinking_highconf': sum(x['correct'] and high(x, 'thinking') for x in rows),
            'math_correct_and_target_final_highconf': sum(x['correct'] and high(x, 'final') for x in rows),
            'math_correct_and_both_target_highconf': sum(x['correct'] and high(x, 'thinking') and high(x, 'final') for x in rows),
            'analysis_sha256': sha(path), 'items_sha256': run['items_sha256']}
        base_path = args.root / 'logs/copsd_official_10lang/sw4b_followup_development200_v1' / old_math['languages'][lang]['conditions']['Base']['source']['run_id'] / 'ITEMS.jsonl'
        assert sha(base_path) == cases['Base_' + lang]['items_sha256']
        base = {x['id']: x for x in map(json.loads, base_path.read_text().splitlines())}
        trained = {x['id']: x for x in map(json.loads, Path(run['items']).read_text().splitlines())}
        assert base.keys() == trained.keys()
        gains = [k for k in base if not base[k]['samples'][0]['correct'] and trained[k]['samples'][0]['correct']]
        losses = [k for k in base if base[k]['samples'][0]['correct'] and not trained[k]['samples'][0]['correct']]
        assert len(gains) == math['languages'][lang]['seed43_pair']['gained']
        assert len(losses) == math['languages'][lang]['seed43_pair']['lost']
        paired[lang] = {
            'gained_IDs': gains, 'lost_IDs': losses,
            'Base_finish_of_gains': dict(Counter(base[k]['samples'][0]['finish_reason'] for k in gains)),
            'Base_extraction_failures_of_gains': sum(base[k]['samples'][0]['boxed_answer'] is None for k in gains),
            'trained_finish_of_losses': dict(Counter(trained[k]['samples'][0]['finish_reason'] for k in losses)),
            'trained_extraction_failures_of_losses': sum(trained[k]['samples'][0]['boxed_answer'] is None for k in losses)}
    partial = read(code / 'sw4b_two_seed_target200_language_partial_20261023.PUBLIC.json')
    assert cases['seed43_sw']['language'] == partial['cases']['seed43']['language']
    from torch.utils.tensorboard import SummaryWriter
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    tb = args.root / 'logs/copsd_official_10lang/tensorboard/sw4b_seed43_confirmation_20261022/language'
    writer = SummaryWriter(str(tb))
    values = {}
    for name, case in cases.items():
        for part in ['thinking', 'final']:
            values[name + '/' + part + '_target_highconf'] = case['language'][part]['target_top1_probability_ge_0_8'] / 200
        values[name + '/correct_and_both_target_highconf'] = case['math_correct_and_both_target_highconf'] / 200
    for name, value in values.items():
        writer.add_scalar(name, value, 20)
    writer.close()
    accumulator = EventAccumulator(str(tb)).Reload()
    assert len(accumulator.Tags()['scalars']) == 18
    for name, value in values.items():
        assert abs(accumulator.Scalars(name)[-1].value - value) < 1e-7
    result = {
        'completed_at': datetime.now().isoformat(), 'protocol': math['protocol'],
        'items_each': 200, 'cases': cases, 'seed43_paired_termination': paired,
        'math_report_sha256': sha(math_path), 'summarizer_sha256': sha(Path(__file__)),
        'TensorBoard_eighteen_language_series_read_back': True,
        'scope': 'Descriptive whole-part GlotLID labels, not calibrated fluency, token purity or proof correctness. Short/numeric final answers can lack reliable labels, not language failures. Fixed questions and two fixed training seeds, not seed-population inference. Original math scoring unchanged; no new GPU generation or cost recount.'}
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(paired))


if __name__ == '__main__':
    main()
