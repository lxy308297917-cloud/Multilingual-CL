"""Describe difficulty and completion of already audited SW PolyMath outputs.

No generation, regrading, filtering or checkpoint selection is performed.
"""
import argparse
from collections import Counter
import datetime
import json
from pathlib import Path

from sequential_evaluation import digest, read, write_new


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--paired-report', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--language-report', type=Path)
    a = p.parse_args()
    report = read(a.paired_report)
    assert report['benchmark'] == 'polymath' and report['all_requested_stage_outputs_verified']
    assert report['protocol'] == 'project_formal_single_vllm8192_v1'
    assert set(report['sources']) == {'base', 'seed42'}
    runs = {}
    for model, source in report['sources'].items():
        folder = a.root / 'logs/copsd_official_10lang/evaluation_vllm8192' / source['run_id']
        assert digest(folder / 'ITEMS.jsonl') == source['raw_items_sha256']
        assert digest(folder / 'CONFIG.json') == source['config_sha256']
        assert digest(folder / 'DONE.json') == source['DONE_sha256']
        runs[model] = [json.loads(line) for line in (folder / 'ITEMS.jsonl').read_text().splitlines()]
        assert len(runs[model]) == len({row['id'] for row in runs[model]}) == 500
    tiers, changes = {}, Counter()
    for base, trained in zip(runs['base'], runs['seed42']):
        assert all(base[key] == trained[key] for key in ['id', 'question', 'answer', 'prompt', 'split'])
        tier = base['split']
        values = tiers.setdefault(tier, {'items': 0, 'base': Counter(), 'seed42': Counter(), 'paired': Counter()})
        values['items'] += 1
        for model, row in [('base', base), ('seed42', trained)]:
            sample = row['samples'][0]
            values[model].update({'correct': int(sample['correct']),
                'normal_terminations': int(sample['finish_reason'] == 'stop'),
                'truncated': int(sample['finish_reason'] == 'length'),
                'answer_extraction_failures': int(sample['boxed_answer'] is None),
                'correct_and_normally_terminated': int(sample['correct'] and sample['finish_reason'] == 'stop'),
                'correct_but_truncated': int(sample['correct'] and sample['finish_reason'] == 'length')})
        b, t = base['samples'][0], trained['samples'][0]
        if b['correct'] != t['correct']:
            name = 'gained' if t['correct'] else 'lost'
            values['paired'][name] += 1
            changes[name] += 1
            if name == 'gained':
                changes['gained_where_Base_had_no_answer'] += b['boxed_answer'] is None
                changes['gained_where_Base_answer_was_wrong'] += b['boxed_answer'] is not None
                changes['gained_and_trained_output_normally_terminated'] += t['finish_reason'] == 'stop'
    assert len(tiers) == 4 and all(t['items'] == 125 for t in tiers.values())
    assert sum(t['base']['correct'] for t in tiers.values()) == 8
    assert sum(t['seed42']['correct'] for t in tiers.values()) == 29
    assert changes['gained'] == 25 and changes['lost'] == 4
    language_summary = None
    if a.language_report:
        supplement = read(a.language_report)
        assert supplement['target_label'] == '__label__swh_Latn'
        assert digest(Path(supplement['model'])) == supplement['model_sha256']
        language_summary = {}
        for model, source in report['sources'].items():
            entry = supplement['runs'][source['run_id']]
            assert entry['items_sha256'] == source['raw_items_sha256']
            assert len(entry['predictions']) == 500
            assert [r['id'] for r in entry['predictions']] == [r['id'] for r in runs[model]]
            assert [r['correct'] for r in entry['predictions']] == [r['samples'][0]['correct'] for r in runs[model]]
            language_summary[model] = entry['summary']
    write_new(a.output, {'observed_at': datetime.datetime.now().isoformat(),
        'paired_report_sha256': digest(a.paired_report), 'protocol': report['protocol'],
        'source_run_ids': {k: v['run_id'] for k, v in report['sources'].items()},
        'difficulty_tiers': tiers, 'transition_counts': changes,
        'language_ID_summary': language_summary,
        'language_report_sha256': digest(a.language_report) if a.language_report else None,
        'language_ID_caveat': 'Whole-section top1 labels are not calibrated fluency, semantic validity or per-token language retention. Missing final section is preserved in the denominator. Math scores are never filtered by language ID.',
        'new_generation': False, 'math_scoring_changed': False,
        'scope': 'One SW20 seed42 formal PolyMath pair, not seed43 replication or English retention. Completion cross-counts are descriptive; conditioning accuracy on stopping would change the sample set. Do not select languages/checkpoints using this formal decomposition.'})
    print(json.dumps({'tiers': tiers, 'changes': changes}))


if __name__ == '__main__':
    main()
