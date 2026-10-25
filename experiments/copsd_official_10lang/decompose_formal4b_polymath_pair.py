"""Describe a completed PolyMath pair without generating or changing scores.

Completion and language transition counts are descriptive associations, not
causal explanations. Formal results must not be used for model selection.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def describe(root, pair_path, language_path):
    pair, language = read(pair_path), read(language_path)
    assert pair['case'].startswith('polymath_') and pair['items'] == 500
    assert not pair['proves_complete_seven_case_stage']
    assert language['model_sha256'] == 'a818b6bd42a628ab47d3dfc1578c7ea615c45381f3494c42535e31e8c4cafc9e'
    rows, predictions = {}, {}
    for side in ['Base', 'model']:
        source = pair[side]
        folder = root / source['output_relative']
        for filename, expected in source['files_sha256'].items():
            assert digest(folder / filename) == expected
        rows[side] = [json.loads(line) for line in (folder / 'ITEMS.jsonl').read_text().splitlines()]
        assert len(rows[side]) == len({r['id'] for r in rows[side]}) == 500
        candidates = [r for r in language['runs'].values()
                      if r['items_sha256'] == source['files_sha256']['ITEMS.jsonl']]
        assert len(candidates) == 1
        predictions[side] = candidates[0]['predictions']
        assert len(predictions[side]) == 500
        for row, pred in zip(rows[side], predictions[side]):
            assert row['id'] == pred['id']
            assert row['samples'][0]['correct'] == pred['correct']

    tiers = {}
    gains, losses, language_transitions, stop_transitions = Counter(), Counter(), Counter(), Counter()
    target = language['target_label']
    def category(pred):
        part = pred['thinking']
        if part.get('label') == target and part.get('prob', 0) >= .8:
            return 'target_highconf'
        if part.get('label') == '__label__eng_Latn':
            return 'English_top1'
        return 'other_or_lower_confidence'

    for i, (base, model) in enumerate(zip(rows['Base'], rows['model'])):
        assert all(base[k] == model[k] for k in ['id', 'question', 'answer', 'prompt', 'split'])
        b, m = base['samples'][0], model['samples'][0]
        assert b['seed'] == m['seed']
        tier = tiers.setdefault(base['split'], {'items': 0, 'Base': Counter(), 'model': Counter(), 'deltas': []})
        tier['items'] += 1
        delta = int(m['correct']) - int(b['correct'])
        tier['deltas'].append(delta)
        for side, sample in [('Base', b), ('model', m)]:
            tier[side].update(correct=int(sample['correct']), generated_tokens=sample['tokens'],
                              truncated=int(sample['finish_reason'] == 'length'),
                              normally_terminated=int(sample['finish_reason'] == 'stop'),
                              extraction_failures=int(sample['boxed_answer'] is None))
        if delta:
            change = gains if delta == 1 else losses
            change['items'] += 1
            change['Base_truncated'] += b['finish_reason'] == 'length'
            change['Base_extraction_failed'] += b['boxed_answer'] is None
            change['model_normally_terminated'] += m['finish_reason'] == 'stop'
            change['model_truncated'] += m['finish_reason'] == 'length'
            change[category(predictions['model'][i])] += 1
            change['Base_' + category(predictions['Base'][i])] += 1
        state = 'gained' if delta == 1 else 'lost' if delta == -1 else 'unchanged'
        language_transitions[(category(predictions['Base'][i]), category(predictions['model'][i]), state)] += 1
        stop_transitions[(b['finish_reason'], m['finish_reason'], state)] += 1

    assert set(tiers) == {'low', 'medium', 'high', 'top'}
    rng = np.random.default_rng(20261025)
    for tier in tiers.values():
        assert tier['items'] == 125
        delta = np.asarray(tier.pop('deltas'), dtype=np.int8)
        tier['gained'] = int((delta == 1).sum())
        tier['lost'] = int((delta == -1).sum())
        tier['delta_pp'] = float(delta.mean() * 100)
        draws = delta[rng.integers(0, 125, size=(10000, 125))].mean(axis=1) * 100
        tier['paired_question_CI95_pp'] = np.quantile(draws, [.025, .975]).tolist()
    assert gains['items'] == pair['gained'] and losses['items'] == pair['lost']
    for side in ['Base', 'model']:
        assert sum(t[side]['correct'] for t in tiers.values()) == pair[side]['metrics']['single_correct']
        assert sum(t[side]['generated_tokens'] for t in tiers.values()) == pair[side]['metrics']['generated_tokens']
    return {
        'model_id': pair['model_id'], 'case': pair['case'], 'items': 500,
        'target_label': target, 'difficulty_tiers': tiers,
        'gained_description': gains, 'lost_description': losses,
        'language_transition_counts': [dict(Base_thinking=b, model_thinking=m, math_change=s, items=n)
                                      for (b, m, s), n in sorted(language_transitions.items())],
        'termination_transition_counts': [dict(Base_finish=b, model_finish=m, math_change=s, items=n)
                                          for (b, m, s), n in sorted(stop_transitions.items())],
        'source_sha256': {'pair_receipt': digest(pair_path), 'language_receipt': digest(language_path),
                          'collector': digest(Path(__file__))},
        'new_generation': False, 'original_math_scores_preserved': True,
        'complete_seven_case_stage': False,
        'scope': 'Descriptive associations in one completed fixed-model PolyMath pair. All500questions retained, four125-question tiers. Confidence intervals are fixed-model paired-question bootstrap, one training seed and no multiplicity correction. Conditioning on language or stopping changes the sample subset and does not establish why training helped. Classifier labels do not prove fluency or per-token language purity. Do not select checkpoints, languages or training data using this formal decomposition.'
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--pair', type=Path, required=True)
    parser.add_argument('--language', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    result = describe(args.root, args.pair, args.language)
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({k: result[k] for k in ['model_id', 'gained_description', 'lost_description']}))


if __name__ == '__main__':
    main()
