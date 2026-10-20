"""CPU-only answer/termination decomposition of the completed seed42 BN control."""
import argparse
from collections import Counter
import datetime
import json
from pathlib import Path

from summarize_saved_candidates8192 import load_eval, pair


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    runs = {'Base': load_eval(a.root, 'copsd-bn-base-dev8192-v1',
                             'bn_base_dev100_8192_20261020_v1', 'bn', None)}
    for mode in ['clip', 'unclip']:
        name = f'bn_17b_{mode}20_horizon100_seed42_20261020_v1'
        cp = a.root / 'models/checkpoint/copsd_bn_clip_control_v1' / name / 'checkpoint-20'
        runs[mode] = load_eval(a.root, f'copsd-bn-{mode}20-dev8192-v1', name + '_dev8192', 'bn', cp)
    pair(runs['Base'], runs['clip'])
    pair(runs['Base'], runs['unclip'])
    stats = {}
    for name, (_, metrics, rows, _) in runs.items():
        samples = [r['samples'][0] for r in rows]
        assert all(s['finish_reason'] in ['stop', 'length'] for s in samples)
        cells = Counter((bool(s['correct']), s['finish_reason'], s['boxed_answer'] is None) for s in samples)
        stats[name] = {
            'official_correct': metrics['single_correct'],
            'correct_and_normal_stop': sum(s['correct'] and s['finish_reason'] == 'stop' for s in samples),
            'correct_and_length_stop': sum(s['correct'] and s['finish_reason'] == 'length' for s in samples),
            'cells': [{'correct': k[0], 'finish_reason': k[1], 'answer_missing': k[2], 'count': n}
                      for k, n in sorted(cells.items())]}
    gained = Counter()
    for a0, a1 in zip(runs['clip'][2], runs['unclip'][2]):
        c, u = a0['samples'][0], a1['samples'][0]
        if not c['correct'] and u['correct']:
            gained[(c['finish_reason'], c['boxed_answer'] is None, u['finish_reason'])] += 1
    out = {
        'observed_at': datetime.datetime.now().isoformat(),
        'protocol': 'project_dev_single_vllm8192_v2',
        'training_seed': 42, 'training_language': 'bn', 'items_each': 100,
        'completion_decomposition': stats,
        'unclip_gained_correct_transitions': [
            {'clip_finish_reason': k[0], 'clip_answer_missing': k[1],
             'unclip_finish_reason': k[2], 'count': n} for k, n in sorted(gained.items())],
        'paired_official_scores': pair(runs['clip'], runs['unclip']),
        'source_parents_exit0_configs_raw_counts_prompts_seeds_verified': True,
        'new_training': False, 'new_inference': False,
        'scope': 'Diagnostic partition only: official answer extraction and scores are unchanged. Correct-and-normal-stop is an additional descriptive count, not a replacement benchmark. Conditional accuracy on stopped responses would be selection-biased and is not used to infer intrinsic mathematical ability. Correct boxes may exist in truncated output; target gain alone does not establish a complete or stable repair.'}
    with a.output.open('x') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({k: {x: v[x] for x in ['official_correct', 'correct_and_normal_stop', 'correct_and_length_stop']}
                      for k, v in stats.items()}))


if __name__ == '__main__':
    main()
