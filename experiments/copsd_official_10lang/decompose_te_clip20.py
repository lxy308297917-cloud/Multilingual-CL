"""CPU-only TE completion/language supplement; official math scores stay unchanged."""
import argparse
from collections import Counter
import datetime
import hashlib
import json
from pathlib import Path
from summarize_saved_candidates8192 import load_eval, pair, read


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--language-report', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    code = Path(__file__).parent
    frozen = read(code / 'te_clip_control20_frozen_20261021.PUBLIC.json')
    names = {'Base': 'te_saved20_screen_base_recovered_dev100_8192_20261020_v2'}
    runs = {'Base': load_eval(a.root, 'copsd-te-saved20-screen8192-v1', names['Base'], 'te', None)}
    for mode, spec in frozen['conditions'].items():
        names[mode] = spec['evaluation_run_ids']['te']
        cp = a.root / spec['checkpoint_output_relative'] / 'checkpoint-20'
        runs[mode] = load_eval(a.root, spec['queue'], names[mode], 'te', cp)
        pair(runs['Base'], runs[mode])
    language = read(a.language_report)
    assert language['target_label'] == '__label__tel_Telu'
    assert hashlib.sha256(Path(language['model']).read_bytes()).hexdigest() == language['model_sha256']
    stats = {}
    for role, name in names.items():
        _, metrics, rows, _ = runs[role]
        lang = language['runs'][name]
        items = a.root / 'logs/copsd_official_10lang/development_vllm8192' / name / 'ITEMS.jsonl'
        assert hashlib.sha256(items.read_bytes()).hexdigest() == lang['items_sha256']
        assert [r['id'] for r in rows] == [r['id'] for r in lang['predictions']]
        samples = [r['samples'][0] for r in rows]
        assert all(s['finish_reason'] in ['stop', 'length'] for s in samples)
        cells = Counter((bool(s['correct']), s['finish_reason'], s['boxed_answer'] is None) for s in samples)
        stats[role] = {'correct': metrics['single_correct'], 'normal_terminations': metrics['normal_terminations'],
                       'truncated': metrics['truncated'], 'answer_extraction_failures': metrics['answer_extraction_failures'],
                       'cells': [{'correct': k[0], 'finish_reason': k[1], 'answer_missing': k[2], 'count': n} for k, n in sorted(cells.items())],
                       'language': lang['summary'], 'items_sha256': lang['items_sha256']}
    transitions = Counter()
    for x, y in zip(runs['clip'][2], runs['unclip'][2]):
        c, u = x['samples'][0], y['samples'][0]
        if not c['correct'] and u['correct']:
            transitions[(c['finish_reason'], c['boxed_answer'] is None, u['finish_reason'])] += 1
    out = {'observed_at': datetime.datetime.now().isoformat(), 'protocol': frozen['protocol'],
           'target_items_each': 100, 'completion_and_language': stats,
           'unclip_gained_correct_transitions': [{'clip_finish': k[0], 'clip_answer_missing': k[1], 'unclip_finish': k[2], 'count': n} for k, n in sorted(transitions.items())],
           'paired_target': {mode: pair(runs['Base'], runs[mode]) for mode in ['clip', 'unclip']},
           'language_detector_model_sha256': language['model_sha256'],
           'source_sha256': {n: hashlib.sha256((code / n).read_bytes()).hexdigest() for n in ['analyze_output_languages.py', Path(__file__).name]},
           'new_training_or_inference': False,
           'scope': 'Descriptive CPU partitions of completed fixed development outputs. No score filtering, formal test or checkpoint selection. A GlotLID top1 language label is not fluency, per-token language purity or reasoning quality. Better termination and answer availability do not establish intrinsic mathematical gains; English retention is separately reported in the full paired collector.'}
    with a.output.open('x') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({k: {n: v[n] for n in ['correct', 'normal_terminations', 'truncated', 'answer_extraction_failures']} for k, v in stats.items()}))


if __name__ == '__main__': main()
