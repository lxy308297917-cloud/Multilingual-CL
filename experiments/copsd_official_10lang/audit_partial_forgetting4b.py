"""Audit a saved old-language pair without claiming a complete stage matrix.

Supports three-stage histories: the learned stage and the immediately preceding
stage contain all earlier post-learning stages. Original scores stay unchanged.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from formal4b_contract import digest, read


def load_pair(root, path):
    pair = read(path)
    assert pair['items'] == 250 and not pair['proves_complete_seven_case_stage']
    case = pair['model']
    folder = root / case['output_relative']
    for name, expected in case['files_sha256'].items():
        assert digest(folder / name) == expected
    config, done = read(folder / 'CONFIG.json'), read(folder / 'DONE.json')
    assert config == case['config'] and done == case['metrics']
    assert config['protocol'] == 'project_formal4b_n1_8192_native16_v1'
    assert done['items'] == 250 and done['n'] == 1 and done['formal_full_test']
    rows = [json.loads(line) for line in (folder / 'ITEMS.jsonl').read_text().splitlines()]
    assert len(rows) == 250 and len({row['id'] for row in rows}) == 250
    assert all(len(row['samples']) == 1 and type(row['samples'][0]['correct']) is bool for row in rows)
    assert sum(row['samples'][0]['correct'] for row in rows) == done['single_correct']
    return pair, rows


def compare(before, after):
    for a, b in zip(before, after):
        assert all(a[key] == b[key] for key in ['id', 'question', 'answer', 'prompt'])
        assert a['samples'][0]['seed'] == b['samples'][0]['seed']
    old = np.array([row['samples'][0]['correct'] for row in before], dtype=np.int8)
    new = np.array([row['samples'][0]['correct'] for row in after], dtype=np.int8)
    change = new - old
    bootstrap = change[np.random.default_rng(20261024).integers(0, 250, (10000, 250))].mean(axis=1) * 100
    interval = np.quantile(bootstrap, [.025, .975]).tolist()
    lost = [b['samples'][0] for a, b in zip(before, after)
            if a['samples'][0]['correct'] and not b['samples'][0]['correct']]
    return dict(previous_correct=int(old.sum()), current_correct=int(new.sum()),
                gained=int((change == 1).sum()), lost=len(lost),
                change_pp=float(change.mean() * 100), change_question_CI95_pp=interval,
                signed_decline_pp=float(-change.mean() * 100),
                signed_decline_question_CI95_pp=[-interval[1], -interval[0]],
                lost_current_truncated=sum(s['finish_reason'] == 'length' for s in lost),
                lost_current_no_extracted_box=sum(s['boxed_answer'] is None for s in lost),
                lost_current_normal_with_box=sum(s['finish_reason'] == 'stop' and s['boxed_answer'] is not None for s in lost))


def language(path, pair, rows, target_label, model_sha):
    value = read(path)
    assert value['target_label'] == target_label and value['model_sha256'] == model_sha
    expected = pair['model']['files_sha256']['ITEMS.jsonl']
    runs = [run for run in value['runs'].values() if run['items_sha256'] == expected]
    assert len(runs) == 1 and digest(Path(runs[0]['items'])) == expected
    predictions = runs[0]['predictions']
    assert len(predictions) == len(rows)
    for row, prediction in zip(rows, predictions):
        assert (row['id'], row['samples'][0]['correct']) == (prediction['id'], prediction['correct'])
    def target(prediction, part):
        return prediction[part].get('label') == target_label and prediction[part].get('prob', 0) >= .8
    return dict(thinking_target_highconf=sum(target(p, 'thinking') for p in predictions),
                thinking_English_top1=sum(p['thinking'].get('label') == '__label__eng_Latn' for p in predictions),
                correct_and_target_thinking_highconf=sum(p['correct'] and target(p, 'thinking') for p in predictions),
                correct_and_both_parts_target_highconf=sum(p['correct'] and target(p, 'thinking') and target(p, 'final') for p in predictions))


def audit(args):
    root = args.root.resolve()
    assert not args.output.exists()
    current, rows = load_pair(root, args.current_pair)
    previous, prior_rows = load_pair(root, args.previous_pair)
    learned, learned_rows = load_pair(root, args.learned_pair)
    freeze = read(args.current_formal_freeze)
    assert digest(args.current_formal_freeze) == current['model_freeze_sha256']
    provenance = freeze['model_provenance']
    release_path = root / provenance['release_relative']
    assert digest(release_path) == provenance['release_sha256']
    release = read(release_path)
    context = release['stage_context']
    evaluation_language = current['model']['config']['language']
    assert context['stage_index'] in [2, 3] and context['language'] != evaluation_language
    order = ['sw', 'te', 'zh'] if context['direction'] == 'forward' else ['zh', 'te', 'sw']
    assert context['direction'] in ['forward', 'reverse']
    assert evaluation_language in order[:context['stage_index'] - 1]
    extra_paths = []
    for old_pair, old_freeze, expected_index in [
            (learned, args.learned_formal_freeze, order.index(evaluation_language) + 1),
            (previous, args.previous_formal_freeze, context['stage_index'] - 1)]:
        assert digest(old_freeze) == old_pair['model_freeze_sha256']
        old_provenance = read(old_freeze)['model_provenance']
        old_release_path = root / old_provenance['release_relative']
        assert digest(old_release_path) == old_provenance['release_sha256']
        old_release = read(old_release_path)
        old_context = old_release['stage_context']
        assert old_context['language'] == order[expected_index - 1]
        assert old_context['stage_index'] == expected_index
        assert old_release['adapter_sha256'] == old_pair['model']['config']['adapter_sha256']
        extra_paths.extend([old_freeze, old_release_path])
        if expected_index > 1:
            assert old_context['direction'] == context['direction']
    for pair in [current, previous, learned]:
        assert pair['case'] == current['case']
        assert pair['selection_record_sha256'] == current['selection_record_sha256']
        for key in ['model_revision', 'input_sha256', 'max_new_tokens', 'max_num_seqs',
                    'temperature', 'top_p', 'top_k', 'seed', 'seed_policy',
                    'official_source_sha256', 'packages']:
            assert pair['model']['config'][key] == current['model']['config'][key]
    train_path = root / release['freeze_relative']
    assert digest(train_path) == release['freeze_sha256']
    train = read(train_path)
    assert train['parent_adapter_sha256'] == previous['model']['config']['adapter_sha256']
    parent = root / train['parent_checkpoint_relative']
    assert parent.resolve() == Path(previous['model']['config']['adapter']).resolve()
    assert digest(parent / 'adapter_model.safetensors') == train['parent_adapter_sha256']
    assert release['adapter_sha256'] == current['model']['config']['adapter_sha256']
    learned_comparison = compare(learned_rows, rows)
    previous_comparison = compare(prior_rows, rows)
    # At most two earlier post-learning stages exist in a three-stage order.
    use_previous = previous['model_correct'] > learned['model_correct']
    best = previous_comparison if use_previous else learned_comparison
    detector_sha = digest(root / 'models/glotlid/model_v3.bin')
    pairs = [learned, previous, current]
    all_rows = [learned_rows, prior_rows, rows]
    lang_paths = [args.learned_language, args.previous_language, args.current_language]
    languages = {name: language(path, pair, data, args.target_label, detector_sha)
                 for name, path, pair, data in zip(['learned', 'previous', 'current'], lang_paths, pairs, all_rows)}
    paths = [args.current_pair, args.previous_pair, args.learned_pair, *lang_paths,
             args.current_formal_freeze, release_path, train_path, *extra_paths]
    result = dict(model_id=current['model_id'], case=current['case'], items=250,
                  learned_stage=learned_comparison, previous_stage=previous_comparison,
                  historical_best_model_id=(previous if use_previous else learned)['model_id'],
                  historical_best_forgetting_pp=best['signed_decline_pp'],
                  historical_best_forgetting_question_CI95_pp=best['signed_decline_question_CI95_pp'],
                  language=languages, language_detector_sha256=detector_sha,
                  metrics={name: pair['model']['metrics'] for name, pair in zip(['learned', 'previous', 'current'], pairs)},
                  Base_comparison={key: current[key] for key in ['Base_correct', 'model_correct', 'gained', 'lost', 'delta_percentage_points', 'paired_question_bootstrap95_CI_pp']},
                  actual_parent_adapter_verified=True, common_question_answer_prompt_seed_verified=True,
                  source_sha256={str(path): digest(path) for path in paths},
                  collector_source_sha256=digest(Path(__file__)),
                  proves_complete_seven_case_stage=False, proves_complete_forgetting_matrix=False,
                  scope='Original-score partial old-language history. Best earlier stage selected by fixed observed score; bootstrap is conditional on those fixed models, not retraining or reselecting the best stage. Before-learning stages and Base excluded from forgetting history. Negative forgetting retained. Truncation/no-box counts overlap. Whole-part language labels do not establish fluency or proof quality.')
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'current-pair', 'previous-pair', 'learned-pair',
                 'current-language', 'previous-language', 'learned-language',
                 'current-formal-freeze', 'previous-formal-freeze', 'learned-formal-freeze', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--target-label', required=True)
    value = audit(p.parse_args())
    print(json.dumps({key: value[key] for key in ['model_id', 'case', 'historical_best_forgetting_pp',
                                                 'historical_best_forgetting_question_CI95_pp', 'language']}))


if __name__ == '__main__':
    main()
