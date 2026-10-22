"""Summarize fixed completed followup language diagnostics and TensorBoard exports."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    code = Path(__file__).parent
    full_path = code / 'followup200_all_completed_20261022.PUBLIC.json'
    full = read(full_path)
    assert full['all_nine_parents_completion_verified'] and full['all_requested_parents_completed_exit0']
    assert full['analysis_source_sha256'] == sha(code / 'collect_followup200.py')
    freeze = code / 'followup200_frozen_20261021.PUBLIC.json'
    frozen = read(freeze)
    assert full['data_freeze_sha256'] == sha(freeze)
    language = {}
    for lang in ['sw', 'ja', 'zh', 'bn', 'en']:
        record = read(a.root / 'logs/copsd_official_10lang/followup200_language_20261022' / (lang + '.PRIVATE.json'))
        assert record['model_sha256'] == sha(Path(record['model']))
        language[lang] = {'target_label': record['target_label'], 'cases': {}}
        for case, sources in full['source_records'].items():
            if lang not in sources: continue
            source = sources[lang]
            result = record['runs'][source['run_id']]
            assert result['items_sha256'] == source['items_sha256']
            assert [r['id'] for r in result['predictions']] == frozen['selected_IDs']
            assert sum(r['correct'] for r in result['predictions']) == full['metrics'][case][lang]['single_correct']
            language[lang]['cases'][case] = {'language': result['summary'], 'completion': full['metrics'][case][lang], 'source': source}
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    tb = {}
    for case, sources in full['source_records'].items():
        directory = a.root / 'logs/copsd_official_10lang/tensorboard/followup200_20261021_v1' / case
        accumulator = EventAccumulator(str(directory)); accumulator.Reload()
        step = 0 if case == 'base' else 20
        for lang, source in sources.items():
            marker = read(directory / ('EXPORTED_' + lang + '.json'))
            assert marker['source'] == source and marker['data_freeze_sha256'] == sha(freeze)
            event = accumulator.Scalars('eval/' + lang + '/accuracy')[-1]
            assert event.step == step and abs(event.value - full['metrics'][case][lang]['single_accuracy']) < 1e-6
        cost = accumulator.Scalars('cost/unique_parent_GPU_hours')[-1]
        assert cost.step == step and abs(cost.value - full['parent_costs'][case]['parent_GPU_hours_counted_once']) < 1e-6
        tb[case] = {'relative_path': str(directory.relative_to(a.root)), 'accuracy_series_read_back': len(sources), 'step': step, 'parent_cost_read_back': True}
    assert sum(len(x['cases']) for x in language.values()) == 21
    out = {'observed_at': datetime.datetime.now().isoformat(), 'protocol': 'project_followup200_language_completion_supplement_v1',
        'math_report_sha256': sha(full_path), 'data_freeze_sha256': sha(freeze),
        'language_and_completion': language, 'TensorBoard_verified': tb,
        'all21_case_accuracy_series_read_back': True, 'all9_parent_cost_series_read_back': True,
        'detector_source_sha256': sha(code / 'analyze_output_languages.py'),
        'analysis_source_sha256': sha(Path(__file__)), 'new_training_or_GPU_inference': False,
        'final_three_selected': False,
        'scope': 'Existing frozen outputs only. GlotLID top1 after math stripping is not fluency, per-token purity or mathematical proof validation. Model-specific thinking/final denominators remain200; absent final sections are not removed. Fixed9 parents already counted in the math report; no extra GPU cost or alternative regrading. This supplement does not override failed math confirmation or independently authorize a three-language research selection.'}
    with a.output.open('x') as f: json.dump(out, f, ensure_ascii=False, indent=2); f.write('\n')
    print(json.dumps({'cases': 21, 'TensorBoard_parents': len(tb), 'language_report': str(a.output)}))


if __name__ == '__main__': main()
