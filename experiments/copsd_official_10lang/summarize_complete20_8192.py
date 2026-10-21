"""Pair all ten target20 references with their actual English retention outputs."""
import argparse
import datetime
import json
from pathlib import Path

from disk_recovery_driver import audit_stage, digest
from summarize_base8192 import RUNS as BASE_RUNS
from summarize_targets20_8192 import TARGET_RUNS
from summarize_saved_candidates8192 import pair


def load(root, code, run_id, language, adapter, step):
    return audit_stage({'directory': str(root / 'logs/copsd_official_10lang/development_vllm8192' / run_id),
        'language': language, 'adapter': adapter, 'step': step,
        'protocol': 'project_dev_single_vllm8192_v2', 'budget': 8192,
        'evaluator': 'evaluate_dev8192.py'}, root, code)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    code = Path(__file__).parent
    folder = args.root / 'logs/copsd_official_10lang/development_vllm8192'
    English_by_adapter = {}
    for path in folder.glob('*/CONFIG.json'):
        config = json.loads(path.read_text())
        if config.get('language') != 'en' or not config.get('adapter'):
            continue
        if config.get('protocol') != 'project_dev_single_vllm8192_v2' or not (path.parent / 'DONE.json').exists():
            continue
        English_by_adapter.setdefault(str(Path(config['adapter']).resolve()), []).append(path.parent.name)
    base_en = load(args.root, code, BASE_RUNS['en'], 'en', None, None)
    results = {}
    for language, target_run in TARGET_RUNS.items():
        target_config = json.loads((folder / target_run / 'CONFIG.json').read_text())
        checkpoint = Path(target_config['adapter'])
        assert json.loads((checkpoint / 'trainer_state.json').read_text())['global_step'] == 20
        assert (checkpoint / 'adapter_model.safetensors').stat().st_size > 0
        candidates = English_by_adapter[str(checkpoint.resolve())]
        assert len(candidates) == 1, (language, candidates)  # No best-score selection.
        English_run = candidates[0]
        English_config = json.loads((folder / English_run / 'CONFIG.json').read_text())
        assert Path(English_config['adapter']).resolve() == checkpoint.resolve()
        target = load(args.root, code, target_run, language, str(checkpoint), 20)
        base = load(args.root, code, BASE_RUNS[language], language, None, None)
        English = load(args.root, code, English_run, 'en', str(checkpoint), English_config.get('checkpoint_step'))
        for group in zip(base[2], target[2], base_en[2], English[2]):
            assert len({row['id'] for row in group}) == 1
            assert len({str(row['answer']) for row in group}) == 1
        results[language] = {'target': pair(base, target), 'English': pair(base_en, English),
            'target_run_id': target_run, 'English_run_id': English_run,
            'English_items_sha256': digest(folder / English_run / 'ITEMS.jsonl'),
            'checkpoint_global_step_actually_checked': 20}
    report = {'observed_at': datetime.datetime.now().isoformat(),
        'protocol': 'project_dev_single_vllm8192_v2', 'results': results,
        'all10_target_and_English_complete100_item_pairs_verified': True,
        'English_comparison_uses_same_actual_target20_checkpoint': True,
        'English_output_selection': 'Unique completed same-adapter match; reject ambiguity, never choose by accuracy.',
        'new_generation': False, 'new_training': False, 'GPU_used': False,
        'scope': 'Complete paired independent-development inventory only. Parent success and scientific training caveats remain documented in individual reports. Historical4GPU/initializer differences and JA literal-prefix repair remain; not an exact paper-reproduction ranking, stable final language selection, formal results or sequential evidence. Uses official-clip reference runs; no-clip interventions and extra seeds stay separate. All outputs/costs overlap earlier reports; no cost sum or additional compute.'}
    with args.output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({lang: {'target': x['target'], 'English': x['English']} for lang, x in results.items()}))


if __name__ == '__main__':
    main()
