"""Inventory actual ten-language20-update target development comparisons.

Per-language paired outputs only; unfinished English/parents remain unfinished.
"""
import argparse
import datetime
import json
from pathlib import Path

from disk_recovery_driver import audit_stage, digest
from summarize_base8192 import RUNS as BASE_RUNS
from summarize_saved_candidates8192 import pair

TARGET_RUNS = {
    'sw': 'sw_clip20_dev100_8192_20261020_v1',
    'te': 'te_saved20_screen_target20_recovered_dev100_8192_20261020_v2',
    'th': 'th_saved20_screen_target20_dev100_8192_20261020_v1',
    'bn': 'bn_17b_clip20_horizon100_seed42_20261020_v1_dev8192',
    'zh': 'zh_early20_candidate_dev100_8192_20261020_v1',
    'ja': 'ja_v4_saved20_target_dev100_8192_20261020_v1',
    'ru': 'ru_saved20_screen_target20_dev100_8192_20261020_v1',
    'de': 'de_saved20_screen_target20_dev100_8192_20261020_v1',
    'fr': 'fr_early20_candidate_dev100_8192_20261020_v1',
    'es': 'es_saved20_screen_target20_dev100_8192_20261020_v1',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    code = Path(__file__).parent
    results = {}
    common_ids = common_answers = None
    for language, target_id in TARGET_RUNS.items():
        stages = []
        for run_id, step in [(BASE_RUNS[language], None), (target_id, 20)]:
            directory = args.root / 'logs/copsd_official_10lang/development_vllm8192' / run_id
            config = json.loads((directory / 'CONFIG.json').read_text())
            adapter = config['adapter']
            if step is None:
                assert adapter is None
            else:
                checkpoint = Path(adapter)
                assert checkpoint.resolve().is_relative_to((args.root / 'models/checkpoint').resolve())
                assert json.loads((checkpoint / 'trainer_state.json').read_text())['global_step'] == step
                assert (checkpoint / 'adapter_model.safetensors').stat().st_size > 0
            stages.append(audit_stage({'directory': str(directory), 'language': language,
                'adapter': adapter, 'step': step, 'protocol': 'project_dev_single_vllm8192_v2',
                'budget': 8192, 'evaluator': 'evaluate_dev8192.py'}, args.root, code))
        paired = pair(*stages)
        ids = [row['id'] for row in stages[0][2]]
        answers = [str(row['answer']) for row in stages[0][2]]
        if common_ids is None:
            common_ids, common_answers = ids, answers
        assert ids == common_ids and answers == common_answers
        results[language] = {'paired': paired, 'base_run_id': BASE_RUNS[language],
            'target20_run_id': target_id,
            'target20_items_sha256': digest(args.root / 'logs/copsd_official_10lang/development_vllm8192' / target_id / 'ITEMS.jsonl'),
            'metrics': {name: {k: stage[1][k] for k in ['items', 'single_correct',
                'truncated', 'normal_terminations', 'answer_extraction_failures']}
                for name, stage in zip(['base', 'target20'], stages)}}
    report = {'observed_at': datetime.datetime.now().isoformat(),
        'protocol': 'project_dev_single_vllm8192_v2', 'results': results,
        'all10_actual_complete_target_pairs_verified': True,
        'same100_original_ID_order_and_gold_across_languages': True,
        'positive_single_reference_deltas_not_final_selection': [lang for lang, value in results.items()
            if value['paired']['delta_correct'] > 0],
        'English_TE_JA_and_parent_completion_not_claimed': True,
        'new_generation': False, 'new_training': False, 'GPU_used': False,
        'scope': 'Independent development100 target-side screening inventory only, not formal scores, stable language selection or sequential results. Original historical4GPU/initializer and JA repair caveats remain. BN no-clip intervention and additional seeds are separate experiments, not substituted into this official-clip reference inventory. All outputs overlap previous run reports; no additional compute or cost sum.'}
    with args.output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({lang: value['paired'] for lang, value in results.items()}))


if __name__ == '__main__':
    main()
