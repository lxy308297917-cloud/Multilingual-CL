"""Audit the completed ten-language plus English development Base inventory.

No generation, checkpoint selection, new scores or cost aggregation.
"""
import argparse
import datetime
import json
from pathlib import Path

from disk_recovery_driver import audit_stage, digest


RUNS = {
    'sw': 'sw_base_dev100_8192_20261020_v1',
    'te': 'te_saved20_screen_base_recovered_dev100_8192_20261020_v2',
    'th': 'th_saved20_screen_base_dev100_8192_20261020_v1',
    'bn': 'bn_base_dev100_8192_20261020_v1',
    'zh': 'zh_base_candidate_dev100_8192_20261020_v1',
    'ja': 'ja_v4_saved20_base_dev100_8192_20261020_v1',
    'ru': 'ru_saved20_screen_base_dev100_8192_20261020_v1',
    'de': 'de_saved20_screen_base_dev100_8192_20261020_v1',
    'fr': 'fr_base_candidate_dev100_8192_20261020_v1',
    'es': 'es_saved20_screen_base_dev100_8192_20261020_v1',
    'en': 'en_sw_base_dev100_8192_20261020_v1',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    code = Path(__file__).parent
    report = {'observed_at': datetime.datetime.now().isoformat(),
              'protocol': 'project_dev_single_vllm8192_v2', 'results': {}}
    common_ids = common_answers = None
    common_packages = None
    for language, run_id in RUNS.items():
        directory = args.root / 'logs/copsd_official_10lang/development_vllm8192' / run_id
        config, metrics, rows, _ = audit_stage({
            'directory': str(directory), 'language': language, 'adapter': None,
            'step': None, 'protocol': report['protocol'], 'budget': 8192,
            'evaluator': 'evaluate_dev8192.py'}, args.root, code)
        assert config['temperature'] == 1. and config['top_p'] == .95 and config['top_k'] == -1
        ids = [row['id'] for row in rows]
        answers = [str(row['answer']) for row in rows]
        if common_ids is None:
            common_ids, common_answers, common_packages = ids, answers, config['packages']
        assert ids == common_ids and answers == common_answers
        assert config['packages'] == common_packages
        assert metrics['normal_terminations'] + metrics['truncated'] == 100
        value = {k: metrics[k] for k in ['items', 'single_correct', 'single_accuracy',
                    'generated_tokens', 'truncated', 'normal_terminations', 'answer_extraction_failures']}
        value.update(run_id=run_id, input_sha256=config['input_sha256'],
                     output_items_sha256=digest(directory / 'ITEMS.jsonl'))
        if language == 'te':
            provenance = json.loads((directory / 'RECOVERY.PROVENANCE.json').read_text())
            assert provenance['cached_records_preserved_exactly']
            assert provenance['reused_items'] == 73 and provenance['new_generated_items'] == 27
            value['recovery_reused73_new27'] = True
        report['results'][language] = value
    report.update(target_languages=10, English_retention_languages=1,
                  same100_original_ID_order_and_gold_across_all11=True,
                  packages=common_packages, no_new_generation=True, no_GPU_used=True,
                  scope='Existing independent development100 Base outputs only, not formal tests or trained-model efficacy. Cached stages can be complete while their parent pipelines remain running/failed; this inventory does not claim those parents succeeded. Output token counts describe scored answers, not additional compute. No cost sum; all outputs overlap existing run reports, TE includes73 cached responses.')
    with args.output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({lang: r['single_correct'] for lang, r in report['results'].items()}))


if __name__ == '__main__':
    main()
