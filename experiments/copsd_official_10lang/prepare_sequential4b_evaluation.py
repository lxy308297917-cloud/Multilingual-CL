"""Freeze seven formal cases for Base or one accepted selected 4B model."""
import argparse
import json
from pathlib import Path
import re

from formal4b_contract import PROTOCOL, REVISION, prepare, digest, read
from sequential4b_selection import local, selection, required_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--release', type=Path)
    parser.add_argument('--base-cache-audit', type=Path)
    parser.add_argument('--output-relative', required=True)
    parser.add_argument('--freeze', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    code = Path(__file__).parent
    assert re.fullmatch(r'[A-Za-z0-9_-]+', args.model_id)
    assert not args.freeze.exists()
    selection_sha = digest(args.selection)
    selected = selection(root, args.selection, selection_sha)
    if args.release:
        accepted = read(args.release)
        assert accepted['engineering_only'] is False and accepted['global_step'] == 20
        assert accepted['selection_record_sha256'] == selection_sha
        assert accepted['checkpoint20_optimizer_scheduler_RNG_verified']
        assert accepted['original_Base_bitwise_unchanged']
        adapter_relative, adapter_sha = accepted['checkpoint_relative'], accepted['adapter_sha256']
        assert digest(local(root, adapter_relative) / 'adapter_model.safetensors') == adapter_sha
        role = accepted['stage_context']['experiment_role']
        provenance = {'release_relative': str(args.release.relative_to(root)),
                      'release_sha256': digest(args.release),
                      'stage_context': accepted['stage_context']}
    else:
        assert args.model_id == 'Base'
        adapter_relative = adapter_sha = None
        role = 'selected_languages_original_Base'
        provenance = {'original_Base': True}
    prepared = read(code / 'sw4b_resource2_prepared_20261022.PUBLIC.json')
    names = ['formal4b_contract.py', 'evaluate_formal4b.py', 'run_formal4b_case.sh',
             'prepare_sequential4b_evaluation.py', 'sequential4b_selection.py',
             'evaluate_transformers.py', 'evaluate_vllm.py',
             'benchmark_eval_concurrency8192.py', 'compat/vllm011_uuid/sitecustomize.py']
    frozen = {'protocol': PROTOCOL, 'model_revision': REVISION,
              'samples': 1, 'max_new_tokens': 8192, 'max_num_seqs': 16,
              'seed': 42, 'native_LoRA': True, 'engineering_preflight_only': False,
              'experiment_role': role, 'model_id': args.model_id,
              'selection_record_relative': str(args.selection.relative_to(root)),
              'selection_record_sha256': selection_sha,
              'model_lock_relative': prepared['model_lock_relative'],
              'model_lock_sha256': prepared['model_lock_sha256'],
              'output_relative': args.output_relative, 'model_provenance': provenance,
              'runtime_code_sha256': {name: digest(code / name) for name in names},
              'max_account_GPUs': 4, 'GPU_per_case': 1,
              'cases': {benchmark + '_' + language: {
                  'benchmark': benchmark, 'language': language,
                  'run_id': args.model_id + '_' + benchmark + '_' + language,
                  'adapter_relative': adapter_relative, 'adapter_sha256': adapter_sha,
                  'expected_step': 20 if adapter_relative else None}
                  for benchmark, language in required_cases(selected['selected_languages'])}}
    if args.base_cache_audit:
        assert args.release is None, 'Base cache cannot stand in for a trained model'
        audit = read(args.base_cache_audit)
        assert audit['model_revision'] == REVISION and audit['original_Base_only']
        assert audit['generation_and_scoring_suffix_AST_identical']
        assert audit['all_prompts_inputs_seeds_token_decodes_and_accounting_verified']
        for name, expected in audit['source_sha256'].items():
            assert digest(code / name) == expected, name
        cached = {}
        for name, case in audit['cases'].items():
            if name not in frozen['cases']:
                continue
            folder = local(root, case['output_relative'])
            for file_name, expected in case['file_sha256'].items():
                assert digest(folder / file_name) == expected
            cached[name] = case
        frozen['cached_cases'] = cached
        frozen['Base_cache_audit_relative'] = str(args.base_cache_audit.relative_to(root))
        frozen['Base_cache_audit_sha256'] = digest(args.base_cache_audit)
        frozen['cache_accounting'] = 'Reuse old outputs with their original protocol/provenance; do not rerun or count old generation cost as new computation.'
    with args.freeze.open('x') as stream:
        json.dump(frozen, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    _, cases = prepare(root, args.freeze)
    assert len(cases) == 7 and all(not case['output'].exists() for case in cases.values())
    print(json.dumps({'CPU_preflight': True, 'GPU_started': False,
                      'model_id': args.model_id, 'cases': list(cases)}))


if __name__ == '__main__':
    main()
