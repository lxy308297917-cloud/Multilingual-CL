"""Audit existing native16 primary SW20 outputs for ten-language reporting.

No model inference or language selection. Never relabel a native32 output.
"""
import argparse
import ast
from collections import Counter
import importlib.metadata
import json
from pathlib import Path

from formal4b_contract import REVISION, benchmark_input, digest, read
from evaluate_transformers import load_items
from evaluate_vllm import official_api, question_seed


def generation_suffix(path):
    tree = ast.parse(path.read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    index = next(i for i, node in enumerate(main.body)
                 if isinstance(node, ast.ImportFrom) and node.module == 'vllm')
    return ast.dump(ast.Module(body=main.body[index:], type_ignores=[]))


def audit(root, output):
    root, output = Path(root), Path(output)
    assert not output.exists()
    code = Path(__file__).parent
    old = code / 'evaluate_sw4b_formal8192.py'
    new = code / 'evaluate_formal4b.py'
    assert generation_suffix(old) == generation_suffix(new)
    frozen = read(code / 'sw4b_formal_stage_frozen_20261022.PUBLIC.json')
    assert digest(old) == frozen['runtime_code_sha256'][old.name]
    source = root / 'COPSD-official-20261015/polymath_eval/evaluate_math.py'
    released = source.with_name('run_eval_4b_all_checkpoints.sh')
    api = official_api(source)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(root / 'models/Qwen3-4B', local_files_only=True)
    packages = {name: importlib.metadata.version(name) for name in ['torch', 'transformers', 'vllm', 'math-verify']}
    spec = frozen['cases']['SW20']
    adapter = root / spec['adapter_relative']
    assert digest(adapter / 'adapter_model.safetensors') == spec['adapter_sha256']
    assert read(adapter / 'trainer_state.json')['global_step'] == 20
    prior_report_path = code / 'sw4b_formal8192_completed_20261022.PUBLIC.json'
    prior_report = read(prior_report_path)
    assert prior_report['model_revision'] == REVISION and prior_report['TensorBoard_four_accuracies_two_costs_read_back']
    accepted = {}
    for benchmark, language in [('mgsm_rev2', 'sw'), ('gsm8k', 'en')]:
        run_id = frozen['cases']['SW20']['run_ids'][benchmark]
        folder = root / 'logs/copsd_official_10lang/evaluation_vllm8192_4b_v1' / run_id
        config, done = read(folder / 'CONFIG.json'), read(folder / 'DONE.json')
        data, kind, count = benchmark_input(root, benchmark, language)
        expected = {'model_revision': REVISION, 'adapter': str(adapter), 'checkpoint_step': 20,
                    'adapter_sha256': spec['adapter_sha256'], 'benchmark': benchmark, 'language': language,
                    'input': str(data), 'input_sha256': digest(data), 'items': count,
                    'model': str(root / 'models/Qwen3-4B'), 'samples': 1,
                    'max_new_tokens': 8192, 'max_num_seqs': 16, 'model_context_limit': 40960,
                    'temperature': 1., 'top_p': .95, 'top_k': -1, 'seed': 42,
                    'seed_policy': 'SHA256(seed:language:id:0), per request',
                    'official_source_sha256': digest(source),
                    'released_length_script_sha256': digest(released),
                    'evaluator_sha256': digest(old), 'packages': packages}
        assert all(config[key] == value for key, value in expected.items())
        assert config['case'] == 'SW20' and config['cohort'] == 'independent_4B'
        assert config['frozen_stage_sha256'] == digest(code / 'sw4b_formal_stage_frozen_20261022.PUBLIC.json')
        assert done['formal_full_test'] and done['n'] == 1 and done['actual_max_num_seqs'] == 16
        assert done['items'] == done['generation_count'] == count
        assert read(folder / 'PHASE.json')['phase'] == 'done'
        prior = prior_report['cases'][benchmark]['SW20']
        assert prior['metrics'] == done
        assert prior['ITEMS_sha256'] == digest(folder / 'ITEMS.jsonl')
        assert prior['CONFIG_sha256'] == digest(folder / 'CONFIG.json')
        items = list(load_items(data, kind, language))
        rows = [json.loads(line) for line in (folder / 'ITEMS.jsonl').read_text().splitlines()]
        assert len(items) == len(rows) == count
        totals = Counter()
        for item, row in zip(items, rows):
            assert all(row[key] == value for key, value in item.items())
            assert row['prompt'] == api['build_qwen3_prompt'](tokenizer, item['question'], language.upper(), True)
            assert len(row['samples']) == 1
            sample = row['samples'][0]
            assert sample['seed'] == question_seed(42, language, item['id'])
            assert type(sample['correct']) is bool
            assert sample['tokens'] == len(sample['token_ids']) <= 8192
            assert tokenizer.decode(sample['token_ids'], skip_special_tokens=False) == sample['raw_output']
            assert api['extract_boxed_answer'](sample['raw_output']) == sample['boxed_answer']
            totals['single_correct'] += sample['correct']
            totals['generated_tokens'] += sample['tokens']
            totals['truncated'] += sample['finish_reason'] == 'length'
            totals['normal_terminations'] += sample['finish_reason'] == 'stop'
            totals['answer_extraction_failures'] += sample['boxed_answer'] is None
        assert all(done[key] == value for key, value in totals.items())
        assert done['single_accuracy'] == totals['single_correct'] / count
        accepted[benchmark + '_' + language] = {
            'output_relative': str(folder.relative_to(root)), 'items': count,
            'single_correct': done['single_correct'], 'generated_tokens': done['generated_tokens'],
            'file_sha256': {name: digest(folder / name) for name in ['CONFIG.json', 'DONE.json', 'ITEMS.jsonl', 'PHASE.json']},
            'semantic_config': expected}
    receipt = {'model_revision': REVISION, 'original_Base_only': False, 'primary_SW20_only': True,
               'adapter_sha256': spec['adapter_sha256'],
               'prior_report_sha256': digest(prior_report_path),
               'generation_and_scoring_suffix_AST_identical': True,
               'all_prompts_inputs_seeds_token_decodes_and_accounting_verified': True,
               'source_sha256': {old.name: digest(old), new.name: digest(new),
                                 'audit_formal4b_primary_SW_cache.py': digest(Path(__file__))},
               'cases': accepted, 'GPU_started': False,
               'scope': 'Primary SW20 native16 cache reuse only, with exact old provenance retained; distinct from matched900SW20 and native32PolyMath. No formal-test result is used to select languages; no new model output or rescored accuracy is created.'}
    with output.open('x') as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root, args.output)
    print(json.dumps({'GPU_started': False, 'cases': {
        key: {field: case[field] for field in ['items', 'single_correct', 'generated_tokens']}
        for key, case in result['cases'].items()}}))


if __name__ == '__main__':
    main()
