"""Recover73 serialized TE Base items; generate only the missing27.

The original frozen evaluator still builds prompts, samples, grades and writes
all100. This wrapper supplies verified cached outputs to its generation call.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import evaluate_dev8192 as original
from evaluate_vllm import official_api, question_seed


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(root, run_id):
    private = json.loads(Path(__file__).with_name('disk_recovery_runtime_20261020.CONFIG.json').read_text())
    spec = private['operations']['te']
    assert run_id == spec['run_ids']['base']
    old = Path(spec['partial_Base_directory'])
    for name, sha in spec['partial_Base_sha256'].items():
        assert digest(old / name) == sha
    config = json.loads((old / 'CONFIG.json').read_text())
    assert config['protocol'] == 'project_dev_single_vllm8192_v2' and config['language'] == 'te'
    assert config['adapter'] is None and config['checkpoint_step'] is None
    assert config['max_new_tokens'] == 8192 and config['samples'] == 1 and config['max_num_seqs'] == 16
    assert config['evaluator_sha256'] == digest(Path(original.__file__))
    assert config['model'] == str(root / 'models/Qwen3-1.7B')
    assert config['input_sha256'] == digest(Path(config['input']))
    assert config['official_source_sha256'] == digest(Path(config['source']))
    assert all(importlib.metadata.version(name) == value for name, value in config['packages'].items())
    lines = (old / 'ITEMS.jsonl').read_bytes().splitlines()
    rows = []
    for i, line in enumerate(lines):
        try: row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            assert i == len(lines) - 1
            break
        rows.append(row)
    assert len(rows) == spec['serialized_complete_items'] == 73
    data = json.loads(Path(config['input']).read_text())
    assert len(data) == 100
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model'], trust_remote_code=True, local_files_only=True)
    api = official_api(Path(config['source']))
    cached = []
    for row, item in zip(rows, data):
        assert row['id'] == item['problem_id'] and row['question'] == item['problem_te'] and row['answer'] == item['Answer']
        assert len(row['samples']) == 1
        sample = row['samples'][0]
        ids = sample['token_ids']
        assert sample['tokens'] == len(ids) <= 8192
        assert sample['seed'] == question_seed(42, 'te', row['id'])
        assert tokenizer.decode(ids, skip_special_tokens=False) == sample['raw_output']
        text = tokenizer.decode(ids, skip_special_tokens=True)
        assert api['extract_boxed_answer'](text) == sample['boxed_answer']
        assert bool(api['grade_answer'](sample['boxed_answer'], row['answer'])) == sample['correct']
        assert api['build_qwen3_prompt'](tokenizer, row['question'], 'TE', True) == row['prompt']
        cached.append(SimpleNamespace(prompt=row['prompt'], outputs=[SimpleNamespace(
            token_ids=ids, text=text, finish_reason=sample['finish_reason'])]))
    return rows, cached, config, old


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args()
    rows, cached, config, old = prepare(a.root, a.run_id)
    sys.argv = [str(Path(original.__file__)), '--root', str(a.root), '--language', 'te', '--run-id', a.run_id]
    if a.prepare_only:
        sys.argv.append('--prepare-only')
        original.main()
        print(json.dumps({'cached_verified_items': len(rows), 'missing_items': 100 - len(rows), 'GPU_initialized': False}))
        return
    import vllm
    actual_llm = vllm.LLM
    observed = {}

    def recovered_llm(**kwargs):
        engine = actual_llm(**kwargs)
        generate = engine.generate

        def recovered_generate(prompts, params, **generation_kwargs):
            assert len(prompts) == len(params) == 100
            for i, row in enumerate(rows):
                assert prompts[i] == row['prompt'] and params[i].seed == row['samples'][0]['seed']
            assert all(x.n == 1 and x.max_tokens == 8192 and x.temperature == 1. and x.top_p == .95 and x.top_k == -1 for x in params)
            started = time.monotonic()
            outputs = generate(prompts[len(rows):], params[len(rows):], **generation_kwargs)
            assert len(outputs) == 100 - len(rows)
            observed.update(new_generated_items=len(outputs), new_generated_tokens=sum(len(x.outputs[0].token_ids) for x in outputs),
                            new_generation_seconds=time.monotonic() - started)
            return cached + outputs

        engine.generate = recovered_generate
        return engine

    vllm.LLM = recovered_llm
    try: original.main()
    finally: vllm.LLM = actual_llm
    output = a.root / 'logs/copsd_official_10lang/development_vllm8192' / a.run_id
    merged = [json.loads(x) for x in (output / 'ITEMS.jsonl').read_text().splitlines()]
    assert len(merged) == 100 and merged[:len(rows)] == rows
    provenance = {**observed, 'reused_items': len(rows),
        'reused_generated_tokens': sum(x['samples'][0]['tokens'] for x in rows),
        'old_partial_items_sha256': digest(old / 'ITEMS.jsonl'),
        'old_config_sha256': digest(old / 'CONFIG.json'),
        'wrapper_sha256': digest(Path(__file__)), 'original_evaluator_sha256': digest(Path(original.__file__)),
        'cached_records_preserved_exactly': True,
        'scope': 'DONE generated_tokens counts all100 scored responses, not new compute. Cost must use new_generated_items/tokens/seconds here. Only missing27 requests generated; same per-ID seeds/config, no promise of bitwise equivalence to a fresh all100 continuous batch. Original interrupted outputs remain intact.'}
    original.write_json(output / 'RECOVERY.PROVENANCE.json', provenance)
    print(json.dumps(provenance))


if __name__ == '__main__':
    main()
