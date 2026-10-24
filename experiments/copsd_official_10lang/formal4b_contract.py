"""Frozen 4B formal-case preparation; no generation or language selection.

The case runner retains native LoRA, official prompts/scoring and n1/8192/c16.
Engineering contracts may inspect real inputs but cannot start GPU inference.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from evaluate_transformers import load_items

PROTOCOL = 'project_formal4b_n1_8192_native16_v1'
REVISION = '1cfa9a7208912126459214e8b04321603b3df60c'
LANGUAGES = set('sw te th bn zh ja ru de fr es'.split())


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 2**20), b''):
            result.update(block)
    return result.hexdigest()


def benchmark_input(root, benchmark, language):
    assert benchmark in ['mgsm_rev2', 'polymath', 'gsm8k']
    assert (benchmark == 'gsm8k') == (language == 'en')
    assert language in LANGUAGES | {'en'}
    if benchmark == 'mgsm_rev2':
        return root / f'data/math_single_10lang_20261015/mgsm_rev2/MGSM-Rev2/mgsm_{language}.tsv', 'mgsm_rev2', 250
    if benchmark == 'polymath':
        return root / f'data/copsd_official_20261015/polymath_project_eval_v1/polymath_all500_{language}.json', 'polymath_project', 500
    return root / 'data/copsd_official_20261015/gsm8k_project_en_zeroshot_v1/gsm8k_en_test_1319.json', 'polymath_project', 1319


def checkpoint(root, spec):
    relative = spec['adapter_relative']
    if relative is None:
        assert spec['expected_step'] is None and spec['adapter_sha256'] is None
        return None
    relative = Path(relative)
    assert not relative.is_absolute() and '..' not in relative.parts
    assert relative.parts[:2] == ('models', 'checkpoint')
    logical = root / relative
    actual = logical.resolve(strict=True)
    allowed = [(root / 'models/checkpoint').resolve(), Path('/data1') / root.name]
    assert any(actual.is_relative_to(parent) for parent in allowed)
    assert spec['expected_step'] == read(actual / 'trainer_state.json')['global_step'] == 20
    assert digest(actual / 'adapter_model.safetensors') == spec['adapter_sha256']
    peft = read(actual / 'adapter_config.json')
    assert Path(peft['base_model_name_or_path']).resolve() == (root / 'models/Qwen3-4B').resolve()
    assert peft['r'] == 64 and peft['lora_alpha'] == 128
    return logical


def prepare(root, freeze):
    code = Path(__file__).parent
    f = read(freeze)
    assert f['protocol'] == PROTOCOL and f['model_revision'] == REVISION
    assert f['samples'] == 1 and f['max_new_tokens'] == 8192 and f['max_num_seqs'] == 16
    assert f['seed'] == 42 and f['native_LoRA'] is True
    for name, expected in f['runtime_code_sha256'].items():
        assert digest(code / name) == expected, name
    if not f['engineering_preflight_only']:
        selection_path = root / f['selection_record_relative']
        assert digest(selection_path) == f['selection_record_sha256']
        selection = read(selection_path)
        assert selection['source_protocol'] == 'independent_development_only'
        assert selection['model_revision'] == REVISION
        assert len(selection['selected_languages']) == len(set(selection['selected_languages'])) == 3
        assert set(selection['selected_languages']) <= LANGUAGES
    source = root / 'COPSD-official-20261015/polymath_eval/evaluate_math.py'
    release = source.with_name('run_eval_4b_all_checkpoints.sh')
    assert re.search(r'(?m)^MAX_NEW_TOKENS=8192\s*$', release.read_text())
    lock = read(root / 'logs/copsd_official_10lang/eval_vllm_runtime_20261019.CONFIG.json')
    assert lock['official_commit'] == '4f4410730e9b6775a30d20f4b6e141cba2e87590'
    assert digest(source) == lock['source_sha256']['polymath_eval/evaluate_math.py']
    model_lock = root / f['model_lock_relative']
    assert digest(model_lock) == f['model_lock_sha256']
    model = read(model_lock)
    assert model['revision'] == REVISION
    for name, info in model['files'].items():
        assert digest(root / model['model_relative'] / name) == info['sha256'], name
    for name in ['evaluate_transformers.py', 'evaluate_vllm.py', 'compat/vllm011_uuid/sitecustomize.py']:
        assert digest(code / name) == lock['code_sha256'][name], name
    output_relative = Path(f['output_relative'])
    assert not output_relative.is_absolute() and '..' not in output_relative.parts
    assert output_relative.parts[:2] == ('logs', 'copsd_official_10lang')
    cases, inputs, adapters = {}, {}, {}
    run_ids = []
    for name, spec in f['cases'].items():
        assert re.fullmatch(r'[A-Za-z0-9_-]+', name)
        run_id = spec['run_id']
        assert re.fullmatch(r'[A-Za-z0-9_-]+', run_id)
        run_ids.append(run_id)
        benchmark, lang = spec['benchmark'], spec['language']
        data, kind, count = benchmark_input(root, benchmark, lang)
        assert digest(data) == lock['input_sha256'][str(data.relative_to(root))]
        if (benchmark, lang) not in inputs:
            items = list(load_items(data, kind, lang))
            assert len(items) == len({row['id'] for row in items}) == count
            difficulties = {row['id']: row['split'] for row in read(data)} if benchmark == 'polymath' else {}
            if difficulties:
                assert len(difficulties) == 500 and sorted(Counter(difficulties.values()).values()) == [125] * 4
            inputs[benchmark, lang] = (items, difficulties)
        items, difficulties = inputs[benchmark, lang]
        key = (spec['adapter_relative'], spec['adapter_sha256'], spec['expected_step'])
        if key not in adapters:
            adapters[key] = checkpoint(root, spec)
        adapter = adapters[key]
        output = root / output_relative / run_id
        config = {
            'protocol': PROTOCOL, 'benchmark': benchmark, 'language': lang,
            'case': name, 'cohort': 'independent_4B',
            'experiment_role': f['experiment_role'], 'frozen_plan_sha256': digest(freeze),
            'model': str(root / 'models/Qwen3-4B'), 'model_revision': REVISION,
            'adapter': str(adapter) if adapter else None, 'checkpoint_step': spec['expected_step'],
            'adapter_sha256': spec['adapter_sha256'],
            'input': str(data), 'input_sha256': digest(data), 'items': count, 'samples': 1,
            'max_new_tokens': 8192, 'max_num_seqs': 16, 'model_context_limit': 40960,
            'temperature': 1., 'top_p': .95, 'top_k': -1, 'seed': 42,
            'seed_policy': 'SHA256(seed:language:id:0), per request',
            'official_source_sha256': digest(source), 'released_length_script_sha256': digest(release),
            'evaluator_sha256': digest(code / 'evaluate_formal4b.py'),
            'scope': 'Full fixed4B native-LoRA n1/8192/c16 formal evaluation. Separate from independent development,1.7B and officialpass12; no language/checkpoint choice here.',
        }
        cases[name] = dict(spec=spec, config=config, data=data, source=source,
                           items=items, difficulties=difficulties, count=count,
                           output=output, adapter=adapter)
    assert cases and len(run_ids) == len(set(run_ids))
    return f, cases
