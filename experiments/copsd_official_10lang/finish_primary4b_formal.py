"""One-shot CPU acceptance and language analysis of all ten primary single20s.

Runs only after the missing formal queue jobs finish. Reuses audited old SW20
native16 results, keeping them distinct from the new matched900 SW reference.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from analyze_sequential4b_languages import LABELS
from collect_sequential4b_formal_pair import collect
from formal4b_contract import digest, read


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    root, code = a.root.resolve(), Path(__file__).parent.resolve()
    logs = root / 'logs/copsd_official_10lang'
    assert not a.output.exists()
    state = dict(pid=os.getpid(), started_at=datetime.datetime.now().isoformat(),
                 automatic_retries=False, GPU_started=False, completed_pairs=[])

    def record(phase, **kwargs):
        state.update(phase=phase, updated_at=datetime.datetime.now().isoformat(), **kwargs)
        temporary = a.output.with_name(a.output.name + '.tmp')
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(a.output)
        print(json.dumps({'phase': phase, **kwargs}), flush=True)

    def run(name, *args):
        subprocess.run([sys.executable, str(code / name), *map(str, args)],
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), check=True)

    try:
        sw_audit_path = logs / 'primary4b_SW20_cache_audit_20261024.PRIVATE.json'
        sw_audit = read(sw_audit_path)
        assert sw_audit['primary_SW20_only'] and sw_audit['generation_and_scoring_suffix_AST_identical']
        assert sw_audit['all_prompts_inputs_seeds_token_decodes_and_accounting_verified']
        record('waiting_primary_formal_queue_jobs')
        while True:
            preceding = read(logs / 'primary4b_formal_wave_20261024.PRIVATE.json')
            if preceding['phase'] == 'completed_formal_queue_jobs':
                break
            assert preceding['phase'] != 'failed', preceding
            assert Path('/proc', str(preceding['pid'])).exists(), preceding
            time.sleep(30)
        base_freeze = code / 'sequential4b_Base_formal_frozen_20261024_v1.LOCAL.json'
        base_additional = code / 'primary4b_Base_additional14_formal_frozen_20261024_v1.LOCAL.json'
        base_map = read(logs / 'sequential4b_Base_formal_queues_20261024.PRIVATE.json')
        additional_map = read(logs / 'primary4b_Base_additional14_formal_queues_20261024.PRIVATE.json')
        legacy_base = read(logs / 'sequential4b_Base_cache_audit_20261024.PRIVATE.json')
        results, events, folders = {}, {}, {}
        record('accepting_all_primary_formal_pairs')
        for language in ['sw', 'te', 'th', 'bn', 'zh', 'ja', 'ru', 'de', 'fr', 'es']:
            upper = language.upper()
            freeze = code / f'primary4b_{upper}20_formal_frozen_20261024_v1.LOCAL.json'
            f = read(freeze)
            queues = read(logs / f'primary4b_{upper}20_formal_queues_20261024.PRIVATE.json')
            assert f['model_id'] == 'Primary_' + upper + '20'
            results[language], folders[language] = {}, {}
            for benchmark in ['mgsm_rev2', 'polymath', 'gsm8k']:
                lang = 'en' if benchmark == 'gsm8k' else language
                name = benchmark + '_' + lang
                if language == 'sw' and benchmark != 'polymath':
                    cached = sw_audit['cases'][name]
                    base = legacy_base['cases'][name]
                    for item in [cached, base]:
                        folder = root / item['output_relative']
                        for filename, expected in item['file_sha256'].items():
                            assert digest(folder / filename) == expected
                    assert cached['semantic_config']['adapter_sha256'] == next(iter(f['cases'].values()))['adapter_sha256']
                    model_metrics = read(root / cached['output_relative'] / 'DONE.json')
                    base_metrics = read(root / base['output_relative'] / 'DONE.json')
                    # Both reuse receipts verified the same fixed input prompts,
                    # IDs, per-question seeds, tokenizer and generation suffix.
                    for key in ['input_sha256', 'model_revision', 'samples', 'max_new_tokens',
                                'max_num_seqs', 'temperature', 'top_p', 'top_k', 'seed', 'packages']:
                        assert cached['semantic_config'][key] == base['semantic_config'][key]
                    results[language][name] = dict(Base=base_metrics, model=model_metrics,
                                                  reused_primary_SW20_native16=True,
                                                  cache_audit_sha256=digest(sw_audit_path))
                    folders[language][name] = root / cached['output_relative']
                    continue
                bf = base_freeze if language in ['sw', 'te', 'zh'] or lang == 'en' else base_additional
                bq = (base_map if bf == base_freeze else additional_map).get(name)
                output = logs / f'primary4b_{upper}20_{name}_PAIR_20261024.PRIVATE.json'
                pair = collect(root, bf, freeze, name, bq, queues[name], output)
                run('record_sequential4b_evaluation_tensorboard.py', '--root', root,
                    '--pair', output,
                    '--directory', logs / 'tensorboard/primary4b_formal_pairs' / (upper + '_' + name),
                    '--receipt', logs / f'primary4b_{upper}20_{name}_TB_20261024.PRIVATE.json')
                for condition, queue in [('Base', bq), ('model', queues[name])]:
                    cost = pair[condition]['cost']
                    if not cost['reused']:
                        assert queue
                        event = dict(GPU_hours=cost['new_GPU_hours'], generated_tokens=cost['new_generated_tokens'])
                        assert queue not in events or events[queue] == event
                        events[queue] = event
                results[language][name] = dict(Base=pair['Base']['metrics'], model=pair['model']['metrics'],
                                              pair_relative=str(output.relative_to(root)), pair_sha256=digest(output))
                folders[language][name] = root / pair['model']['output_relative']
                state['completed_pairs'].append(upper + ':' + name)
                record('accepting_all_primary_formal_pairs')
        assert len(results) == 10 and all(len(cases) == 3 for cases in results.values())
        math_report = {'protocol': 'project_formal4b_n1_8192_native16_v1',
                       'cohort': 'Original primary ten independent4B single20s; not matched900 references.',
                       'cases': results, 'unique_new_evaluation_cost_events': events,
                       'unique_new_evaluation_GPU_hours': sum(x['GPU_hours'] for x in events.values()),
                       'unique_new_generated_tokens': sum(x['generated_tokens'] for x in events.values()),
                       'scope': 'All30primarymodel cases accepted,includingtwoauditedlegacySW20native16cases. SharedBasecostcountedonce; reusedhistoricalBase/SWcostszeroNEWcost. Training/development/selectedsequencecostsremainseparate. Noformaltestselectionorpass12claim.'}
        with (logs / 'primary4b_TEN_LANGUAGE_FORMAL_MATH_COMPLETED_20261024.PRIVATE.json').open('x') as stream:
            json.dump(math_report, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        record('analyzing_primary_formal_languages')
        for language, cases in folders.items():
            directory = logs / f'primary4b_{language.upper()}20_FORMAL_LANGUAGES_20261024'
            directory.mkdir(exist_ok=False)
            groups = [(language, ['mgsm_rev2_' + language, 'polymath_' + language]),
                      ('en', ['gsm8k_en'])]
            if language == 'sw':
                # These two unchanged outputs already have accepted language
                # predictions; preserve their original provenance and analyze
                # only the missing native16 PolyMath output.
                previous_path = code / 'sw4b_formal_language_completed_20261022.PUBLIC.json'
                previous = read(previous_path)
                assert previous['TensorBoard_twelve_language_series_read_back']
                assert previous['accepted_formal_report_sha256'] == sw_audit['prior_report_sha256']
                reused = {}
                for benchmark, lang in [('mgsm_rev2', 'sw'), ('gsm8k', 'en')]:
                    source = logs / 'sw4b_formal_language_20261022' / f'SW20_{benchmark}.PRIVATE.json'
                    expected = previous['cases']['SW20_' + benchmark]
                    assert digest(source) == expected['analysis_sha256']
                    assert digest(cases[benchmark + '_' + lang] / 'ITEMS.jsonl') == expected['items_sha256']
                    analysis = read(source)
                    assert analysis['model_sha256'] == digest(root / 'models/glotlid/model_v3.bin')
                    assert analysis['target_label'] == '__label__' + LABELS[lang]
                    reused[benchmark + '_' + lang] = dict(source_relative=str(source.relative_to(root)),
                                                         source_sha256=digest(source), original_summary=expected)
                with (directory / 'REUSED_LANGUAGE_ANALYSES.PRIVATE.json').open('x') as stream:
                    json.dump(reused, stream, ensure_ascii=False, indent=2)
                    stream.write('\n')
                groups = [('sw', ['polymath_sw'])]
            for lang, names in groups:
                command = [str(root / 'envs/copsd-official-20261015/bin/python'),
                           str(code / 'analyze_output_languages.py'), '--items',
                           *[str(cases[name] / 'ITEMS.jsonl') for name in names],
                           '--model', str(root / 'models/glotlid/model_v3.bin'),
                           '--target-label', '__label__' + LABELS[lang],
                           '--output', str(directory / (lang + '.PRIVATE.json'))]
                with (directory / (lang + '.log')).open('x') as log:
                    subprocess.run(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
        record('completed_primary_math_and_language_reports',
               remaining='Consolidate training/failure/recovery costs and publish final source-bound scientific results; this is not goal completion.')
    except BaseException as error:
        record('failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
