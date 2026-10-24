"""Analyze the seven additional Base language references after formal acceptance.

One manually launched CPU job; never starts inference or retries failed work.
The selected-language analysis already covers SW/TE/ZH and English.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

from analyze_sequential4b_languages import LABELS
from formal4b_contract import digest, read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, code = args.root.resolve(), Path(__file__).parent.resolve()
    logs = root / 'logs/copsd_official_10lang'
    assert not args.output.exists()
    state = dict(pid=os.getpid(), GPU_started=False, automatic_retries=False,
                 started_at=datetime.datetime.now().isoformat(), completed_languages=[])

    def record(phase, **values):
        state.update(phase=phase, updated_at=datetime.datetime.now().isoformat(), **values)
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(args.output)
        print(json.dumps(state), flush=True)

    try:
        marker_path = logs / 'primary4b_TEN_LANGUAGE_FORMAL_MATH_COMPLETED_20261024.PRIVATE.json'
        record('waiting_primary_math_acceptance')
        while not marker_path.exists():
            parent = read(logs / 'primary4b_formal_acceptance_20261024.PRIVATE.json')
            assert parent['phase'] != 'failed', parent
            assert Path('/proc', str(parent['pid'])).exists(), parent
            time.sleep(30)
        marker = read(marker_path)
        assert len(marker['cases']) == 10
        model = root / 'models/glotlid/model_v3.bin'
        model_sha = digest(model)
        reports = {}
        record('analyzing_additional_Base_languages')
        for language in ['th', 'bn', 'ja', 'ru', 'de', 'fr', 'es']:
            directory = logs / f'primary4b_Base_{language.upper()}_FORMAL_LANGUAGES_20261024'
            directory.mkdir(exist_ok=False)
            cases, inputs = {}, []
            for benchmark in ['mgsm_rev2', 'polymath']:
                name = benchmark + '_' + language
                reference = marker['cases'][language][name]
                pair_path = root / reference['pair_relative']
                assert digest(pair_path) == reference['pair_sha256']
                pair = read(pair_path)
                assert pair['case'] == name and pair['Base']['config']['adapter'] is None
                case = pair['Base']
                folder = root / case['output_relative']
                for filename, expected in case['files_sha256'].items():
                    assert digest(folder / filename) == expected
                assert read(folder / 'DONE.json') == reference['Base'] == case['metrics']
                cases[name] = case
                inputs.append(folder / 'ITEMS.jsonl')
            output = directory / 'LANGUAGES.PRIVATE.json'
            command = [str(root / 'envs/copsd-official-20261015/bin/python'),
                       str(code / 'analyze_output_languages.py'), '--items',
                       *map(str, inputs), '--model', str(model), '--target-label',
                       '__label__' + LABELS[language], '--output', str(output)]
            with (directory / 'analysis.log').open('x') as stream:
                subprocess.run(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                               stdout=stream, stderr=subprocess.STDOUT, check=True)
            analysis = read(output)
            assert analysis['model_sha256'] == model_sha
            assert analysis['target_label'] == '__label__' + LABELS[language]
            for name, path in zip(cases, inputs):
                run = analysis['runs'][path.parent.name]
                assert run['items_sha256'] == cases[name]['files_sha256']['ITEMS.jsonl']
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                predictions = run['predictions']
                assert len(rows) == len(predictions) == cases[name]['metrics']['items']
                assert all((row['id'], row['samples'][0]['correct']) ==
                           (prediction['id'], prediction['correct'])
                           for row, prediction in zip(rows, predictions))
                def target(prediction, part):
                    value = prediction[part]
                    return value.get('label') == analysis['target_label'] and value.get('prob', 0) >= .8
                reports[name] = dict(summary=run['summary'],
                    correct_and_target_thinking=sum(p['correct'] and target(p, 'thinking') for p in predictions),
                    correct_and_both_parts_target=sum(p['correct'] and target(p, 'thinking') and target(p, 'final') for p in predictions),
                    analysis_relative=str(output.relative_to(root)), analysis_sha256=digest(output),
                    items_sha256=run['items_sha256'])
            state['completed_languages'].append(language)
            record('analyzing_additional_Base_languages')
        assert len(reports) == 14
        receipt = dict(cases=reports, language_model_sha256=model_sha,
                       formal_math_marker_sha256=digest(marker_path),
                       source_sha256=digest(Path(__file__)), GPU_started=False,
                       scope='Seven additional Base languages, two full formal benchmarks each. SW/TE/ZH/English remain in selected-language analysis. Whole-part language ID does not measure fluency, token purity or proof validity. No inference or score modification.')
        with (logs / 'primary4b_Base_ADDITIONAL14_LANGUAGES_COMPLETED_20261024.PRIVATE.json').open('x') as stream:
            json.dump(receipt, stream, indent=2)
            stream.write('\n')
        record('completed_additional_Base_languages')
    except BaseException as error:
        record('failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
