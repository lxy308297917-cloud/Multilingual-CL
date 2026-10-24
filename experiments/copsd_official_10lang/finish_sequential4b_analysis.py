"""One-shot CPU analysis after the complete selected-language formal wave.

Creates language reports, both full forgetting matrices, and six comparisons to
the matched single references. Does not generate model answers or change scores.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    root, code = a.root.resolve(), Path(__file__).parent.resolve()
    logs = root / 'logs/copsd_official_10lang'
    assert not a.output.exists()
    state = dict(pid=os.getpid(), started_at=datetime.datetime.now().isoformat(),
                 automatic_retries=False, GPU_started=False, completed_analyses=[])

    def read(path):
        return json.loads(path.read_text())

    def record(phase, **kwargs):
        state.update(phase=phase, updated_at=datetime.datetime.now().isoformat(), **kwargs)
        temporary = a.output.with_name(a.output.name + '.tmp')
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(a.output)
        print(json.dumps({'phase': phase, **kwargs}), flush=True)

    def run(name, *args):
        subprocess.run([sys.executable, str(code / name), *map(str, args)],
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), check=True)

    def marker(model):
        return logs / f'selected4b_{model}_FORMAL_COMPLETED_20261024.PRIVATE.json'

    try:
        record('waiting_complete_seven_case_evaluations')
        while True:
            preceding = read(logs / 'sequential4b_formal_wave_20261024.PRIVATE.json')
            if preceding['phase'] == 'completed_evaluation_and_TensorBoard':
                break
            assert preceding['phase'] != 'failed', preceding
            assert Path('/proc', str(preceding['pid'])).exists(), preceding
            time.sleep(30)
        selected = code / 'sequential4b_selection_frozen_20261024_v1.LOCAL.json'
        models = ['Base', 'SWmatched', 'TEmatched', 'ZHmatched',
                  'forward_s2', 'reverse_s2', 'forward_s3', 'reverse_s3']
        record('analyzing_languages')
        for model in models:
            assert read(marker(model))['all_seven_cases_accepted']
            run('analyze_sequential4b_languages.py', '--root', root,
                '--evaluation', marker(model),
                '--output-dir', logs / f'selected4b_{model}_FORMAL_LANGUAGES_20261024')
            state['completed_analyses'].append(model + ':languages')
            record('analyzing_languages')
        record('building_forgetting_and_matched_comparisons')
        for direction, order in [('forward', ['sw', 'te', 'zh']),
                                  ('reverse', ['zh', 'te', 'sw'])]:
            names = [order[0].upper() + 'matched', direction + '_s2', direction + '_s3']
            run('build_sequential4b_matrix.py', '--root', root, '--selection', selected,
                '--direction', direction, '--markers', marker('Base'),
                *[marker(model) for model in names],
                '--output', logs / f'selected4b_{direction}_FORGETTING_20261024.PRIVATE.json')
            state['completed_analyses'].append(direction + ':forgetting')
            for index, (language, model) in enumerate(zip(order, names), 1):
                stage_exposures = (logs / f'selected4b_matched_{language}20_EXPOSURES_20261024.PRIVATE.json'
                                   if index == 1 else logs / f'selected4b_{direction}_s{index}_EXPOSURES_20261024.PRIVATE.json')
                run('compare_sequential4b_matched_single.py', '--root', root,
                    '--selection', selected, '--direction', direction, '--stage-index', index,
                    '--stage-marker', marker(model),
                    '--single-marker', marker(language.upper() + 'matched'),
                    '--stage-exposures', stage_exposures,
                    '--single-exposures', logs / f'selected4b_matched_{language}20_EXPOSURES_20261024.PRIVATE.json',
                    '--output', logs / f'selected4b_{direction}_s{index}_MATCHED_COMPARISON_20261024.PRIVATE.json')
                state['completed_analyses'].append(direction + ':matched_stage' + str(index))
                record('building_forgetting_and_matched_comparisons')
        record('completed_selected_language_analysis',
               remaining='Primary ten-language full formal audits, unique cost consolidation and final scientific report remain required.')
    except BaseException as error:
        record('failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
