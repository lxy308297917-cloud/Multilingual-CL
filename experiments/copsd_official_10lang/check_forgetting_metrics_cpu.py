"""Hand-calculated synthetic matrix checks; never scientific model results."""
import argparse
import copy
import datetime
import hashlib
import json
from pathlib import Path

from forgetting_metrics import calculate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    order = ['L1', 'L2', 'L3']
    values = [[20, 30, 40, 80], [50, 31, 39, 79], [55, 60, 42, 78], [52, 65, 70, 77]]
    names = ['Base', 'after_L1', 'after_L2', 'after_L3']
    matrix = {'order': order, 'scores': {stage: {lang: {'correct': n, 'total': 100}
        for lang, n in zip(order + ['en'], row)} for stage, row in zip(names, values)}}
    result = calculate(matrix)
    assert len(result['accuracy_matrix_percent']) == 4
    assert result['accuracy_matrix_percent']['Base']['L1'] == 20
    assert result['input_counts'] == matrix['scores']
    stages = result['stage_metrics']
    assert stages[1]['old_languages']['L1']['forgetting_pp'] == -5
    assert stages[2]['old_languages']['L1']['forgetting_pp'] == 3
    assert stages[2]['old_languages']['L1']['decline_from_own_learned_stage_pp'] == -2
    assert stages[2]['old_languages']['L2']['forgetting_pp'] == -5
    assert result['final_mean_old_language_forgetting_pp'] == -1
    assert abs(result['final_three_target_macro_accuracy_percent'] - 187 / 3) < 1e-12
    assert result['final_English_change_from_Base_pp'] == -3
    assert [s['current_language_gain_from_previous_stage_pp'] for s in stages] == [30, 29, 28]
    assert not stages[0]['old_languages'] and stages[0]['mean_old_language_forgetting_pp'] is None
    assert set(stages[-1]['old_languages']) == {'L1', 'L2'}
    rejected = []
    for mode in ['missing_language', 'missing_stage', 'missing_count', 'changed_denominator']:
        broken = copy.deepcopy(matrix)
        if mode == 'missing_language': del broken['scores']['after_L2']['L1']
        elif mode == 'missing_stage': del broken['scores']['after_L3']
        elif mode == 'missing_count': broken['scores']['after_L3']['L3'] = None
        else: broken['scores']['after_L3']['L1']['total'] = 99
        try: calculate(broken)
        except ValueError: rejected.append(mode)
        else: raise AssertionError(mode)
    report = {'observed_at': datetime.datetime.now().isoformat(),
        'scope': 'Synthetic CPU arithmetic verification only, not actual languages, models, sequential training, evaluation scores or final candidate selection.',
        'checks': {'negative_forgetting_preserved': True, 'current_stage_excluded_from_historical_max': True,
            'learned_stage_decline_distinguished_from_historical_forgetting': True,
            'English_excluded_from_target_macro_average': True,
            'new_language_not_assigned_zero_forgetting': True, 'missing_scores_not_filled_with_zero': True},
        'invalid_matrices_rejected': rejected,
        'code_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                        for name in ['forgetting_metrics.py', 'check_forgetting_metrics_cpu.py']},
        'GPU_used': False, 'new_training_or_generation': False}
    with a.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report['checks']))


if __name__ == '__main__':
    main()
