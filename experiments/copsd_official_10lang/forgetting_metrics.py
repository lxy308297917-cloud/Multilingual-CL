"""Arithmetic for a complete three-stage accuracy matrix; not a provenance audit."""
import argparse
from fractions import Fraction
import json
from pathlib import Path


def calculate(data):
    order = data['order']
    if len(order) != 3 or len(set(order)) != 3 or 'en' in order:
        raise ValueError('Three distinct training languages required; English is retention only')
    stages = ['Base'] + ['after_' + lang for lang in order]
    languages = order + ['en']
    scores = data['scores']
    if set(scores) != set(stages):
        raise ValueError('Base and all three completed stages required')
    rates, denominators = [], {}
    for stage in stages:
        if set(scores[stage]) != set(languages):
            raise ValueError('Every stage must include all three target languages and English')
        row = {}
        for lang in languages:
            value = scores[stage][lang]
            if not isinstance(value, dict) or set(value) != {'correct', 'total'}:
                raise ValueError('Actual correct/total counts required; missing scores are not zero')
            correct, total = value['correct'], value['total']
            if type(correct) is not int or type(total) is not int or not 0 <= correct <= total or total <= 0:
                raise ValueError('Invalid integer counts')
            if lang in denominators and denominators[lang] != total:
                raise ValueError('Each evaluation language needs the same denominator at all stages')
            denominators[lang] = total
            row[lang] = Fraction(correct, total)
        rates.append(row)
    percent = lambda x: float(100 * x)
    result = []
    for k in range(1, 4):
        current = order[k - 1]
        old = {}
        for i, lang in enumerate(order[:k - 1], start=1):
            previous_best = max(rates[j][lang] for j in range(i, k))
            old[lang] = {
                'decline_from_own_learned_stage_pp': percent(rates[i][lang] - rates[k][lang]),
                'historical_best_before_current_percent': percent(previous_best),
                'forgetting_pp': percent(previous_best - rates[k][lang]),
            }
        result.append({
            'stage': stages[k], 'training_language': current,
            'accuracy_percent': {lang: percent(rates[k][lang]) for lang in languages},
            'current_language_gain_from_previous_stage_pp': percent(rates[k][current] - rates[k - 1][current]),
            'current_language_gain_from_Base_pp': percent(rates[k][current] - rates[0][current]),
            'old_languages': old,
            'mean_old_language_forgetting_pp': sum(v['forgetting_pp'] for v in old.values()) / len(old) if old else None,
            'English_change_from_Base_pp': percent(rates[k]['en'] - rates[0]['en']),
        })
    return {
        'order': order, 'evaluation_denominators': denominators,
        'input_counts': scores,
        'accuracy_matrix_percent': {stage: {lang: percent(row[lang]) for lang in languages}
                                    for stage, row in zip(stages, rates)},
        'stage_metrics': result,
        'final_three_target_macro_accuracy_percent': percent(sum(rates[3][lang] for lang in order) / 3),
        'final_English_change_from_Base_pp': result[-1]['English_change_from_Base_pp'],
        'final_mean_old_language_forgetting_pp': result[-1]['mean_old_language_forgetting_pp'],
        'definition': 'For old language i at stage k: max accuracy over completed stages i..k-1 minus current accuracy. Base and current stage are excluded from that history; negative values are retained. Last newly learned language is not assigned artificial zero forgetting. English is excluded from training-stage and target-language averages.',
        'scope': 'Arithmetic from supplied counts only. Caller must independently audit model/checkpoint identities, common inputs/prompts/sampling/scoring, completed evaluations and disjoint training pools. No experiments or language selection are performed here.',
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matrix', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    result = calculate(json.loads(a.matrix.read_text()))
    with a.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
