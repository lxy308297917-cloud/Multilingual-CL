"""Audit a completed fixed followup model instance; not a two-model selection report."""
import argparse
from collections import Counter
import datetime
import json
from pathlib import Path
from collect_followup200 import audit_case, paired, read, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--case', required=True)
    p.add_argument('--language-report', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    code = Path(__file__).parent
    freeze = code / 'followup200_frozen_20261021.PUBLIC.json'
    frozen = read(freeze)
    for name, expected in frozen['runtime_code_sha256'].items():
        assert sha(code / name) == expected, name
    assert a.case != 'base'
    base, base_cost = audit_case(a.root, code, frozen, sha(freeze), 'base')
    model, model_cost = audit_case(a.root, code, frozen, sha(freeze), a.case)
    target = next(lang for lang in model if lang != 'en')
    comparisons = {lang: paired(base[lang], model[lang]) for lang in model}
    transitions = {}
    for lang in model:
        for kind in ['gained', 'lost']:
            cells = Counter()
            for x, y in zip(base[lang]['rows'], model[lang]['rows']):
                b, m = x['samples'][0], y['samples'][0]
                changed = (not b['correct'] and m['correct']) if kind == 'gained' else (b['correct'] and not m['correct'])
                if changed:
                    cells[(b['finish_reason'], b['boxed_answer'] is None, m['finish_reason'], m['boxed_answer'] is None)] += 1
            transitions[lang + '_' + kind] = [dict(Base_finish=k[0], Base_answer_missing=k[1], model_finish=k[2], model_answer_missing=k[3], count=n) for k, n in sorted(cells.items())]
    out = dict(observed_at=datetime.datetime.now().isoformat(), protocol=frozen['protocol'],
        case=a.case, data_freeze_sha256=sha(freeze), both_parents_completed_exit0=True,
        paired= comparisons, completion_transitions=transitions,
        metrics={role: {lang: run[lang]['metrics'] for lang in model} for role, run in [('Base', base), ('model', model)]},
        source_records={role: {lang: run[lang]['source'] for lang in model} for role, run in [('base', base), (a.case, model)]},
        cost=dict(new_model_parent=model_cost, shared_Base_parent=base_cost,
            scope='Base is shared with other cases and already reported; do not sum overlapping staged/full reports.'),
        final_three_selected=False, analysis_source_sha256=sha(Path(__file__)),
        scope='One fixed model instance on the prespecified development200. No two-model bootstrap, final language selection, new training or inference. Completion improvement is not proof of intrinsic mathematical gains.')
    if a.language_report:
        language = read(a.language_report)
        assert language['target_label'] == {'sw': '__label__swh_Latn', 'ja': '__label__jpn_Jpan', 'zh': '__label__cmn_Hani', 'bn': '__label__ben_Beng'}[target]
        assert sha(Path(language['model'])) == language['model_sha256']
        summaries = {}
        for role, run in [('base', base), (a.case, model)]:
            record = run[target]['source']
            lid = language['runs'][record['run_id']]
            assert lid['items_sha256'] == record['items_sha256']
            assert [r['id'] for r in lid['predictions']] == [r['id'] for r in run[target]['rows']]
            summaries[role] = lid['summary']
        out['language'] = dict(target_label=language['target_label'], model_sha256=language['model_sha256'], summaries=summaries,
            scope='GlotLID top1 after math stripping; not fluency, calibrated purity or reasoning quality.')
    with a.output.open('x') as f:
        json.dump(out, f, ensure_ascii=False, indent=2); f.write('\n')
    print(json.dumps(comparisons))


if __name__ == '__main__': main()
