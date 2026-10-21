"""Build complete three-language forgetting matrices from seven-suite markers.

Refuses missing stages, changed protocols, mismatched checkpoints, prompts or
training pools. Does not choose languages, generate answers or train models.
"""
import argparse
import datetime
import json
from pathlib import Path

from forgetting_metrics import calculate
from sequential_evaluation import (PRESETS, audit_benchmark, digest, read,
                                   selection, verify_plan, write_new)


def audit_marker(root, code, path, selection_sha):
    marker = read(path)
    assert marker['completed'] and marker['protocol'] in PRESETS
    assert marker['selection_record_sha256'] == selection_sha
    assert marker['evaluation_parent_exit_code'] == 0
    plan_path = Path(marker['plan_path_PRIVATE'])
    assert digest(plan_path) == marker['plan_sha256']
    plan = read(plan_path)
    assert plan.get('experiment_role', 'sequential_stage') == 'sequential_stage', 'Matched singles are references, not sequential stages'
    assert marker.get('experiment_role', 'sequential_stage') == 'sequential_stage'
    assert marker['protocol'] == plan['protocol']
    _, checkpoint = verify_plan(root, code, plan)
    assert plan['selection_record_sha256'] == selection_sha
    assert marker['checkpoint_adapter_sha256'] == plan['checkpoint_adapter_sha256']
    assert marker['order_id'] == plan['order_id'] and marker['stage_index'] == plan['stage_index']
    observed = [audit_benchmark(root, code, case, checkpoint) for case in plan['cases']]
    assert observed == marker['completed_benchmarks']
    return marker, plan, observed


def build(root, code, selection_path, selection_sha, direction, markers):
    s = selection(selection_path, selection_sha)
    assert len(markers) == 4 and len({Path(p).resolve() for p in markers}) == 4
    order = s['selected_languages'] if direction == 'forward' else list(reversed(s['selected_languages']))
    order_id = s['experiment_id'] + '_' + direction
    audits = [audit_marker(root, code, p, selection_sha) for p in markers]
    assert [a[1]['stage_index'] for a in audits] == [0, 1, 2, 3]
    # The same frozen Base evaluation may be reused for both directions.
    assert all(plan['direction'] == direction and plan['order_id'] == order_id for _, plan, _ in audits[1:])
    baseline = {(r['benchmark'], r['language']): r for r in audits[0][2]}
    scores = {benchmark: {} for benchmark in ['mgsm_rev2', 'polymath']}
    training_sources, actual_pool_ids = [], []
    stage_names = ['Base'] + ['after_' + lang for lang in order]
    for index, ((marker, plan, results), label) in enumerate(zip(audits, stage_names)):
        by_case = {(r['benchmark'], r['language']): r for r in results}
        for key, result in by_case.items():
            first = baseline[key]
            assert result['canonical_config'] == first['canonical_config']
            assert result['input_sha256'] == first['input_sha256']
            folder = root / 'logs/copsd_official_10lang/evaluation_vllm8192'
            base_rows = [json.loads(x) for x in (folder / first['run_id'] / 'ITEMS.jsonl').read_text().splitlines()]
            now_rows = [json.loads(x) for x in (folder / result['run_id'] / 'ITEMS.jsonl').read_text().splitlines()]
            assert len(base_rows) == len(now_rows)
            for b, n in zip(base_rows, now_rows):
                assert all(b[k] == n[k] for k in ['id', 'question', 'answer', 'prompt'])
                assert b['samples'][0]['seed'] == n['samples'][0]['seed']
        English = by_case[('gsm8k', 'en')]
        for benchmark in scores:
            row = {lang: {'correct': by_case[(benchmark, lang)]['correct'],
                          'total': by_case[(benchmark, lang)]['items']} for lang in order}
            row['en'] = {'correct': English['correct'], 'total': English['items']}
            scores[benchmark][label] = row
        if index == 0:
            continue
        cp = Path(plan['checkpoint'])
        state = read(root / 'gpu-queue/queues' / plan['training_queue'] / 'state.json')
        spec_path = Path(state['job']['config']['job']['args'][0])
        spec = read(spec_path)
        context = read(cp.parent / 'STAGE_CONTEXT.PRIVATE.json')['context']
        previous_cp = audits[index - 1][1]['checkpoint']
        if previous_cp is None:
            assert spec.get('parent_checkpoint') is None
        else:
            assert Path(spec['parent_checkpoint']).resolve() == Path(previous_cp).resolve()
        assert digest(spec['training_data']) == context['training_data_sha256']
        pool = read(spec['pool_manifest'])
        assert digest(spec['pool_manifest']) == context['pool_manifest_sha256']
        ids = [r['problem_id'] for r in read(spec['training_data'])]
        assert ids == pool['pool_ids'][s['pool_by_language'][order[index - 1]]]
        assert len(ids) == len(set(ids)) and not set(ids) & set(pool['development_ids'])
        actual_pool_ids.append(set(ids))
        proof = read(cp.parent / 'COLLOCATE_TRAIN_UPDATES_DONE.json')
        progress = read(cp / 'COST_SEGMENT_PROGRESS.json')
        assert progress['global_step'] == proof['global_step'] == 20
        training_sources.append({'stage_index': index, 'language': order[index - 1],
            'run_id': cp.parent.name, 'checkpoint_adapter_sha256': digest(cp / 'adapter_model.safetensors'),
            'training_spec_sha256': digest(spec_path), 'training_data_sha256': context['training_data_sha256'],
            'pool_manifest_sha256': context['pool_manifest_sha256'], 'pool_rows': len(ids),
            'final_segment_start_step': proof['start_step'],
            'final_segment_cost_at_accepted_checkpoint': progress,
            'cost_scope': 'Final process segment at accepted checkpoint only. A resumed stage requires its earlier checkpoint-segment ledgers and failed work to be accounted separately; no total training cost claim here.'})
    assert all(not actual_pool_ids[i] & actual_pool_ids[j] for i in range(3) for j in range(i + 1, 3))
    assert len({x['pool_manifest_sha256'] for x in training_sources}) == 1
    return {'observed_at': datetime.datetime.now().isoformat(), 'order_id': order_id,
        'selection_record_sha256': selection_sha, 'protocol': s['stage_evaluation_protocol'],
        'matrices': {benchmark: calculate({'order': order, 'scores': matrix}) for benchmark, matrix in scores.items()},
        'source_markers': [{'stage_index': i, 'marker_sha256': digest(path),
            'plan_sha256': audits[i][0]['plan_sha256']} for i, path in enumerate(markers)],
        'training_stages': training_sources,
        'stage_evaluation_cost_records': [marker['new_evaluation_cost'] for marker, _, _ in audits],
        'actual_training_pools_pairwise_disjoint': True,
        'scope': 'Two complete target-language math matrices share the same English GSM8K retention column; English costs count once, not twice. Actual stage provenance, common prompts/seeds/configs, checkpoint chain and disjoint pools verified. Math accuracy/forgetting only: language-holding analyses, matched-pool single references, full recovery cost ledgers and comparison across directions remain separate required report components. Negative forgetting is retained. No language selection or new generation.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--selection-sha', required=True)
    parser.add_argument('--direction', choices=['forward', 'reverse'], required=True)
    parser.add_argument('--markers', type=Path, nargs=4, required=True, metavar=('BASE', 'STAGE1', 'STAGE2', 'STAGE3'))
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    assert not a.output.exists()
    result = build(a.root, Path(__file__).parent, a.selection, a.selection_sha, a.direction, a.markers)
    write_new(a.output, result)
    print(json.dumps({'order_id': result['order_id'], 'complete_math_matrices': 2}))


if __name__ == '__main__':
    main()
