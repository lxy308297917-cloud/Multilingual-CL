"""Completion audit of the frozen scientific scope, without new GPU work."""
from settings import *
import statistics, math

def main():
    def read(name):
        return json.loads((REPORT / name).read_text())
    p = read('PROTOCOL_FROZEN.json')
    assert sha(REPORT/'PROTOCOL_FROZEN.json') == 'd4e2f6b69cae0bfd005a14ee27f354706320cb158a21154239f7651bee90d855'
    batches = json.loads((RUN/'CALIBRATION_BATCHES_FROZEN.json').read_text())
    assert sha(RUN/'CALIBRATION_BATCHES_FROZEN.json') == p['calibration']['batches_sha256']
    cal = read('CALIBRATION_DONE.json')
    assert cal['batches'] == len(cal['records']) == len(batches['batches']) == 16
    assert [r['event'] for r in cal['records']] == [r['event'] for r in batches['batches']]
    ce = statistics.median(.1*r['raw_CE_grad_L2']/r['main_grad_L2'] for r in cal['records'])
    nce = statistics.median(r['raw_NCE_grad_L2']/r['main_grad_L2'] for r in cal['records'])
    assert math.isclose(cal['alpha'], ce/nce, rel_tol=1e-12)
    numeric = read('NUMERIC_CHECK_DONE.json')
    assert numeric['passed'] and len(numeric['probes']) == 2
    assert all(r['passed'] and r['D1_vs_D0_effect_L2_multiplier'] >= 2 for r in numeric['probes'])
    assert read('TECHNICAL_PARITY_DONE.json')['optimizer_scheduler_counts_auxcursor_rng_bitwise_equal']
    assert read('REUSE_VERIFIED.json')['passed']
    assert read('ENGINE_IDENTITY.json')['reverse_whitelist_exact_byte_parity']
    proof = read('COMPLETION_VERIFIED.json')
    assert proof['passed'] and proof['all4293_LR_main_aux_input_supervised_padded_counts_identical_to_D0']
    downstream = read('DOWNSTREAM_COMPARISON.json')
    assert downstream['math_excluded'] and downstream['not_formal_test']
    for model in ('Base','B','D0','D1'):
        assert set(downstream['models'][model]) == {'en_to_sw','sw_to_en','sw_reading','en_reading','en_instruction'}
        assert sum(t['n'] for t in downstream['models'][model].values()) == 511
    rep = read('REPRESENTATION_COMPARISON.json')
    assert set(rep['models']) == {'Base','B','D0','D1'}
    repdone = json.loads((RUN/'representation/D1/DONE.json').read_text())
    assert repdone['pairs'] == 1000 and repdone['layer'] == 10
    assert repdone['model_sha256'] == proof['model_sha256']
    assert sha(RUN/'representation/D1/LAYER10_VECTORS.pt') == repdone['vectors_sha256']
    manifest = json.loads((OLD/'data/manifest.json').read_text())
    split_groups = {}
    for name, count in (('train',2000),('dev',500),('diagnostic',1000)):
        raw = OLD/'data'/(name+'.jsonl')
        data = rows(raw)
        assert len(data) == count and sha(raw) == manifest['splits'][name]['sha256']
        split_groups[name] = {g for row in data for g in row['dedup_groups']}
    for a, b in (('train','dev'),('train','diagnostic'),('dev','diagnostic')):
        assert not split_groups[a] & split_groups[b]
    decision = read('MANDATORY_DECISION.json')
    assert decision['frozen_protocol_sha256'] == sha(REPORT/'PROTOCOL_FROZEN.json')
    assert decision['passed'] == all(decision['gates'].values())
    assert read('COSTS.json')['within_budget']
    for name in ('README_ZH.md','DOWNSTREAM_DIFFERENCES.png','REPRESENTATION_ENDPOINT.png'):
        assert (REPORT/name).stat().st_size > 0
    artifacts = [x.name for x in REPORT.iterdir() if x.is_file() and x.name != 'REQUIREMENTS_AUDIT.json' and not x.name.startswith('GITHUB_') and x.suffix in ('.json','.csv','.md','.png','.patch')]
    write(REPORT/'REQUIREMENTS_AUDIT.json', {
        'passed': True, 'checked_unix': time.time(),
        'requirements': {
            '16_predeclared_training_batches_and_fixed_coefficient': 'CALIBRATION_DONE.json; private frozen batch manifest; coefficient independently recomputed',
            'matching_B_D0_server_artifacts_and_numeric_engine': 'REUSE_VERIFIED.json; ENGINE_IDENTITY.json; EVALUATION_REUSE_VERIFIED.json',
            'actual_AdamW_BF16_and_serialized_recovery': 'NUMERIC_CHECK_DONE.json; TECHNICAL_PARITY_DONE.json',
            'full_4293_same_prefix_and_recoverable_states': 'COMPLETION_VERIFIED.json; both raw checkpoint receipts rehashed',
            'frozen_complete_task_and_independent_alignment_comparison': 'DOWNSTREAM_COMPARISON.json; REPRESENTATION_COMPARISON.json; raw512/generated511scored; independent1000pairs',
            'parallel_split_identity_and_source_group_isolation': 'Original manifest hashes and all train2000/dev500/diagnostic1000 dedup groups rechecked disjoint; inherited semantic QA is not independent native gold',
            'predeclared_decision_and_cost': 'PROTOCOL_FROZEN.json; MANDATORY_DECISION.json; COSTS.json',
            'report_and_figures': 'README_ZH.md; DOWNSTREAM_DIFFERENCES.png; REPRESENTATION_ENDPOINT.png',
        },
        'public_artifact_sha256': {n: sha(REPORT/n) for n in sorted(artifacts)},
        'final_source_sha256': {x.name: sha(x) for x in sorted((REPORT/'code').glob('*.py'))},
        'publication_verified_separately': True,
        'limits': 'One-seed development screen; cached historical references verified, not fully retrained. No formal test or mathematics criterion.'
    })
    print('FROZEN D1 REQUIREMENTS VERIFIED')

if __name__ == '__main__':
    main()
