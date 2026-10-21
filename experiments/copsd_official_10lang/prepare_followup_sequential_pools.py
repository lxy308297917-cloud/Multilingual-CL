"""Prepare three disjoint pools excluding both originaldev100 and followup200.

No language choice, model training or evaluation is performed.
"""
import argparse
from collections import Counter, defaultdict
import datetime
import hashlib
import json
from pathlib import Path
import random
from prepare_sequential_pools import LANGUAGES, ENGLISH_FIELDS, quartiles, read, save, sha
from prepare_followup200 import normalized


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    root, code = a.root, Path(__file__).parent
    freeze_path = code / 'followup200_frozen_20261021.PUBLIC.json'
    frozen = read(freeze_path)
    exclusion = root / frozen['future_training_exclusion']['relative']
    assert sha(exclusion) == frozen['future_training_exclusion']['sha256']
    excluded = set(read(exclusion)['IDs'])
    folder = root / 'data/copsd_official_20261015/sequential_pools_followup200_20261021_v1'
    public = code / 'sequential_pools_followup200_20261021.PUBLIC.json'
    assert not folder.exists() and not public.exists()
    inputs, reference, old_dev = {}, None, None
    input_dir = root / 'data/copsd_official_20261015/adapted_v2_ordered'
    for lang in LANGUAGES:
        path = input_dir / f'project_train_2900_{lang}.json'
        rows = read(path)
        assert len(rows) == len({str(r['problem_id']) for r in rows}) == 2900
        assert all(r.get('problem_' + lang) and r.get('problem_' + lang + '_ok') is True for r in rows)
        dev = read(input_dir / f'project_dev_100_{lang}.json')
        assert len(dev) == len({str(r['problem_id']) for r in dev}) == 100
        if reference is None: reference, old_dev = rows, dev
        else:
            assert [r['problem_id'] for r in rows] == [r['problem_id'] for r in reference]
            assert [r['problem_id'] for r in dev] == [r['problem_id'] for r in old_dev]
            assert all(all(r[k] == base[k] for k in ENGLISH_FIELDS) for r, base in zip(rows, reference))
        inputs[lang] = {'path': str(path.resolve()), 'sha256': sha(path)}
    assert inputs['sw']['sha256'] == frozen['source_inputs']['sw']['sha256']
    lookup = {str(r['problem_id']): r for r in reference}
    assert excluded <= set(lookup) and set(frozen['selected_IDs']) <= excluded
    development_ids = {r['problem_id'] for r in old_dev} | {lookup[i]['problem_id'] for i in excluded}
    dev_text = {normalized(r['problem']) for r in old_dev} | {normalized(lookup[i]['problem']) for i in excluded}
    eligible = [r for r in reference if str(r['problem_id']) not in excluded]
    assert len(eligible) == 2700
    assert len({normalized(r['problem']) for r in eligible}) == len(eligible)
    assert not {normalized(r['problem']) for r in eligible} & dev_text
    qbin, sbin = quartiles(eligible, 'problem'), quartiles(eligible, 'solution')
    groups = defaultdict(list)
    for row in eligible: groups[(row['source'], qbin[row['problem_id']], sbin[row['problem_id']])].append(row['problem_id'])
    rng = random.Random(20261021)
    pools = [[], [], []]
    differences = []
    for key in sorted(groups):
        group = groups[key].copy(); rng.shuffle(group); local = [0, 0, 0]
        for identifier in group:
            index = min(range(3), key=lambda k: (local[k], len(pools[k]), k))
            pools[index].append(identifier); local[index] += 1
        differences.append(max(local) - min(local))
    order = {r['problem_id']: i for i, r in enumerate(reference)}
    pools = [sorted(pool, key=order.__getitem__) for pool in pools]
    assert list(map(len, pools)) == [900, 900, 900]
    assert len(set().union(*map(set, pools))) == sum(map(len, pools)) == 2700
    assert not set().union(*map(set, pools)) & development_ids
    texts = [{normalized(lookup[str(i)]['problem']) for i in pool} for pool in pools]
    assert all(not texts[i] & texts[j] for i in range(3) for j in range(i))
    report = {'observed_at': datetime.datetime.now().isoformat(),
        'protocol': 'project_sequential_pools_excluding_followup200_v1', 'seed': 20261021,
        'languages_checked': list(LANGUAGES), 'pool_sizes': [900, 900, 900], 'union_size': 2700,
        'development_ids_excluded': len(development_ids), 'additional_training_IDs_excluded': len(excluded),
        'pairwise_ID_overlap': 0, 'pairwise_normalized_English_overlap': 0, 'development_overlap': 0,
        'english_fields_equal_across_languages': True,
        'source_counts': dict(Counter(r['source'] for r in eligible)),
        'stratification': 'source x English question character-length quartile x English reference character-length quartile',
        'difficulty_limit': 'Length strata are proxies, not verified mathematical difficulty or equal token costs; semantic near-duplicates are not exhaustively ruled out.',
        'stratum_pool_count_max_difference': max(differences),
        'binding_rule': 'After supported independent-development selection, bind chosen L1/L2/L3 to pools0/1/2; reverse the language/pool pairs together.',
        'matched_single_rule': 'Each language reference must start from Base with its identical bound900-question pool and same stage budget; old2900-question singles are not matched controls.',
        'exposure_note': '20updates x effectivebatch32 gives640 unique exposures inside each900-question pool, before its first epoch boundary. Longer budgets require actual epoch-end exposure accounting.',
        'followup_data_freeze_sha256': sha(freeze_path), 'followup_exclusion_sha256': sha(exclusion),
        'training_started': False, 'languages_selected': False,
        'JA_note': 'Manifest retains original input identity. Materialization must use the existing audited literal-prefix-repair option for JA; no new text repair.'}
    folder.mkdir()
    private = folder / 'POOLS.PRIVATE.json'
    save(private, {'seed': 20261021, 'inputs': inputs, 'pool_ids': pools,
        'development_ids': sorted(development_ids, key=str), 'report': report,
        'followup_data_freeze_sha256': sha(freeze_path), 'followup_exclusion_sha256': sha(exclusion)})
    pub = {**report, 'private_manifest_sha256': sha(private),
        'source_sha256': {name: sha(code / name) for name in ['prepare_followup_sequential_pools.py', 'prepare_sequential_pools.py', 'prepare_followup200.py']}}
    save(folder / 'REPORT.PUBLIC.json', pub)
    save(public, pub)
    print(json.dumps({'pool_sizes': pub['pool_sizes'], 'development_ids_excluded': pub['development_ids_excluded'],
                      'pairwise_normalized_English_overlap': 0, 'training_started': False, 'languages_selected': False}))


if __name__ == '__main__': main()
