"""Freeze one extra development200 for the audited existing20-update checkpoints."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import unicodedata


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def normalized(text): return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text)).strip()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    root, code = a.root, Path(__file__).parent
    dest = root / 'data/copsd_official_20261015/followup_dev200_20261021_v1'
    public = code / 'followup200_frozen_20261021.PUBLIC.json'
    assert not dest.exists() and not public.exists()
    evidence_path = code / 'prefix20_unseen_pool_feasibility_20261021.PUBLIC.json'
    evidence = read(evidence_path)
    footprints = read(root / 'logs/copsd_official_10lang/prefix20_unseen_IDs_20261021.PRIVATE.json')
    exposed = set()
    for name, spec in evidence['models_audited'].items():
        cp = root / spec['checkpoint_relative']
        assert sha(cp / 'adapter_model.safetensors') == spec['adapter_sha256']
        assert sha(cp / 'training_args.bin') == spec['training_args_sha256']
        ids = footprints['reconstructed_exposure_IDs'][name]
        assert len(ids) == len(set(ids)) == 640
        assert hashlib.sha256(json.dumps(ids).encode()).hexdigest() == spec['reconstructed_ID_sequence_sha256']
        exposed.update(ids)
    rows = {}
    sources = {}
    for lang in ['sw', 'ja', 'zh', 'bn']:
        variant = 'adapted_v3_ja_prefix_repair' if lang == 'ja' else 'adapted_v2_ordered'
        path = root / 'data/copsd_official_20261015' / variant / f'project_train_2900_{lang}.json'
        h = sha(path)
        assert h == evidence['models_audited'][lang + ('_unclip42' if lang == 'bn' else '42')]['training_data_sha256']
        rr = read(path)
        rows[lang] = {str(r['problem_id']): r for r in rr}
        assert len(rr) == len(rows[lang]) == 2900
        sources[lang] = {'relative': str(path.relative_to(root)), 'sha256': h}
    bank = rows['sw']
    assert set(bank) - exposed == set(footprints['remaining_IDs'])
    for lang, rr in rows.items():
        assert set(rr) == set(bank)
        for item_id in bank:
            assert all(rr[item_id][k] == bank[item_id][k] for k in ['problem', 'solution', 'Answer'])
    old_path = root / 'data/copsd_official_20261015/adapted_v2_ordered/project_dev_100_sw.json'
    old_cfg = read(root / 'logs/copsd_official_10lang/development_vllm8192/sw_base_dev100_8192_20261020_v1/CONFIG.json')
    assert sha(old_path) == old_cfg['input_sha256']
    old = read(old_path)
    old_ids = {str(r['problem_id']) for r in old}
    assert len(old_ids) == 100 and not set(bank) & old_ids
    old_text = {normalized(r['problem']) for r in old}
    exposed_text = {normalized(bank[item_id]['problem']) for item_id in exposed}
    by_text = {}
    removed_exposed_text, removed_old_text = [], []
    for item_id in sorted(set(bank) - exposed):
        text = normalized(bank[item_id]['problem'])
        assert text
        if text in exposed_text:
            removed_exposed_text.append(item_id)
        elif text in old_text:
            removed_old_text.append(item_id)
        else:
            by_text.setdefault(text, []).append(item_id)
    representatives = [min(ids) for ids in by_text.values()]
    salt = 'copsd_followup_dev200_20261021_v1'
    selected = sorted(representatives, key=lambda item_id: hashlib.sha256(f'{salt}:{item_id}'.encode()).hexdigest())[:200]
    assert len(selected) == len(set(selected)) == 200
    selected_text = {normalized(bank[item_id]['problem']) for item_id in selected}
    assert len(selected_text) == 200 and not selected_text & (old_text | exposed_text)
    exclude = sorted(item_id for item_id, r in bank.items() if normalized(r['problem']) in selected_text | old_text)
    assert not set(exclude) & exposed
    dest.mkdir()
    inputs = {}
    for lang, rr in rows.items():
        path = dest / f'project_followup_200_{lang}.json'
        path.write_text(json.dumps([rr[item_id] for item_id in selected], ensure_ascii=False, indent=2) + '\n')
        inputs[lang] = {'relative': str(path.relative_to(root)), 'sha256': sha(path)}
    inputs['en'] = dict(inputs['sw'])  # English evaluation uses the original problem field.
    exclusion = dest / 'EXCLUDE_FROM_FUTURE_TRAINING.PRIVATE.json'
    exclusion.write_text(json.dumps({'IDs': exclude, 'selected_IDs': selected,
        'normalized_English_sha256': [hashlib.sha256(t.encode()).hexdigest() for t in sorted(selected_text)],
        'scope': 'Exclude IDs and equivalent normalized English question text from all later training/reference pools.'}, indent=2) + '\n')
    cases = {'base': {'adapter_relative': None, 'evaluation_languages': ['sw', 'ja', 'zh', 'bn', 'en']}}
    for name, spec in evidence['models_audited'].items():
        cases[name] = {'adapter_relative': spec['checkpoint_relative'], 'adapter_sha256': spec['adapter_sha256'],
                       'evaluation_languages': [name[:2], 'en'],
                       'identity': 'project_unclipped_intervention' if name.startswith('bn') else 'default_component_clip0.05'}
    result = {'frozen_at': datetime.datetime.now().isoformat(), 'protocol': 'project_followup_dev200_single8192_c64_v1',
        'model_revision': '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e',
        'official_commit': '4f4410730e9b6775a30d20f4b6e141cba2e87590',
        'items_each': 200, 'selected_IDs': selected, 'ID_selection_salt': salt,
        'ID_selection_rule': 'One representative minimum lexicographic ID per unique normalized English question; rank by SHA256(salt:ID); first200. No labels or scores used.',
        'normalization': 'Unicode NFKC, collapse whitespace, strip; case and math punctuation preserved. Not a proof of absence of semantic near-duplicates.',
        'overlap_audit': {'reconstructed_exposed_IDs': len(exposed), 'candidate_IDs_before_text_audit': len(bank) - len(exposed),
            'candidate_IDs_removed_exposed_text': len(removed_exposed_text), 'candidate_IDs_removed_old_dev_text': len(removed_old_text),
            'unique_eligible_English_questions': len(representatives), 'selected_unique_questions': 200,
            'future_training_excluded_IDs_including_aliases': len(exclude)},
        'source_inputs': sources, 'old_dev100_sha256': sha(old_path), 'exposure_evidence_sha256': sha(evidence_path),
        'inputs': inputs, 'future_training_exclusion': {'relative': str(exclusion.relative_to(root)), 'sha256': sha(exclusion)},
        'cases': cases, 'max_new_tokens': 8192, 'max_num_seqs': 64, 'samples': 1,
        'temperature': 1.0, 'top_p': .95, 'top_k': -1, 'seed': 42,
        'primary_default_candidates': ['sw', 'ja', 'zh'], 'separate_intervention': ['bn_unclip42', 'bn_unclip43'],
        'selection_policy': 'One fixed confirmation set; no resampling, new checkpoint grid or automatic extra seeds. Default SW/JA/ZH candidates require positive target delta in both fixed training instances; report paired question-bootstrap95% CI of the mean effect across the two fixed models (10000 replicates, seed20261021). A supported candidate additionally has CI lower>0; review language, termination and English costs before selection. If fewer than three supported default candidates, do not force three. BN unclipped remains a separate adaptation, not silently mixed into the default ranking. Existing initialization/order differences are not a pure seed-only contrast.',
        'scope': 'Additional project development confirmation only for these existing20-update checkpoints. Unexposed is reconstructed from saved sampler settings, not direct per-row logs, and does not apply to historical100-update models. No formal test selection or new training. Existing sequential pools must be revised to exclude this set before research training.',
        'prepare_source_sha256': sha(Path(__file__))}
    public.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'overlap_audit': result['overlap_audit'], 'cases': len(cases), 'evaluation_cases': sum(len(x['evaluation_languages']) for x in cases.values()), 'new_training': False}))


if __name__ == '__main__': main()
