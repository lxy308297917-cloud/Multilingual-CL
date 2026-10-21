"""CPU feasibility audit of unexposed questions for existing20-update candidates.

Does not select languages, allocate a new split, evaluate answers or submit jobs.
Exposure IDs are reconstructed from saved sampler settings, not originally logged IDs.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--private-output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists() and not a.private_output.exists()
    import torch
    from torch.utils.data import DataLoader, RandomSampler
    from accelerate import Accelerator, DataLoaderConfiguration
    from accelerate.data_loader import SeedableRandomSampler
    from transformers import set_seed
    names = {
        'sw42': 'sw_clip20_dev100_8192_20261020_v1',
        'sw43': 'sw_17b_seed43_prefix20_dev100_8192_20261020_v1',
        'ja42': 'ja_v4_saved20_target_dev100_8192_20261020_v1',
        'ja43': 'ja_17b_seed43_prefix20_dev100_8192_20261021_v1',
        'zh42': 'zh_early20_candidate_dev100_8192_20261020_v1',
        'zh43': 'zh_17b_seed43_prefix20_dev100_8192_20261021_v1',
        'bn_unclip42': 'bn_17b_unclip20_horizon100_seed42_20261020_v1_dev8192',
        'bn_unclip43': 'bn_17b_unclip20_horizon100_seed43_20261020_v2_dev8192',
    }
    results, exposed, common = {}, set(), None
    all_ids = {}
    for label, run in names.items():
        lang = label[:2]
        cfg = read(a.root / 'logs/copsd_official_10lang/development_vllm8192' / run / 'CONFIG.json')
        cp = Path(cfg['adapter'])
        state = read(cp / 'trainer_state.json')
        assert state['global_step'] == 20 and state['max_steps'] == 100
        args = torch.load(cp / 'training_args.bin', map_location='cpu', weights_only=False)
        assert args.per_device_train_batch_size == 1 and args.gradient_accumulation_steps == 32
        assert args.accelerator_config.use_seedable_sampler and not args.group_by_length
        assert not args.dataloader_drop_last
        assert args.seed in [42, 43] and args.data_seed == args.seed
        data_dir = 'adapted_v3_ja_prefix_repair' if lang == 'ja' else 'adapted_v2_ordered'
        source = a.root / 'data/copsd_official_20261015' / data_dir / f'project_train_2900_{lang}.json'
        rows = read(source)
        assert len(rows) == 2900
        assert all(isinstance(r.get('problem_' + lang), str) and r['problem_' + lang].strip()
                   and (r['problem_' + lang + '_ok'] is True if 'problem_' + lang + '_ok' in r else True) for r in rows)
        ids = [str(r['problem_id']) for r in rows]
        assert len(set(ids)) == 2900
        if common is None: common = ids
        assert ids == common, label
        assert abs(state['epoch'] - 640 / 2900) < 1e-8, (label, state['epoch'])
        set_seed(args.seed)
        accelerator = Accelerator(cpu=True, dataloader_config=DataLoaderConfiguration(
            split_batches=False, dispatch_batches=False, even_batches=True,
            use_seedable_sampler=True, data_seed=args.data_seed))
        loader = accelerator.prepare_data_loader(DataLoader(range(2900), batch_size=1,
            sampler=RandomSampler(range(2900)), num_workers=0), device_placement=False)
        assert isinstance(loader.batch_sampler.sampler, SeedableRandomSampler)
        assert loader.batch_sampler.sampler.initial_seed == args.data_seed
        order = [int(batch.item()) for _, batch in zip(range(640), loader)]
        assert len(order) == len(set(order)) == 640
        seen = [ids[i] for i in order]
        all_ids[label] = seen
        exposed.update(seen)
        results[label] = {'checkpoint_relative': str(cp.relative_to(a.root)),
                          'adapter_sha256': sha(cp / 'adapter_model.safetensors'),
                          'training_args_sha256': sha(cp / 'training_args.bin'),
                          'training_data_sha256': sha(source), 'seed': args.seed,
                          'reconstructed_model_exposures': 640,
                          'reconstructed_ID_sequence_sha256': hashlib.sha256(json.dumps(seen).encode()).hexdigest()}
    unseen = sorted(set(common) - exposed)
    original_dev = read(a.root / 'data/copsd_official_20261015/adapted_v2_ordered/project_dev_100_sw.json')
    dev_ids = {str(r['problem_id']) for r in original_dev}
    assert not set(common) & dev_ids
    out = {'observed_at': datetime.datetime.now().isoformat(), 'models_audited': results,
           'common_ordered_training_IDs': 2900, 'union_reconstructed_exposed_IDs': len(exposed),
           'remaining_unexposed_IDs': len(unseen), 'original_dev100_disjoint': True,
           'new_dev_or_training_split_created': False, 'new_evaluation_or_GPU_task': False,
           'exposure_provenance': 'Reconstruction via actual saved seedable-sampler settings and CPU Accelerate DataLoader, epoch0,640 examples, same ordered data; not direct per-row training logs.',
           'scope': 'Feasibility only for these frozen20-update checkpoints. NOT unseen for historical100-update models. No new question selection, scores, final languages or claim of stable benefit. Any future follow-up split must be frozen before scoring and excluded from subsequent training/pools; cannot repeatedly resample to seek positive results.'}
    with a.private_output.open('x') as f:
        json.dump({'reconstructed_exposure_IDs': all_ids, 'remaining_IDs': unseen}, f, ensure_ascii=False, indent=2)
        f.write('\n')
    with a.output.open('x') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({k: out[k] for k in ['common_ordered_training_IDs', 'union_reconstructed_exposed_IDs', 'remaining_unexposed_IDs', 'new_dev_or_training_split_created']}))


if __name__ == '__main__': main()
