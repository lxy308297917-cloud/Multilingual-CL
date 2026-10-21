"""Audit measured training-loop cost for five completed batched100 singles.

Historical clocks only; no extrapolation from smoke runs or extra GPU work.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    assert not a.output.exists()
    code = Path(__file__).parent
    inventory_path = code / 'ten_language_training100_inventory_20261021.PUBLIC.json'
    inventory = read(inventory_path)
    results = {}
    for language in ['de', 'es', 'fr', 'ru', 'zh']:
        cp = a.root / inventory['languages'][language]['checkpoint_relative']
        assert read(cp / 'trainer_state.json')['global_step'] == 100
        model = cp.parent
        proof = read(model / 'COLLOCATE_TRAIN_UPDATES_DONE.json')
        cost = read(model / 'COST_SEGMENT_DONE.json')
        groups = read(model / 'UPDATE_BATCH_GENERATION.json')['groups']
        assert proof['start_step'] == cost['start_step'] == 0
        assert proof['global_step'] == cost['global_step'] == proof['completed_updates'] == 100
        assert proof['base_unchanged_bitwise'] and proof['effective_batch'] == 32
        assert proof['max_completion_length'] == 2048
        assert proof['micro_steps'] == cost['loss_calls'] == cost['teacher_forward_calls'] == cost['student_forward_calls'] == 3188
        assert [g['global_step'] for g in groups] == list(range(100))
        assert sum(g['rollouts'] for g in groups) == 3188
        assert read(model / 'RUNTIME_SHUTDOWN_DONE.json')['ordered_pool_release_completed']
        generated = sum(g['generated_tokens'] for g in groups)
        assert 0 < cost['valid_response_loss_positions'] <= generated
        seconds = proof['train_seconds_including_diagnostic_hashing']
        assert seconds > 0
        results[language] = {'run_id': model.name, 'updates': 100, 'question_exposures': 3188,
            'training_loop_seconds': seconds, 'training_loop_hours': seconds / 3600,
            'training_generated_tokens': generated,
            'valid_response_loss_positions': cost['valid_response_loss_positions'],
            'teacher_forward_calls': cost['teacher_forward_calls'],
            'teacher_forward_cuda_seconds': cost['teacher_forward_cuda_seconds'],
            'student_forward_cuda_seconds': cost['student_forward_cuda_seconds'],
            'policy_sync_seconds_including_diagnostic_hashing': sum(s['seconds_including_diagnostic_hashing'] for s in proof['syncs']),
            'source_sha256': {name: digest(model / name) for name in [
                'COLLOCATE_TRAIN_UPDATES_DONE.json', 'COST_SEGMENT_DONE.json', 'UPDATE_BATCH_GENERATION.json']}}
    result = {'observed_at': datetime.datetime.now().isoformat(), 'model': inventory['model'],
        'model_revision': inventory['model_revision'], 'training_generation_budget': 2048,
        'effective_batch': 32, 'scientific_optimizer_or_loss_budget_reduced': False,
        'source_inventory_sha256': digest(inventory_path), 'completed_full100_run_count': 5,
        'results': results,
        'training_loop_hours_range': [min(x['training_loop_hours'] for x in results.values()),
                                      max(x['training_loop_hours'] for x in results.values())],
        'new_training': False, 'new_generation': False, 'GPU_used': False,
        'scope': 'Five unique full0-to100 single-GPU batched-vLLM training loops, from actual completion proofs. Includes updates, rollout, saves and diagnostic hashing inside the observed loop; excludes initial model/engine startup, queue waiting, later development/formal evaluation and failed/discarded work. Sync timing includes real weight synchronization as well as hashing, not a pure removable-check overhead. Counts3188 include the short epoch-boundary update; do not report nominal3200 as actual exposures. This is not the old serial single-GPU SW runtime, a future completion guarantee, or proof of method effectiveness. Costs overlap earlier reports and must not be added twice.'}
    with a.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'actual_completed100_training_loop_hours': {k: round(v['training_loop_hours'], 3) for k, v in results.items()}}))


if __name__ == '__main__':
    main()
