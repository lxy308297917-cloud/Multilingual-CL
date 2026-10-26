"""Account for ten completed primary training lineages, including TE recovery.

Accepted token/forward counts and actual whole-job allocation are distinct.
No inference, source mutation, or extrapolation of missing cost receipts.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

from audit_primary4b_training_sources import PLANS

QUEUES = {'sw': 'copsd-sw4b-server-pilot20-v1', 'te': 'copsd-te4b-resume5-v1',
    'th': 'copsd-th4b-single20-v1', 'bn': 'copsd-bn4b-server20-v1',
    'zh': 'copsd-zh4b-server20-v1', 'ja': 'copsd-ja4b-server20-alloc-v2',
    'ru': 'copsd-ru4b-single20-v1', 'de': 'copsd-de4b-single20-v1',
    'fr': 'copsd-fr4b-single20-v1', 'es': 'copsd-es4b-single20-v1'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def summarize(root):
    code = Path(__file__).resolve().parent
    evidence, events, models = {}, {}, {}

    def bound(path):
        evidence[str(path.relative_to(root))] = digest(path)
        return read(path)

    source = bound(code / 'PRIMARY4B_TRAINING_SOURCE_AUDIT_20261026.PUBLIC.json')

    def event(queue, expected_phase, plan):
        paths = sorted((root / 'gpu-queue/queues' / queue / 'runs').glob('*/job.json'))
        assert len(paths) == 1, (queue, 'ambiguous queue lineage')
        path = paths[0]
        job = bound(path)
        assert job['phase'] == expected_phase and job['finished_at']
        assert len(job['gpu_uuids']) == len(set(job['gpu_uuids'])) == 2
        script = Path(job['script'])
        assert digest(script) == job['script_sha256'] == plan['code_sha256'][script.name]
        seconds = (datetime.datetime.fromisoformat(job['finished_at']) - datetime.datetime.fromisoformat(job['started_at'])).total_seconds()
        assert seconds >= 0
        key = queue + '/' + path.parent.name
        assert key not in events
        events[key] = {'phase': expected_phase, 'allocated_GPU_hours': 2 * seconds / 3600,
            'source_receipt_sha256': digest(path)}
        return key

    for language, name in PLANS.items():
        assert digest(code / name) == source['eleven_training_plans_sha256'][name]
        plan = bound(code / name)
        folder = root / plan['checkpoint_relative']
        done = bound(folder / 'SERVER_PILOT20_UPDATES_DONE.json')
        assert digest(folder / 'SERVER_PILOT20_UPDATES_DONE.json') == source['ten_primary_models'][language]['completion_receipt_sha256']
        cost = dict(done['cost'])
        batches = done['batch_generation']
        generated = sum(batch['generated_tokens'] for batch in batches)
        rollouts = sum(batch['rollouts'] for batch in batches)
        lineage = [event(QUEUES[language], 'completed', plan)]
        if language == 'te':
            old_plan = bound(code / 'te4b_server20_frozen_20261024.PUBLIC.json')
            lineage.append(event('copsd-te4b-server20-v1', 'failed', old_plan))
            old = root / plan['resume_checkpoint_relative']
            assert read(old / 'trainer_state.json')['global_step'] == 5
            trace = old.parent / 'SERVER_ROLLOUTS.PRIVATE.jsonl'
            evidence[str(trace.relative_to(root))] = digest(trace)
            # Physical LF only: U+2028 is legal inside JSON string values.
            records = [json.loads(line) for line in trace.read_text().split('\n') if line]
            kept = records[:160]
            assert len(kept) == 160 and all(row['global_step'] < 5 for row in kept)
            generated += sum(len(row['completion_ids']) for row in kept)
            rollouts += len(kept)
            retained = done['retained_checkpoint5_cost']
            for key in cost:
                cost[key] += retained[key]
        assert rollouts == cost['loss_calls'] == cost['student_forward_calls'] == cost['teacher_forward_calls'] == 640
        assert generated == cost['valid_response_loss_positions']
        models[language] = {'accepted_updates': 20, 'accepted_rollouts': rollouts,
            'accepted_generated_tokens': generated, 'accepted_forward_and_loss_cost': cost,
            'whole_lineage_queue_events': lineage,
            'whole_lineage_allocated_GPU_hours': sum(events[key]['allocated_GPU_hours'] for key in lineage),
            'last_successful_segment_training_seconds_including_diagnostic_hashing': done['train_seconds_including_diagnostic_hashing']}
    assert len(models) == 10 and len(events) == 11
    return {'observed_at': datetime.datetime.now().astimezone().isoformat(),
        'cohort': 'Original2700 ten independent Qwen3-4B single20 models.', 'models': models,
        'unique_lineage_events': events,
        'whole_lineage_allocated_GPU_hours': sum(row['allocated_GPU_hours'] for row in events.values()),
        'accepted_rollouts': sum(row['accepted_rollouts'] for row in models.values()),
        'accepted_generated_tokens': sum(row['accepted_generated_tokens'] for row in models.values()),
        'accepted_teacher_forward_calls': sum(row['accepted_forward_and_loss_cost']['teacher_forward_calls'] for row in models.values()),
        'source_evidence_sha256': evidence, 'collector_sha256': digest(Path(__file__)),
        'scope': 'Accepted checkpoint20 token and forward counts exclude discarded TE work. Whole-lineage allocation includes the entire failed TE predecessor plus successful recovery; no artificial split of failed-job wall time. Other abandoned pre-lineage attempts, prelaunch GPU holds, development and evaluation are excluded. These11 events overlap the all-queue ledger; never add them. CUDA event forward seconds are component timing, not total training time, energy or money. Last successful segment time is not total TE lineage duration.',
        'GPU_work_started_or_changed': False, 'full_goal_complete': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    result = summarize(args.root.resolve())
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({key: result[key] for key in ['accepted_rollouts', 'accepted_generated_tokens',
        'accepted_teacher_forward_calls', 'whole_lineage_allocated_GPU_hours']}))


if __name__ == '__main__':
    main()
