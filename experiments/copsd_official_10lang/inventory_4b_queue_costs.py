"""Snapshot distinct recorded 4B queue runs, including failed attempts.

This accounts for allocated parent GPU wall time, not CUDA compute time.
Queue-name coverage is explicit; it does not claim whole-project cost.
"""
import argparse
from collections import Counter
import datetime
import hashlib
import json
from pathlib import Path


def snapshot(root, output):
    root, output = Path(root).resolve(), Path(output)
    assert not output.exists()
    now = datetime.datetime.now()
    terminal, incomplete, events, seen = Counter(), [], [], set()
    for path in sorted((root / 'gpu-queue/queues').glob('copsd-*4b*/runs/*/job.json')):
        raw = path.read_bytes()
        job = json.loads(raw)
        queue, run_id = path.parents[2].name, path.parent.name
        identity = queue + '/' + run_id
        assert identity not in seen
        seen.add(identity)
        assert Path(job['run_dir']).resolve() == path.parent.resolve()
        devices = job['gpu_uuids']
        assert devices and len(devices) == len(set(devices))
        event = dict(event_id=identity, queue=queue, run_id=run_id,
                     source_relative=str(path.relative_to(root)),
                     source_sha256=hashlib.sha256(raw).hexdigest(),
                     phase=job['phase'], exit_code=job.get('exit_code'),
                     GPU_count=len(devices), started_at=job['started_at'],
                     script=job['script'], script_sha256=job['script_sha256'])
        started = datetime.datetime.fromisoformat(job['started_at'])
        finished = job.get('finished_at')
        if finished is None:
            event['observed_elapsed_GPU_hours'] = max(0, (now - started).total_seconds()) * len(devices) / 3600
            incomplete.append(event)
            continue
        end = datetime.datetime.fromisoformat(finished)
        assert end >= started and job['phase'] in ['completed', 'failed', 'cancelled']
        event['finished_at'] = finished
        event['parent_GPU_hours'] = (end - started).total_seconds() * len(devices) / 3600
        terminal[job['phase']] += event['parent_GPU_hours']
        events.append(event)
    assert events
    report = dict(observed_at=now.astimezone().isoformat(),
                  unique_terminal_parent_events=len(events),
                  terminal_parent_GPU_hours=sum(terminal.values()),
                  terminal_parent_GPU_hours_by_phase=dict(terminal),
                  terminal_events=events, unfinished_events=incomplete,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='Only recorded runs under queue-name glob copsd-*4b*. Distinct queue/run identities, not current state alone. Includes terminal failures/cancellations; unfinished elapsed time excluded from terminal total. Allocation wall time includes model loading and per-parent work, excludes waiting before job start. Token counts and teacher CUDA time require separate source-bound records. This is not total project cost: nonmatching queues, manually run work and older 1.7B cohorts are outside coverage. Do not add overlapping report subtotals to these events.')
    with output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = snapshot(args.root, args.output)
    print(json.dumps({k: report[k] for k in ['observed_at', 'unique_terminal_parent_events',
                                            'terminal_parent_GPU_hours', 'terminal_parent_GPU_hours_by_phase']}))
