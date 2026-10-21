"""Inventory allocated GPU time from unique COPSD queue job receipts.

No GPU probing, process changes or cost inference from scientific result
reports. Open/unfinished receipts remain gaps instead of assumed zero cost.
"""
import argparse
from collections import Counter
import datetime
import hashlib
import json
from pathlib import Path

from sequential_evaluation import write_new


def timestamp(value):
    t = datetime.datetime.fromisoformat(value)
    return t if t.tzinfo else t.replace(tzinfo=datetime.timezone(datetime.timedelta(hours=8)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    queue_root = a.root / 'gpu-queue/queues'
    records, seen, duplicates = [], {}, []
    for path in sorted(queue_root.glob('copsd*/runs/*/job.json')):
        raw = path.read_bytes()
        receipt = json.loads(raw)
        key = (receipt['run_dir'], receipt['started_at'], receipt['pid'])
        if key in seen:
            duplicates.append({'receipt': str(path.relative_to(a.root)), 'same_execution_as': seen[key]})
            continue
        seen[key] = str(path.relative_to(a.root))
        gpus = receipt['gpu_uuids']
        assert len(gpus) == len(set(gpus)) and len(gpus) >= 1
        closed = bool(receipt.get('finished_at')) and receipt.get('exit_code') is not None
        hours = None
        if closed:
            elapsed = (timestamp(receipt['finished_at']) - timestamp(receipt['started_at'])).total_seconds()
            assert elapsed >= 0
            hours = elapsed * len(gpus) / 3600
        records.append({'queue': path.parents[2].name, 'queue_run_id': path.parent.name,
            'source_receipt_relative': str(path.relative_to(a.root)),
            'source_receipt_sha256': hashlib.sha256(raw).hexdigest(),
            'script_basename': Path(receipt['script']).name,
            'started_at': receipt['started_at'], 'finished_at': receipt.get('finished_at'),
            'recorded_phase': receipt['phase'], 'recorded_exit_code': receipt.get('exit_code'),
            'allocated_GPU_count': len(gpus), 'closed_receipt': closed,
            'allocated_GPU_hours': hours,
            'unclosed_receipt_is_not_a_live_process_claim': not closed})
    assert records
    closed = [r for r in records if r['closed_receipt']]
    def total(rows):
        return sum(r['allocated_GPU_hours'] for r in rows)
    by_status = {}
    for name, subset in [('exit0', [r for r in closed if r['recorded_exit_code'] == 0]),
            ('nonzero_exit', [r for r in closed if r['recorded_exit_code'] != 0])]:
        by_status[name] = {'jobs': len(subset), 'allocated_GPU_hours': total(subset)}
    by_script = {}
    for script in sorted({r['script_basename'] for r in closed}):
        group = [r for r in closed if r['script_basename'] == script]
        by_script[script] = {'jobs': len(group), 'allocated_GPU_hours': total(group)}
    write_new(a.output, {'observed_at': datetime.datetime.now().isoformat(),
        'scope': 'Only unique gpu-queue/queues/copsd*/runs/*/job.json receipts present on this server. Covers heterogeneous official-code project adaptation, engineering probes, reference training, diagnoses and evaluations; not a matched-method efficiency comparison or the complete cost of every OPD attempt.',
        'time_zone_for_naive_receipts': 'Asia/Shanghai UTC+08:00',
        'unique_job_receipts': len(records), 'duplicate_receipts_excluded': duplicates,
        'closed_job_receipts': len(closed), 'unclosed_receipts': len(records) - len(closed),
        'closed_allocated_GPU_hours': total(closed), 'closed_by_exit_status': by_status,
        'closed_by_script': by_script,
        'recorded_phase_counts': dict(Counter(r['recorded_phase'] for r in records)),
        'records': records,
        'accounting_boundaries': [
            'Allocated process wall time times distinct GPU count, not CUDA kernel time or energy.',
            'Includes startup, loading and any CPU work inside an allocated job; separate sub-stage reports overlap and must never be added to this ledger.',
            'Closed failure/cancellation receipts count their actual recorded allocation; nonzero exit alone does not identify cancellation versus failure.',
            'Open receipts are not treated as alive, completed or zero cost. They need reconciliation with actual processes or missing terminal evidence before a full experiment total can be claimed.',
            'GPU reservation/hold intervals before job startup, manual jobs without these receipts, missing or archived receipts and unfinished allocation are excluded.',
            'Model scale, training protocol and scientific roles differ across rows. No cross-method cost ranking or money estimate is inferred.'
        ], 'GPU_used_by_this_analysis': False, 'new_generation': False})
    print(json.dumps({'closed_jobs': len(closed), 'open_receipts': len(records)-len(closed),
        'closed_allocated_GPU_hours': total(closed), 'by_exit_status': by_status}))


if __name__ == '__main__':
    main()
