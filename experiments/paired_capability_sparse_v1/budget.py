"""Physical reserved-GPU accounting from this experiment's manual queue receipts.

The parent queue covers all serial children exactly once. Never add logical
borrowed-prefix durations a second time. Failed and technical jobs are charged.
"""
import datetime,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
QUEUES=Path('/data0/users/lixiaoyang/gpu-queue/queues')

def software_versions():
    import importlib.metadata
    return {name:importlib.metadata.version(name) for name in ('torch','transformers','numpy','sacrebleu','nltk','langdetect')}

def verify_software(release):
    if release['software_versions']!=software_versions():raise RuntimeError('Frozen software versions changed')


def ledger(now=None):
    now=now or datetime.datetime.now();rows=[]
    for path in sorted(QUEUES.glob('paired-cap-*/runs/*/job.json')):
        job=json.loads(path.read_text())
        script=Path(job.get('script',''))
        if not str(script).startswith(str(ROOT.parents[2])+'/') or script.parent.name!=ROOT.name:continue
        if not job.get('started_at'):continue
        begin=datetime.datetime.strptime(job['started_at'],'%Y-%m-%d %H:%M:%S')
        end=datetime.datetime.strptime(job['finished_at'],'%Y-%m-%d %H:%M:%S') if job.get('finished_at') else now
        seconds=max(0.,(end-begin).total_seconds())*len(job.get('gpu_uuids',[]))
        if not job.get('gpu_uuids'):raise RuntimeError('GPU receipt lacks assignment: '+str(path))
        rows.append({'receipt':str(path),'script':job['script'],'phase':job.get('phase'),'gpu_seconds':seconds,'finished':bool(job.get('finished_at'))})
    return {'jobs':rows,'physical_gpu_hours_including_failures':sum(r['gpu_seconds'] for r in rows)/3600,'accounting':'whole queue job reservation; serial children and borrowed prefixes never double counted','at_local':now.isoformat()}

def remaining_seconds():
    protocol=json.loads((ROOT/'comparison_protocol.v1.json').read_text());hours=protocol['budget']['total_gpu_hours_including_failed_jobs']-ledger()['physical_gpu_hours_including_failures']
    if hours<=0:raise RuntimeError('Frozen total GPUh budget exhausted')
    return hours*3600

def stage_deadline(begun,allowance_seconds):
    remaining_local=allowance_seconds-(time.monotonic()-begun)
    if remaining_local<=0:raise RuntimeError('Frozen local GPUh budget exhausted')
    return time.monotonic()+min(remaining_local,remaining_seconds())
if __name__=='__main__':print(json.dumps(ledger(),indent=2))
