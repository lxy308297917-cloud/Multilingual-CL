"""Recount complete existing Aya80/MT20 stream, never extend a short-run endpoint."""
from settings import *
import collections,shutil

assert not (REPORT/'DATA_FROZEN.json').exists()
folder=H/'reproductions/sw_recipe_fft_20261010/data/aya80_mt20'
old=read(folder/'manifest.json'); indices=read(folder/'indices.json')
assert sha(old['source_file'])==old['source_sha256']
assert hashlib.sha256(json.dumps(indices,separators=(',',':')).encode()).hexdigest()==old['indices_sha256']
rows={}; needed=set(indices);counts=collections.defaultdict(collections.Counter);ids=[];groups=set();longest=[]
with Path(old['source_file']).open() as f:
 for i,s in enumerate(f):
  if i in needed:
   r=json.loads(s);assert len(r['input_ids'])==len(r['labels'])<=2048
   assert any(y!=-100 for y in r['labels'][1:]);rows[i]=(r['bucket'],len(r['input_ids']),sum(y!=-100 for y in r['labels'][1:]),r['id'])
   ids.append(r['id']);groups.add(r['dedup_group']);longest.append((len(r['input_ids']),i))
assert len(rows)==len(needed)==len(indices)==84930
for i in indices:
 bucket,n,k,_=rows[i];counts[bucket].update(presentations=1,input_tokens=n,supervised_tokens=k)
n=sum(v['input_tokens'] for v in counts.values());k=sum(v['supervised_tokens'] for v in counts.values())
assert (n,k)==(15360935,5984371) and len(indices)%2==0
source_manifest=Path(old['source_file']).parent/'manifest.json'
assert sha(source_manifest)==old['source_manifest_sha256']
dev=Path(old['source_file']).parent/'dev.jsonl'
with dev.open() as f:
 assert not groups & {json.loads(s)['dedup_group'] for s in f}
write(RUN/'INDICES.json',indices)
previous=ROOT/'work/benchmark_direction_20261026/direction/CALIBRATION_POOL.json'
pool=[r for r in read(previous) if r['id']!='RACE:train:high3601.txt:2'];assert len(pool)==766 and all(r['id']!='alpaca:43259' for r in pool)
write(RUN/'CALIBRATION_POOL.json',pool)
# Same provenance/task-balanced ordered pool for static SSU and dynamic sensitivity.
cal=[];buckets={t:sorted([r for r in pool if r['task']==t],key=lambda r:r['id']) for t in ('instruction','reading','math')}
import random
for t in buckets:random.Random(42).shuffle(buckets[t])
cursor=collections.Counter()
for j in range(128):
 t=list(buckets)[j%3];cal.append(buckets[t][cursor[t]%len(buckets[t])]['id']);cursor[t]+=1
write(RUN/'STATIC_CALIBRATION_IDS.json',cal)
events=[]
for event,step in enumerate(range(0,len(indices)//2,200)):
 selected=[]
 for off in range(32):
  j=event*32+off;t=list(buckets)[j%3];selected.append(buckets[t][(j//3)%len(buckets[t])]['id'])
 events.append(dict(step=step,ids=selected))
write(RUN/'CALIBRATION_SCHEDULE.json',events)
shapes=read(ROOT/'outputs/module_mix50_v1/BUDGET_FROZEN.json')['shapes']
write(REPORT/'DATA_FROZEN.json',dict(source=str(Path(old['source_file']).resolve()),source_sha256=sha(old['source_file']),
 origin_manifest=str(folder/'manifest.json'),origin_manifest_sha256=sha(folder/'manifest.json'),
 indices_sha256=sha(RUN/'INDICES.json'),main_input_tokens=n,main_supervised_tokens=k,steps=len(indices)//2,
 seed=42,effective_batch=2,max_length=2048,counts=dict(counts),
 fractions={t:v['input_tokens']/n for t,v in counts.items()},unique_samples=len(rows),max_repetitions=1,
 complete_batch_boundary=True,target_input_tokens=15360000,budget_overshoot=n-15360000,
 longest_source_indices=[i for _,i in sorted(longest,reverse=True)[:2]],
 longest_lengths=[l for l,_ in sorted(longest,reverse=True)[:2]],
 train_dev_dedup_groups_disjoint=True,shapes=shapes,
 calibration_pool_sha256=sha(RUN/'CALIBRATION_POOL.json'),calibration_pool_examples=len(pool),
 static_ids_sha256=sha(RUN/'STATIC_CALIBRATION_IDS.json'),calibration_schedule_sha256=sha(RUN/'CALIBRATION_SCHEDULE.json'),
 protected_calibration_events=len(events),protected_calibration_presentations=sum(len(e['ids']) for e in events),
 calibration_mastery_certified=False,known_wrong_reference_excluded=['alpaca:43259'],benchmark_overlap_excluded=['RACE:train:high3601.txt:2'],
 original_source_manifest_sha256=sha(source_manifest),created_unix=time.time()))
print(json.dumps(dict(steps=len(indices)//2,input=n,supervised=k,counts=dict(counts)),indent=2))
