"""Conservative body-only exact token-window screen; no semantic guarantee."""
import collections,hashlib,json
from pathlib import Path
from transformers import AutoTokenizer
CAL=Path('/root/autodl-tmp/sw_update_utility_v1/calibration_candidates.jsonl')
DATA=Path('/root/autodl-tmp/d2_bilingual_v1/sw')
OUT=Path(__file__).parent/'D2_CALIBRATION_OVERLAP_AUDIT.json'
K=20;B=1000003;MOD=(1<<64)-1;POWER=pow(B,K-1,1<<64)
def windows(a):
    if len(a)<K:return
    v=0
    for t in a[:K]:v=((v*B)+(t+1))&MOD
    yield v
    for old,new in zip(a,a[K:]):
        v=(((v-((old+1)*POWER))&MOD)*B+(new+1))&MOD
        yield v
cal=[json.loads(s) for s in CAL.open()]
tok=AutoTokenizer.from_pretrained('/root/models/Qwen2.5-1.5B-Instruct',local_files_only=True)
known=collections.defaultdict(set)
for x in cal:
    for part in x['parts']:
        if not isinstance(part,str):continue
        ids=tok(part,add_special_tokens=False)['input_ids']
        for h in windows(ids):known[h].add(x['id'])
counts=collections.Counter();hits=collections.defaultdict(set);examples=[]
for split in ('train','dev'):
    with (DATA/(split+'.jsonl')).open() as f:
        for line in f:
            x=json.loads(line);counts[split]+=1
            ids=set()
            for h in windows(x['input_ids']):ids.update(known.get(h,()))
            if ids:
                hits[split].update(ids)
                if len(examples)<30:examples.append({'split':split,'data_id':x['id'],'calibration_ids':sorted(ids)[:10]})
result={'status':'screen_complete','method':'20-token exact rolling hash of calibration body parts against full D2 sample; hashes not independently rechecked; short/common content and semantic overlap not covered','calibration_sha256':hashlib.sha256(CAL.read_bytes()).hexdigest(),'d2_manifest_sha256':hashlib.sha256((DATA/'manifest.json').read_bytes()).hexdigest(),'rows':dict(counts),'overlapping_calibration_ids':{k:len(v) for k,v in hits.items()},'examples':examples}
OUT.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='examples'}))
