from pathlib import Path
import json,hashlib,random,re,datetime,statistics
import argparse
parser=argparse.ArgumentParser(description='Prepare only three disjoint English original pools; no language selection or GPU work.')
parser.add_argument('--root',type=Path,required=True)
a=parser.parse_args();root=a.root;code=Path(__file__).resolve().parent
m_path=root/'data/copsd_official_20261015/bn4b_pool2700_20261022_v1/MANIFEST.PRIVATE.json';m=json.loads(m_path.read_text());data=root/m['data_relative'];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert sha(data)==m['data_sha256'];rows=json.loads(data.read_text())
normalize=lambda x:re.sub(r'\s+',' ',x).strip().casefold()
assert len(rows)==len({str(x['problem_id']) for x in rows})==len({normalize(x['problem']) for x in rows})==2700
hold=json.loads((code/'followup200_frozen_20261021.PUBLIC.json').read_text());assert not {str(x['problem_id']) for x in rows}&set(hold['selected_IDs'])
# Ten equal rank bins balance English prompt length, not unobserved mathematical difficulty.
ordered=sorted(rows,key=lambda x:(len(x['problem']),str(x['problem_id'])));parts={x:[] for x in 'ABC'};rng=random.Random(20261023)
for start in range(0,2700,270):
 group=ordered[start:start+270];rng.shuffle(group)
 for offset,row in enumerate(group):parts['ABC'[offset%3]].append(row)
for pool in parts.values():rng.shuffle(pool)
assert all(len(v)==900 for v in parts.values())
ids={k:{str(x['problem_id']) for x in v} for k,v in parts.items()};norms={k:{normalize(x['problem']) for x in v} for k,v in parts.items()}
for a,b in [('A','B'),('A','C'),('B','C')]:assert not ids[a]&ids[b] and not norms[a]&norms[b]
out=root/'data/copsd_official_20261015/sequential_3pool_candidate_20261023_v1';out.mkdir(exist_ok=False)
manifest={'prepared_at':datetime.datetime.now().isoformat(),'seed':20261023,'source_manifest_sha256':sha(m_path),'source_data_sha256':sha(data),'pools':{},'selected_languages':None,'training_started':False,'scope':'Candidate stage-data interface only. No language selection, new training, final sequentialbudget or successclaim. Assign one pool per selected language, reuse same languagepool in reverse order and same-data Base single references. Requires independentdev language gate and frozencompleteconfiguration beforelaunch.','difficulty_gap':'Adapted2700 rows have source=olympiads and no difficulty label. Bins match English character lengths only; do notclaim mathematicaldifficulty matching. Before use document available source labels or seeded random allocation as limited balancing; do not claim observed mathematicaldifficulty matching or reorder pools based on evaluation outcomes.'}
for part,pool in parts.items():
 entries=[{k:x[k] for k in ['problem_id','source','problem','solution','Answer']} for x in pool]
 p=out/f'pool_{part}_900.PRIVATE.json';p.write_text(json.dumps(entries,ensure_ascii=False,indent=2)+'\n')
 manifest['pools'][part]={'rows':900,'relative':str(p.relative_to(root)),'sha256':sha(p),'IDs':[str(x['problem_id']) for x in pool],'English_char_median':statistics.median(len(x['problem']) for x in pool),'source_counts':{'olympiads':900}}
(out/'MANIFEST.PRIVATE.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
summary={k:v for k,v in manifest.items() if k not in ['pools','source_manifest_sha256','source_data_sha256']}
summary['pools']={k:{key:v[key] for key in ['rows','English_char_median','source_counts']} for k,v in manifest['pools'].items()};summary['ID_and_normalized_English_cross_pool_overlap']=0;summary['original_source_row_fields_preserved']=True;summary['reserved_development200_ID_overlap']=0
(code/'sequential_3pool_interface_prepared_20261023.PUBLIC.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'pools':summary['pools'],'pairwise_overlap':0,'training_started':False,'difficulty_matching_claimed':False}))
