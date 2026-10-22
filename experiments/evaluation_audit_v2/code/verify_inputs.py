"""Independent pre/post-inference identity check without changing source selection."""
import time
from settings import *
m=json.loads((RUN/'DATA_FROZEN.json').read_text());accept=json.loads((RUN/'ACCEPTANCE_FROZEN.json').read_text())
assert accept['data_manifest_sha256']==sha(RUN/'DATA_FROZEN.json');assert accept['scorer_sha256']==sha(pathlib.Path(__file__).parent/'scoring_v2.py')
parts={}
for role,info in m['partitions'].items():
    assert sha(info['path'])==info['sha256'];rr=rows(info['path']);assert len(rr)==info['rows'];assert len({r['id'] for r in rr})==len(rr);assert len({r['group_id'] for r in rr})==len(rr);parts[role]=rr
assert set(r['group_id'] for r in parts['acceptance']).isdisjoint(r['group_id'] for r in parts['comparison'])
assert len(parts['comparison'])==192 and all(sum(r['task']==t for r in parts['comparison'])==64 for t in ['sw_reading','en_reading','en_math'])
for path,expected in m['sources'].items():assert sha(path)==expected
for r in parts['acceptance']+parts['comparison']:
    assert 'johnson' not in r.get('source_text','').casefold() or 'arkansas' not in r.get('source_text','').casefold()
    assert not ('umuhimu wa elimu' in r.get('source_text','').casefold() and 'mfumo wa mafunzo' in r.get('source_text','').casefold())
    if r['task']=='en_math':assert r.get('expected_unit')
configs={name:{'path':path,'config_sha256':sha(pathlib.Path(path)/'config.json'),'generation_config_sha256':sha(pathlib.Path(path)/'generation_config.json')} for name,path in MODELS.items()}
effective={}
for name,info in configs.items():
    config=json.loads((pathlib.Path(info['path'])/'generation_config.json').read_text());config.pop('transformers_version',None);config['do_sample']=False;effective[name]=config
assert all(v==effective['Base'] for v in effective.values())
code={p.name:sha(p) for p in pathlib.Path(__file__).parent.glob('*') if p.is_file()}
write(REPORT/'INPUT_AND_CODE_IDENTITIES.json',{'verified_unix':time.time(),'source_data_hashes_match':True,'acceptance_and_comparison_group_overlap':0,'comparison_rows_per_task':64,'all_math_expected_units_frozen':True,'data_manifest_sha256':sha(RUN/'DATA_FROZEN.json'),'scorer_sha256':accept['scorer_sha256'],'scorer_acceptance_sha256':sha(RUN/'ACCEPTANCE_FROZEN.json'),'source_QA_changes_sha256':sha(RUN/'SOURCE_QA_CHANGES.json'),'models':configs,'effective_generation_config_identical':True,'effective_generation_settings':effective['Base'],'generation_hash_difference_reason':'Version metadata only; decoded settings identical after explicit greedy override.','code_sha256':code,'scope':'Raw-source hashes and grouped files verified; lexical exclusion is not semantic decontamination proof. Model weight identities separately hashed in GPU CONFIG and original checkpoint receipts.'})
print('Independent input/code verification passed')
