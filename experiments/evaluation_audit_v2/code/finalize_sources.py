"""Apply pre-generation source QA and reviewed unit metadata, preserve originals."""
import shutil,time
from settings import *
assert not (RUN/'ACCEPTANCE_FROZEN.json').exists()
original=RUN/'data_before_source_QA';original.mkdir(exist_ok=False)
for name in ['DATA_FROZEN.json','acceptance.jsonl','comparison.jsonl']:shutil.copyfile(RUN/name,original/name)
manifest=json.loads((RUN/'DATA_FROZEN.json').read_text());annotations=json.loads((RUN/'UNIT_ANNOTATIONS.json').read_text());changes=[]
for role in ('acceptance','comparison'):
    path=RUN/(role+'.jsonl');rr=rows(path)
    for r in rr:
        if r['id'] in annotations:r['expected_unit']=annotations[r['id']]['expected_unit']
        if r['id']=='gsm8k_train:474':
            assert r['reference']=='6';r['source_original_reference']='6';r['reference']='12'
            changes.append({'source_id':r['id'],'kind':'source_gold_correction','reason':'30 total,12 lemon,18 remaining,6 mango,12 orange; supplied6 counts mango rather than requested orange','before':'6','after':'12'})
        if r['id']=='gsm8k_train:4556':
            r['source_original_prompt']=r['prompt'];r['prompt']+=' Give the minimum whole number of hours per student per day.'
            changes.append({'source_id':r['id'],'kind':'pre_generation_question_clarification','reason':'Exact required mean is300/(20*7)=15/7; source gold3 assumes whole-hour ceiling. New prompt makes that condition explicit for every model.','reference_unchanged':'3'})
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rr));manifest['partitions'][role]['sha256']=sha(path)
assert len(changes)==2
write(RUN/'SOURCE_QA_CHANGES.json',{'changes':changes,'unit_annotations_sha256':sha(RUN/'UNIT_ANNOTATIONS.json'),'model_generations_before_changes':0,'assistant_review_not_external_human':True})
manifest.update(parent_manifest_sha256=sha(original/'DATA_FROZEN.json'),source_QA_sha256=sha(RUN/'SOURCE_QA_CHANGES.json'),unit_annotations_sha256=sha(RUN/'UNIT_ANNOTATIONS.json'),frozen_unix=time.time(),pre_generation_source_QA_counts={'gold_corrected':1,'question_clarified':1,'unit_annotated':76,'same_source_near_duplicate_families_excluded':2})
write(RUN/'DATA_FROZEN.json',manifest)
summary=json.loads((REPORT/'DATA_AGGREGATE_AUDIT.json').read_text());summary.update(data_manifest_sha256=sha(RUN/'DATA_FROZEN.json'),source_QA_counts=manifest['pre_generation_source_QA_counts'],limits=['Source-GSM training-side diagnostic with one disclosed answer correction and one clarified whole-hour question; not off-the-shelf official GSM8K','Assistant source review is not independent human gold; count noun semantics are not independently verified','Two pre-generation near-duplicate families excluded after top-neighbor review; lexical exclusion is not proof of complete semantic separation'])
write(REPORT/'DATA_AGGREGATE_AUDIT.json',summary)
print('Final source QA frozen',sha(RUN/'DATA_FROZEN.json'))
