"""Posthoc anonymous packet, fixed content rubric; never alter frozen v2 scorer."""
import time,collections
from settings import *
assert not (RUN/'POSTHOC_MATH_REVIEW_RUBRIC.json').exists()
write(RUN/'POSTHOC_MATH_REVIEW_RUBRIC.json',{'frozen_unix':time.time(),'scope':'All320 fresh mathematics outputs; exact duplicate text for same question reviewed once and mapped back to all occurrences','annotation':'Assistant content review; not external human gold or independent confirmation','blinding_limits':'Packet omits model identities; reviewer has already seen aggregate automatic results and a small set of identity-labelled cases. Do not call this a strictly blind independent study.','labels':['correct','wrong','unclear'],'correct':'Unambiguous answer to requested target, with logically compatible units and numeric equivalent; accepted natural conclusion need not use answer prefix or box. Correct intermediate quantity alone is insufficient.','wrong':'Clear incorrect requested result, wrong relation/arithmetic/unit, answering a different requested quantity, or missing target after otherwise completed reasoning.','unclear':'Conflicting final claims, insufficient/ambiguous ending, or generation cut before an unambiguous target answer; never default-correct','truncated':'Keep all denominators; may be correct if a clear target answer already occurs, otherwise unclear/wrong with reason.','source_QA':'Use pre-generation corrected/clarified reference; flag any newly identified source uncertainty, retain all64-source denominator','frozen_v2_scorer_modified':False,'classification_use':'Separate POSTHOC_ASSISTANT_REVIEW, not a replacement primary metric or hyperparameter selection signal'})
source={r['id']:r for r in rows(RUN/'comparison.jsonl') if r['task']=='en_math'};unique={};mapping=[]
for model in MODELS:
    for r in rows(RUN/'generation'/model/'PREDICTIONS.jsonl'):
        if r['task']!='en_math':continue
        digest=hashlib.sha256((r['id']+'\0'+r['text']).encode()).hexdigest();code='H'+digest[:10]
        unique.setdefault(code,{'code':code,'source_id':r['id'],'question':source[r['id']]['prompt'],'gold':source[r['id']]['reference'],'expected_unit':source[r['id']]['expected_unit'],'text':r['text'],'capped':r['capped'],'occurrences':0});unique[code]['occurrences']+=1
        mapping.append({'code':code,'source_id':r['id'],'model':model,'original_score':r['scores'],'generation_capped':r['capped']})
assert len(mapping)==320
write(RUN/'POSTHOC_MATH_ANONYMOUS_PACKET.json',list(unique.values()));write(RUN/'POSTHOC_MATH_REVIEW_MAPPING_PRIVATE.json',mapping)
print('Anonymous unique answer reviews',len(unique),'of320')
