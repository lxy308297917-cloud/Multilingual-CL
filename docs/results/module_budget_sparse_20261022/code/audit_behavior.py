from settings import *
import hashlib
r=RUN;p=REPORT
by={g:{x['id']:x for x in map(json.loads,(r/'evaluation'/g/'PREDICTIONS.jsonl').read_text().splitlines())} for g in ['S-All','S-Attn','S-MLP']}
for g in by:
 done=json.loads((r/'evaluation'/g/'DONE.json').read_text());assert hashlib.sha256((r/'evaluation'/g/'PREDICTIONS.jsonl').read_bytes()).hexdigest()==done['predictions_sha256']
ids=[x['id'] for x in by['S-All'].values() if x['task']=='en_to_sw' and x['language_diagnostic']=='en' and by['S-Attn'][x['id']]['language_diagnostic']=='sw'][:2]
(r/'QUALITATIVE_CASES.json').write_text(json.dumps({'selection':'First2 in frozen order with All langdetect=en and Attn langdetect=sw; posthoc illustrative, not random or native gold','cases':[{g:by[g][i] for g in by} for i in ids]},ensure_ascii=False,indent=2))
d=json.loads((p/'DOWNSTREAM_COMPARISON.json').read_text())
audit={'posthoc_behavior_only_not_new_ranking':True,'language_detector_not_accuracy':True,'ENtoSW':{g:{k:d['models'][g]['en_to_sw'][k] for k in ['languages','capped','mean_repeat_4gram_fraction']} for g in by},'case_selection':'First2 frozen-order outputs with All detected English and Attn detected Swahili; assistant inspection, not independent native gold','case_ids':ids,'findings':['Both inspected All/MLP outputs retain English sentences despite requested Swahili translation','Attention output switches language but first case repeats heavily to cap and wording does not establish a correct semantic translation','Higher target-language detection rate must not be reported as translation accuracy'],'raw_cases_private_path':str(r/'QUALITATIVE_CASES.json')}
(p/'BEHAVIOR_AUDIT.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
print(json.dumps({'cases':len(ids),'no_raw_question_or_response_publication':True}))
