"""Predeclared all-model, all-task paired-source comparison. No data omission."""
import csv,collections,shutil
import numpy as np
from settings import *

def main():
    data=json.loads((RUN/'DATA_FROZEN.json').read_text());rr=rows(data['partitions']['comparison']['path']);reference={r['id']:r for r in rr};models={};table=[];comparisons=[];review=[]
    for model in ['Base','A','B','C','D']:
        folder=RUN/'generation'/model;done=json.loads((folder/'DONE.json').read_text());assert sha(folder/'PREDICTIONS.jsonl')==done['predictions_sha256'];pred=rows(folder/'PREDICTIONS.jsonl');assert len(pred)==192 and {r['id'] for r in pred}==set(reference);models[model]={r['id']:r for r in pred}
        for task in ('sw_reading','en_reading','en_math'):
            pp=[r for r in pred if r['task']==task];assert len(pp)==64;key='content_f1' if task=='sw_reading' else 'content_correct';values=[r['scores'][key] for r in pp]
            row={'model':model,'task':task,'n':64,'content_metric':'token_f1' if task=='sw_reading' else 'accuracy','content_score':100*np.mean(values),'content_correct_count':sum(values) if task!='sw_reading' else None,'exact_count':sum(r['scores'].get('content_exact',r['scores'].get('content_correct',0)) for r in pp),'format_compliant':None if task=='sw_reading' else sum(r['scores'].get('format_correct',r['scores'].get('format_boxed',0)) for r in pp),'extraction_failed':sum(r['scores']['status']=='extraction_failed' for r in pp),'ambiguous':sum(r['scores']['status']=='ambiguous' for r in pp),'unit_needs_review':sum(r['scores'].get('unit_status')=='unit_needs_review' for r in pp),'unit_incompatible':sum(r['scores'].get('unit_status')=='incompatible_unit' for r in pp),'capped':sum(r['capped'] for r in pp),'repeated_4gram_over20pct':sum(r['behavior']['repeated_4gram_fraction']>.2 for r in pp),'generated_tokens':sum(r['generated_tokens'] for r in pp),'prompt_tokens':sum(r['prompt_tokens'] for r in pp),'language_diagnostic_counts':dict(collections.Counter(r['language_diagnostic'] for r in pp)),'mean_generated_tokens':np.mean([r['generated_tokens'] for r in pp])};table.append(row)
            categories={'parse_failure':[r for r in pp if r['scores']['status']=='extraction_failed'],'ambiguous':[r for r in pp if r['scores']['status']=='ambiguous'],'format_only_missed':[r for r in pp if r['scores'].get('content_correct',0) and r['scores'].get('format_correct',r['scores'].get('format_boxed',1))==0],'wrong_extracted_math':[r for r in pp if task=='en_math' and r['scores']['status']=='extracted' and not r['scores']['content_correct']],'truncated':[r for r in pp if r['capped']]}
            for category,items in categories.items():
                for r in sorted(items,key=lambda r:hashlib.sha256(('case-audit:'+r['id']).encode()).hexdigest())[:2]:review.append({'category':category,'model':model,'task':task,'source':reference[r['id']],'prediction':r})
    for task in ('sw_reading','en_reading','en_math'):
        ids=[r['id'] for r in rr if r['task']==task];groups=[reference[i]['group_id'] for i in ids];assert len(set(groups))==64
        rng=np.random.default_rng(42);index=rng.integers(0,64,size=(10000,64));key='content_f1' if task=='sw_reading' else 'content_correct'
        for a,b in [('A','Base'),('B','A'),('C','B'),('D','B')]:
            delta=np.array([models[a][i]['scores'][key]-models[b][i]['scores'][key] for i in ids]);ci=np.quantile(delta[index].mean(1)*100,[.025,.975]);comparisons.append({'task':task,'comparison':a+'-'+b,'paired_difference_points':float(delta.mean()*100),'CI95_low':float(ci[0]),'CI95_high':float(ci[1]),'source_groups':64,'bootstrap':10000,'seed':42,'interpretation':'small fresh development diagnostic, not confirmatory formal test'})
    write(REPORT/'V2_SCORES.json',table);write(REPORT/'V2_PAIRED_COMPARISONS.json',comparisons)
    for filename,values in [('V2_SCORES.csv',table),('V2_PAIRED_COMPARISONS.csv',comparisons)]:
        with (REPORT/filename).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    write(RUN/'CASE_REVIEW_QUEUE.json',review)
    shutil.copyfile(ROOT/'outputs/ALL_ENDPOINT_SCORES.csv',REPORT/'V1_AND_POSTHOC_UNCHANGED.csv')
    write(REPORT/'GENERATION_COSTS.json',{'methods':{m:json.loads((RUN/'generation'/m/'DONE.json').read_text()) for m in models},'new_answers':960,'all_models_complete_same192_ids':True,'scorer_frozen_before_first_generation':json.loads((RUN/'ACCEPTANCE_FROZEN.json').read_text())['frozen_unix']<min(json.loads((RUN/'generation'/m/'CONFIG.json').read_text())['started_unix'] for m in models),'v1_and_POSTHOC_are_separate_from_v2':True})
    print(json.dumps({'table':table,'comparisons':comparisons},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
