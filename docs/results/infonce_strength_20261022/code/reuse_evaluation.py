"""Verify exact cached model/input/generation identity; rescore frozen512subset, never alter cache."""
from evaluation_common import *

def main():
 meta,rr=inputs();proof={}
 for name,oldname in [('Base','Base'),('B','B'),('D0','D')]:
  old=TRAIN/'development'/oldname;v2=ROOT/'work/evaluation_audit_v2/generation'/oldname
  configs=[json.loads((p/'CONFIG.json').read_text()) for p in (old,v2)]
  path=BASE if name=='Base' else TRAIN/'comparison_v3'/oldname/'checkpoint_004293/model';mh=sha(path/'model.safetensors');assert all(c['model_sha256']==mh for c in configs)
  raw={};receipts={}
  for folder in (old,v2):
   done=json.loads((folder/'DONE.json').read_text());assert sha(folder/'PREDICTIONS.jsonl')==done['predictions_sha256'];receipts[str(folder)]={'prediction_sha256':done['predictions_sha256'],'config_sha256':sha(folder/'CONFIG.json'),'done_sha256':sha(folder/'DONE.json')};raw[str(folder)]={r['id']:r for r in rows(folder/'PREDICTIONS.jsonl')}
  result=[]
  for r in rr:
   folder=v2 if r['study_partition'] in ('sw_reading','en_reading') else old;record=raw[str(folder)][r['id']];assert record['group_id']==r['group_id'];s=scoring(r,record['text'])
   # Byte-selected reading scorer gives exactly previous accepted outputs.
   if folder==v2:assert s==record['scores'],(name,r['id'],s,record['scores'])
   else:assert s==record['scores']
   result.append({**record,'model':name,'study_partition':r['study_partition'],'score_eligible':r['score_eligible'],'scores':s,'behavior':behavior(record['text'])})
  target=RUN/'evaluation'/name;target.mkdir(parents=True,exist_ok=False);(target/'PREDICTIONS.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in result));write(target/'DONE.json',{'rows':512,'scored_rows':511,'reused':True,'model_sha256':mh,'data_sha256':meta['data_sha256'],'predictions_sha256':sha(target/'PREDICTIONS.jsonl'),'source_receipts':receipts,'new_generation_GPUh':0})
  proof[name]={'model_sha256':mh,'data_sha256':meta['data_sha256'],'source_receipts':receipts,'512_unique_predictions_rescored_unchanged':True}
 write(REPORT/'EVALUATION_REUSE_VERIFIED.json',{'models':proof,'limitations':'Reused exact historical prompt/order/batch/cap policy, preserved512subset. Deterministic CUDA policy and config verified by code and manifests; not new replay of every cached answer. No cross-protocol pooling within any task.'});print('Base/B/D0 cached evaluation verified')
if __name__=='__main__':main()
