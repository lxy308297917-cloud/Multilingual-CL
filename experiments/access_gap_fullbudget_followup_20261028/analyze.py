"""Matched full-budget v2 tables; the original three-model diagnostic is untouched."""
from pathlib import Path
import sys,json,hashlib,csv,collections,time
import numpy as np
ROOT=Path(__file__).resolve().parents[3];REPORT=ROOT/'outputs/access_gap_fullbudget_followup_20261028'
RUN=ROOT/'work/access_gap_fullbudget_followup_20261028'
sys.path.insert(0,str(ROOT/'outputs/sw_fullbudget_baselines_20261028/closeout'))
from result_io import paired_interval,read,write,sha

def summarize(sources,prompts,proto):
 expected={(r['id'],r['condition'],r['permutation']):r for r in prompts}
 assert len(expected)==1280
 table=[];comparisons={};states={};all_records={}
 for model,(path,digest) in sources.items():
  receipt=read(path/'DONE.json');assert receipt['passed'] and receipt['n']==1280 and receipt['model_sha256']==digest
  assert receipt['predictions_sha256']==sha(path/'PREDICTIONS.jsonl')
  records={}
  for line in (path/'PREDICTIONS.jsonl').open():
   r=json.loads(line);key=(r['id'],r['condition'],r['permutation']);assert key not in records
   assert r['model']==model and r['model_sha256']==digest
   prompt=expected[key];raw=np.array(r['raw_logprob']);prob=np.array(r['normalized_probability'])
   assert raw.shape==prob.shape==(4,) and np.isfinite(raw).all() and np.isfinite(prob).all()
   computed=np.exp(raw-raw.max());computed/=computed.sum();assert np.allclose(computed,prob,atol=1e-6,rtol=1e-5)
   assert r['predicted']=='ABCD'[int(raw.argmax())] and r['correct']==(r['predicted']==r['gold'])
   assert r['gold']==prompt['gold'] and r['candidate_ids']==prompt['candidate_ids']
   assert r['input_ids_sha256']==hashlib.sha256(json.dumps(prompt['input_ids']).encode()).hexdigest()
   assert r['prompt_sha256']==hashlib.sha256(prompt['prompt'].encode()).hexdigest()
   assert r['input_tokens']==len(prompt['input_ids'])<=proto['maximum_input_tokens']
   records[key]=r
  assert set(records)==set(expected);all_records[model]=records
  for task in ('belebele','knowledge'):
   for permutation in ('original','cyclic_control'):
    conditions=proto['primary_conditions']+(proto['auxiliary_conditions'] if permutation=='original' else [])
    for condition in conditions:
     here=[r for r in records.values() if r['task']==task and r['condition']==condition and r['permutation']==permutation]
     assert len(here)==64
     table.append(dict(model=model,task=task,condition=condition,permutation=permutation,n=64,
      correct=sum(r['correct'] for r in here),ACC_percent=100*sum(r['correct'] for r in here)/64))
    ids=sorted({r['id'] for r in records.values() if r['task']==task})
    for a,b in [('SW_EN','SW_only'),('SW_Control','SW_only'),('SW_EN','SW_Control'),('EN_only','SW_only')]:
     x=[records[i,a,permutation] for i in ids];y=[records[i,b,permutation] for i in ids]
     comparisons[f'{model}/{task}/{permutation}/{a}-{b}']=dict(
      paired_interval([r['correct'] for r in x],[r['correct'] for r in y],[r['group'] for r in x]),
      wrong_to_correct=sum(p['correct'] and not q['correct'] for p,q in zip(x,y)),
      correct_to_wrong=sum(q['correct'] and not p['correct'] for p,q in zip(x,y)))
   state=collections.Counter()
   for i in ids:state[f"EN{int(records[i,'EN_only','original']['correct'])}_SW{int(records[i,'SW_only','original']['correct'])}"]+=1
   states[model+'/'+task]=dict(state)
 for model,records in all_records.items():
  if model=='Base':continue
  for reference in ('Base','Element'):
   if model==reference or reference not in all_records:continue
   other=all_records[reference]
   for task in ('belebele','knowledge'):
    for condition in proto['primary_conditions']:
     x=[r for r in records.values() if r['task']==task and r['condition']==condition and r['permutation']=='original']
     y=[other[r['id'],condition,'original'] for r in x]
     comparisons[f'{model}-{reference}/{task}/{condition}']=paired_interval(
      [r['correct'] for r in x],[r['correct'] for r in y],[r['group'] for r in x])
 return table,comparisons,states

def main():
 scope=read(REPORT/'FOLLOWUP_PROTOCOL_FROZEN.json');proto=read(scope['original_protocol']);assert sha(scope['original_protocol'])==scope['original_protocol_sha256']
 prompts=read(RUN/'PROMPTS_PRIVATE.json');assert sha(RUN/'PROMPTS_PRIVATE.json')==scope['prompts_sha256']
 identities=read(RUN/'MODELS_FROZEN.json');base=Path(scope['reused_Base']);base_receipt=read(base/'DONE.json')
 sources={'Base':(base,base_receipt['model_sha256'])}
 sources.update({m:(RUN/'MCF'/m,identities[m]['sha256']) for m in scope['methods']})
 table,comparisons,states=summarize(sources,prompts,proto)
 with (REPORT/'MCF_RESULTS.csv').open('w') as stream:
  writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
 write(REPORT/'PAIRED_COMPARISONS.json',comparisons);write(REPORT/'CAPABILITY_STATES.json',states)
 write(REPORT/'RAW_MANIFEST.json',{m:dict(path=str(p/'PREDICTIONS.jsonl'),sha256=sha(p/'PREDICTIONS.jsonl'),model_sha256=d) for m,(p,d) in sources.items()})
 lines=['# 完整预算模型：冻结能力访问诊断v2复评','',
  '复用同一128道已暴露题目、英语语义控制、四标签MCF及预定循环选项排列。六个新模型为匹配完整预算终点；Base复用通过验收的原v2结果。无训练、无新解释生成，不代替全量benchmark主表。','',
  '|模型|任务|EN-only|SW-only|SW+EN|SW+Control|','|---|---|---:|---:|---:|---:|']
 for model in sources:
  for task in ('belebele','knowledge'):
   here={r['condition']:r for r in table if r['model']==model and r['task']==task and r['permutation']=='original'}
   lines.append('|'+model+'|'+task+'|'+'|'.join(f"{here[c]['correct']}/64" for c in proto['primary_conditions'])+'|')
 lines.extend(['','配对区间、循环排列及相对Base/无保护元素稀疏的差值分别保存。区间不代表训练seed总体；该128题诊断属于探索性补充。',
  '无关英语控制未完成母语语义认证，长度并非精确匹配，可能产生干扰；E−F不能认证内部因果通路。C/D仅为提示条件，不认证实际输出语言。',
  '旧1920条生成解析已单独交付，不在本轮重复生成或混入MCF成绩。'])
 (REPORT/'README_ZH.md').write_text('\n'.join(lines)+'\n')
 write(REPORT/'COMPLETION_VERIFIED.json',dict(passed=True,new_models=6,reused_Base_models=1,new_MCF_records=7680,reused_Base_records=1280,
  same_protocol_verified=True,no_new_generation=True,no_new_training=True,completed_unix=time.time()))

if __name__=='__main__':main()
