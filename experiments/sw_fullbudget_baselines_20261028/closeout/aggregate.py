"""Endpoint-only results. Never chooses checkpoints or changes a scorer."""
import csv,json,collections,time
from result_io import *

MODELS=('Base',)+METHODS
OUT=REPORT/'results'

def collect_costs():
 rows=[];known_seconds=0.;missing=[]
 for start in sorted(RUN.glob('GPU_START_*.json')):
  a=read(start);end=start.with_name(start.name.replace('GPU_START_','GPU_END_',1))
  row=dict(label=start.stem.removeprefix('GPU_START_'),uuid=a['uuid'],pid=a['pid'],started_unix=a['started_unix'])
  if end.exists():
   b=read(end);elapsed=b['finished_unix']-a['started_unix'];assert elapsed>=0
   known_seconds+=elapsed;row.update(b);row['GPU_seconds']=elapsed
  else:
   missing.append(str(start));row['cost_unknown']=True
  rows.append(row)
 return dict(contexts=rows,known_GPU_hours=known_seconds/3600,unclosed_contexts=missing,
  includes_preflight_and_failed_contexts=True,waiting_for_GPU_excluded=True,
  GPU_context_wall_time_not_kernel_time=True,missing_exit_times_not_imputed=True,
  no_time_limit_or_stop_threshold=True)

def corpus_chrf_interval(left,right,groups,repeats=10000):
 from sacrebleu.metrics import CHRF
 metric=CHRF(char_order=6,word_order=2,beta=2)
 keys=list(dict.fromkeys(groups));index={k:i for i,k in enumerate(keys)}
 a=np.zeros((len(keys),len(left[0])));b=np.zeros_like(a)
 for x,y,g in zip(left,right,groups):a[index[g]]+=x;b[index[g]]+=y
 score=lambda x:metric._compute_score_from_stats(x.tolist()).score
 rng=np.random.default_rng(42);samples=[]
 for _ in range(repeats):
  ix=rng.integers(0,len(keys),len(keys));samples.append(score(a[ix].sum(0))-score(b[ix].sum(0)))
 return dict(difference_chrFpp=score(a.sum(0))-score(b.sum(0)),paired95CI=np.quantile(samples,[.025,.975]).tolist(),
  source_groups=len(keys),bootstrap_replicates=repeats,training_seed_uncertainty=False,corpus_statistic=True)

def instruction_interval(a,b,repeats=10000):
 def parts(rows):return np.array([[sum(r['scores']['instruction_pass']),len(r['scores']['instruction_pass'])] for r in rows],dtype=float)
 x,y=parts(a),parts(b);assert np.array_equal(x[:,1],y[:,1]);rng=np.random.default_rng(42);samples=[]
 for _ in range(repeats):
  ix=rng.integers(0,len(a),len(a));samples.append(100*(x[ix,0].sum()/x[ix,1].sum()-y[ix,0].sum()/y[ix,1].sum()))
 return dict(difference_pp=100*(x[:,0].sum()/x[:,1].sum()-y[:,0].sum()/y[:,1].sum()),
  paired95CI_pp=np.quantile(samples,[.025,.975]).tolist(),resampling_unit='prompt',bootstrap_replicates=repeats,training_seed_uncertainty=False)

def aggregate():
 OUT.mkdir(exist_ok=True);all_rows={};table=[];behavior={};manifest={}
 for model in MODELS:
  row=dict(method=model,main_input_tokens=0 if model=='Base' else 15360935)
  if model!='Base':
   training=read(REPORT/'training_acceptance'/f'{model}.json');assert training['passed']
   row.update(selected_min=training['selected_min'],selected_max=training['selected_max'],
    candidate_denominator=training['candidate_denominator'],trainable_elements=training['trainable_elements'],
    actual_nonzero_write_events=training['actual_nonzero_write_events'],
    final_net_displacement=training['final_net_displacement'],cumulative_actual_path=training['cumulative_actual_path'])
   expected_model_sha=training['final_weights_sha256']
  else:expected_model_sha=sha(BASE/'model.safetensors')
  for task in TASKS:
   output=evaluation_output(model,'formal',task);receipt=read(output/'DONE.json')
   assert receipt['model_sha256']==expected_model_sha,(model,task,'model identity mismatch')
   for path,digest in receipt['files'].items():assert sha(output/path)==digest,(model,task,path)
   if task.startswith(('belebele','knowledge')):
    records,scores=read_mcf(model,task);row[task+'_ACC']=scores['ACC_percent']
    if task.startswith('knowledge'):row[task+'_subject_macro_ACC']=scores['macro_source_ACC_percent']
   else:
    records,scores=read_generation(model,task)
    if task=='translation':
     for direction in ('en_to_sw','sw_to_en'):row[direction+'_chrFpp']=scores[direction]['corpus_chrfpp']
    elif task=='ifeval':
     row['IFEval_prompt_strict']=scores['prompt_strict_ACC'];row['IFEval_instruction_strict']=scores['instruction_strict_ACC']
    else:row['GSM8K_strict_ACC']=scores['strict_ACC'];row['GSM8K_flexible_ACC']=scores['flexible_ACC']
   all_rows[model,task]=records
   manifest[model+'/'+task]=dict(path=str(output),DONE_sha256=sha(output/'DONE.json'),model_sha256=receipt['model_sha256'])
   if task in ('translation','ifeval'):
    behavior[model+'/'+task]=dict(n=len(records),capped=sum(r['capped'] for r in records.values()),
     repeated_4grams=sum(r['repeated_4grams'] for r in records.values()),
     generated_tokens=sum(r['generated_tokens'] for r in records.values()),
     output_language=dict(collections.Counter(r['output_language'] for r in records.values())))
  table.append(row)
 # Paired prompt identities must be exactly equal across models, not merely counts.
 comparisons={};states={}
 for model in METHODS:
  for reference in ('Base','Dense','Element'):
   if model==reference:continue
   for task in TASKS:
    a,b=all_rows[model,task],all_rows[reference,task];assert set(a)==set(b)
    ids=sorted(a);x=[a[i] for i in ids];y=[b[i] for i in ids];label=f'{model}-{reference}/{task}'
    if task.startswith(('belebele','knowledge')):
     assert all(p['input_sha256']==q['input_sha256'] and p['gold']==q['gold'] for p,q in zip(x,y))
     comparisons[label]=paired_interval([p['correct'] for p in x],[p['correct'] for p in y],[p['group'] for p in x])
    elif task=='translation':
     assert all(p['prompt_token_ids']==q['prompt_token_ids'] for p,q in zip(x,y))
     for direction in ('en_to_sw','sw_to_en'):
      pairs=[(p,q) for p,q in zip(x,y) if p['task']==direction]
      comparisons[label+'/'+direction]=corpus_chrf_interval([p['scores']['chrfpp_stats'] for p,q in pairs],
       [q['scores']['chrfpp_stats'] for p,q in pairs],[p['group_id'] for p,q in pairs])
    elif task=='ifeval':
     assert all(p['prompt_token_ids']==q['prompt_token_ids'] for p,q in zip(x,y))
     comparisons[label+'/prompt_strict']=paired_interval([p['scores']['prompt_strict'] for p in x],
      [p['scores']['prompt_strict'] for p in y],ids)
     comparisons[label+'/instruction_strict']=instruction_interval(x,y)
    else:
     for filter_name in ('strict-match','flexible-extract'):
      assert all(p[filter_name]['doc']==q[filter_name]['doc'] for p,q in zip(x,y))
      comparisons[label+'/'+filter_name]=paired_interval([p[filter_name]['exact_match'] for p in x],
       [p[filter_name]['exact_match'] for p in y],ids)
 for model in MODELS:
  for family in ('belebele','knowledge'):
   en,sw=all_rows[model,family+'_en'],all_rows[model,family+'_sw'];assert set(en)==set(sw)
   base_en,base_sw=all_rows['Base',family+'_en'],all_rows['Base',family+'_sw']
   count=collections.Counter();transitions=collections.Counter()
   for i in en:
    new=f"EN{en[i]['correct']}_SW{sw[i]['correct']}";old=f"EN{base_en[i]['correct']}_SW{base_sw[i]['correct']}"
    count[new]+=1;transitions[old+' -> '+new]+=1
   states[model+'/'+family]=dict(n=len(en),states=dict(count),transitions_from_Base=dict(transitions))
 fields=list(dict.fromkeys(k for row in table for k in row))
 with (OUT/'MAIN_TABLE.csv').open('w') as stream:
  writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(table)
 write(OUT/'PAIRED_COMPARISONS.json',comparisons);write(OUT/'CAPABILITY_STATES.json',states)
 write(OUT/'GENERATION_BEHAVIOR.json',behavior);write(OUT/'RAW_MANIFEST.json',manifest)
 write(OUT/'COSTS.json',collect_costs())
 labels={'method':'方法','main_input_tokens':'主输入Token','belebele_sw_ACC':'SW阅读ACC',
  'knowledge_sw_ACC':'SW知识微平均ACC','en_to_sw_chrFpp':'EN→SW chrF++','sw_to_en_chrFpp':'SW→EN chrF++',
  'belebele_en_ACC':'EN阅读ACC','knowledge_en_ACC':'EN知识微平均ACC',
  'IFEval_prompt_strict':'IFEval提示严格','IFEval_instruction_strict':'IFEval指令严格','GSM8K_strict_ACC':'GSM8K严格ACC'}
 lines=['# 完整预算SW适配统一端点结果','',
  '六组均使用相同15,360,935主输入Token、5,984,371监督Token和42,465次更新；Base仅评测。单个训练顺序，不根据测试成绩选择checkpoint。','',
  '| '+' | '.join(labels.values())+' |','| '+' | '.join(['---']*len(labels))+' |']
 for row in table:
  values=[str(row[k]) if k in ('method','main_input_tokens') else f'{row[k]:.3f}' for k in labels]
  lines.append('| '+' | '.join(values)+' |')
 lines.extend(['','阅读为Belebele四选一准确率，非KenSwQuAD词面F1。知识题的微平均与57学科宏平均均保存在CSV中。',
  '样本数：EN/SW Belebele各900题；EN MMLU与SW Global-MMLU各14,042题；翻译各方向1,012题；IFEval 541提示/834指令；GSM8K 1,319题。','',
  'HFT15、SSU、元素稀疏和保护版每步名额均为196,608,000，占196个候选矩阵的15.0060024%。HFT15为历史整矩阵预算匹配版本，非原论文50%配方。MoFO阈值并列及全候选weight decay另列，不能声称所有方法实际写回量相同。','',
  'PAIRED_COMPARISONS.json保留相对Base、Dense和无保护元素稀疏的差值与来源配对区间；这些区间不代表跨训练种子稳定性。翻译区间重算语料级chrF++，不平均句级chrF。',
  'GSM8K严格与宽松提取分开报告；格式提取失败不必然代表内容错误。历史暴露benchmark属于补充比较，不包装为未见最终确认。',
  'RAW_MANIFEST.json指向保留的逐题原始结果；COSTS.json包含预检和失败运行，缺失退出时间明确保留为不确定性。','',
  '**该表由验收脚本生成；是否值得继续研究仍需结合配对区间、输出行为和人工错误复核判断。**'])
 (OUT/'README_ZH.md').write_text('\n'.join(lines)+'\n')
 write(OUT/'ACCEPTANCE.json',dict(passed=True,models=list(MODELS),same_prompt_identities_verified=True,
  endpoint_only=True,MCF_records=209188,unique_generated_responses=27188,completed_unix=time.time(),
  limitations=['single training order; intervals reflect sampled questions/sources, not training-seed population',
   'exposed benchmark material is supplementary evaluation, not unseen final confirmation',
   'strict GSM8K includes required answer-format compliance; not all strict failures are content errors']))
 return table

if __name__=='__main__':print(json.dumps(aggregate(),ensure_ascii=False,indent=2))
