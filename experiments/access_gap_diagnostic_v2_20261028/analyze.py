from settings import *
from parser import parse,CASES
import csv,collections,re,numpy as np
def savecsv(path,rows):
 with Path(path).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def ci(diff,groups):
 keys=list(dict.fromkeys(groups));index=np.array([keys.index(g) for g in groups]);s=np.bincount(index,weights=diff,minlength=len(keys));n=np.bincount(index,minlength=len(keys))
 rng=np.random.default_rng(42);w=rng.multinomial(len(keys),[1/len(keys)]*len(keys),size=10000);v=(w@s)/(w@n)
 return [float(x)*100 for x in np.quantile(v,[.025,.975])]
proto=read(REPORT/'PROTOCOL_FROZEN.json')
for p,digest in proto['v1_source_hashes'].items():assert sha(p)==digest
format_rows=[];format_summary=[];old_posthoc={(r['model'],r['id'],r['condition']):r for r in read(OLD_RUN/'POSTHOC_PARSED_PRIVATE.json')}
for model in MODELS:
 rows=[json.loads(s) for s in (OLD_RUN/'generation'/model/'PREDICTIONS.jsonl').read_text().splitlines()]
 assert len(rows)==640
 for r in rows:
  value=parse(r['text']);explanation=value.pop('explanation');language='und';confidence=None
  if len(re.findall(r'[^\W\d_]+',explanation))>=5:
   try:
    from langdetect import detect_langs,DetectorFactory
    DetectorFactory.seed=0;guess=detect_langs(explanation)[0];confidence=guess.prob
    if confidence>=.7:language=guess.lang
   except Exception:pass
  expected='en' if r['condition'] in 'AC' else 'sw'
  format_rows.append(dict(model=model,id=r['id'],task=r['task'],condition=r['condition'],gold=r['gold'],
   answer=value['answer'],content_correct=None if value['unparsed'] else value['answer']==r['gold'],
   frozen_v1_correct=r['scores']['correct'],v1_posthoc_correct=old_posthoc[model,r['id'],r['condition']]['correct'],
   original_format_compliant=r['scores']['format_compliant'],strict_independent_Final_line=value['strict_independent_Final_line'],
   strict_Final_last_line=value['strict_Final_last_line'],unparsed=value['unparsed'],ambiguous=value['ambiguous'],conflicting=value['conflicting'],
   explanation_language_heuristic=language,expected_explanation_language=expected,
   language_compliant_heuristic=None if language=='und' else language==expected,language_confidence=confidence,
   source_text_sha256=hashlib.sha256(r['text'].encode()).hexdigest()))
 for task in ('belebele','knowledge'):
  for condition in 'ABCDE':
   here=[r for r in format_rows if r['model']==model and r['task']==task and r['condition']==condition];assert len(here)==64
   format_summary.append(dict(model=model,task=task,condition=condition,n=64,
    explicit_correct=sum(r['content_correct'] is True for r in here),explicit_incorrect=sum(r['content_correct'] is False for r in here),
    unresolved=sum(r['unparsed'] for r in here),frozen_v1_correct=sum(r['frozen_v1_correct'] for r in here),
    v1_posthoc_correct=sum(r['v1_posthoc_correct'] for r in here),strict_Final_last_line=sum(r['strict_Final_last_line'] for r in here),
    ambiguous_or_conflicting=sum(r['ambiguous'] or r['conflicting'] for r in here),
    explanation_language_compliant_heuristic=sum(r['language_compliant_heuristic'] is True for r in here),
    explanation_language_unknown=sum(r['language_compliant_heuristic'] is None for r in here)))
savecsv(REPORT/'FORMAT_REANALYSIS.csv',format_rows);savecsv(REPORT/'FORMAT_SUMMARY.csv',format_summary)
table=[];comparisons={};private=[]
for model in MODELS:
 out=RUN/'MCF'/model;done=read(out/'DONE.json');assert done['passed'] and done['n']==1280
 assert sha(out/'PREDICTIONS.jsonl')==done['predictions_sha256']
 rows=[json.loads(s) for s in (out/'PREDICTIONS.jsonl').read_text().splitlines()];assert len(rows)==1280
 keyed={(r['id'],r['condition'],r['permutation']):r for r in rows};assert len(keyed)==1280
 for task in ('belebele','knowledge'):
  for permutation in ('original','cyclic_control'):
   conditions=proto['primary_conditions']+(proto['auxiliary_conditions'] if permutation=='original' else [])
   for condition in conditions:
    here=[r for r in rows if r['task']==task and r['condition']==condition and r['permutation']==permutation];assert len(here)==64
    labels=collections.Counter(r['predicted'] for r in here)
    table.append(dict(model=model,task=task,condition=condition,permutation=permutation,n=64,correct=sum(r['correct'] for r in here),
     ACC_percent=100*sum(r['correct'] for r in here)/64,predicted_A=labels['A'],predicted_B=labels['B'],predicted_C=labels['C'],predicted_D=labels['D'],
     prompt_tokens=sum(r['input_tokens'] for r in here)))
   ids=sorted({r['id'] for r in rows if r['task']==task})
   for c1,c0 in [('SW_EN','SW_only'),('SW_Control','SW_only'),('SW_EN','SW_Control'),('EN_only','SW_only')]:
    rr=[(keyed[q,c1,permutation],keyed[q,c0,permutation]) for q in ids]
    diff=np.array([int(x['correct'])-int(y['correct']) for x,y in rr]);groups=[x['group'] for x,y in rr]
    comparisons[f'{model}/{task}/{permutation}/{c1}-{c0}']=dict(n=64,difference_pp=float(diff.mean()*100),paired95CI_pp=ci(diff,groups),
     wrong_to_correct=sum(not y['correct'] and x['correct'] for x,y in rr),correct_to_wrong=sum(y['correct'] and not x['correct'] for x,y in rr))
    private.extend(dict(model=model,task=task,id=x['id'],permutation=permutation,comparison=c1+'-'+c0,first_correct=x['correct'],second_correct=y['correct']) for x,y in rr)
savecsv(REPORT/'MCF_RESULTS.csv',table)
write(REPORT/'MCF_PAIRED_TRANSITIONS.json',dict(comparisons=comparisons,bootstrap=proto['bootstrap'],exploratory=True))
write(RUN/'PAIRED_TRANSITIONS_PRIVATE.json',private)
controls=read(RUN/'CONTROLS_PRIVATE.json');d=np.array([r['token_difference'] for r in controls]);relative=np.array([abs(r['token_difference'])/r['english_tokens'] for r in controls])
write(REPORT/'CONTROL_LENGTH_AUDIT.json',dict(n=len(controls),min_difference=int(d.min()),max_difference=int(d.max()),median_absolute_difference=float(np.median(abs(d))),
 p95_absolute_difference=float(np.quantile(abs(d),.95)),max_relative_difference=float(relative.max()),not_strict_causal_control=True))
costs=[]
for p in [p for root in (RUN.parent,RUN) for p in root.glob('GPU_START_*.json')]:
 s=read(p);end=p.parent/p.name.replace('GPU_START_','GPU_END_');assert end.exists();e=read(end)
 costs.append(dict(label=str(p.relative_to(RUN.parent)),uuid=s['uuid'],seconds=e['wall_seconds'],success=e['success'],physical_peak_MiB=e['observed_physical_peak_MiB']))
write(REPORT/'COSTS.json',dict(contexts=costs,total_GPU_hours=sum(r['seconds'] for r in costs)/3600,new_MCF_decisions=3840,new_generation_answers=0,no_training=True))
(REPORT/'ANSWER_PARSER_TESTS.md').write_text('# 明确选项解析检查\n\n23个受控案例全部通过；规则在v1失败暴露后制定，旧1920条结果仅作为回顾性分析。无法解析单列，不认证内容错误；语言检测仅为启发式。\n\n'+
 '\n'.join('- '+repr(t)+' → '+str(a) for t,a in CASES)+'\n')
write(REPORT/'SOURCE_AND_IDENTITY.json',dict(protocol_sha256=sha(REPORT/'PROTOCOL_FROZEN.json'),models=read(RUN/'MODELS_FROZEN.json'),
 v1_hashes=proto['v1_source_hashes'],v2_prompt_sha256=sha(RUN/'PROMPTS_PRIVATE.json'),control_sha256=sha(RUN/'CONTROLS_PRIVATE.json'),
 raw_MCF={m:dict(path=str(RUN/'MCF'/m/'PREDICTIONS.jsonl'),sha256=sha(RUN/'MCF'/m/'PREDICTIONS.jsonl')) for m in MODELS},
 code_hashes={p.name:sha(p) for p in (REPORT/'code').glob('*.py')},no_model_weights_changed=True))
lines=['# 跨语言能力访问诊断v2','',
 '本轮仅评测已有三个快照，无训练、无新生成。主要指标为统一四标签MCF；旧1920条生成结果的解析单独列为回顾性分析。题目已暴露，不是独立确认。',
 '','历史保护与LR077的配方和训练预算不同，不能作为匹配算法排名；英语辅助条件不是无辅助SW能力成绩。',
 '','## 主MCF结果','', '|模型|任务|EN-only|SW-only|SW+EN|SW+Control|','|---|---|---:|---:|---:|---:|']
for m in MODELS:
 for task in ('belebele','knowledge'):
  vals={r['condition']:r for r in table if r['model']==m and r['task']==task and r['permutation']=='original'}
  lines.append('|'+m+'|'+task+'|'+'|'.join(f"{vals[c]['correct']}/64 ({vals[c]['ACC_percent']:.2f}%)" for c in proto['primary_conditions'])+'|')
lines+=['','## 正确英语与无关英语控制的差异','']
for m in MODELS:
 for task in ('belebele','knowledge'):
  c=comparisons[f'{m}/{task}/original/SW_EN-SW_Control'];e=comparisons[f'{m}/{task}/original/SW_EN-SW_only']
  lines.append(f"- {m}/{task}：E−B {e['difference_pp']:+.2f}pp；E−F {c['difference_pp']:+.2f}pp，来源配对95%区间{c['paired95CI_pp']}。")
lines+=['','## 原生成评分差异','',
 '|模型|任务|条件|v1冻结正确数|v1事后正确数|v2明确正确数|v2未解析|','|---|---|---|---:|---:|---:|---:|']
for r in format_summary:lines.append('|'+ '|'.join(str(r[k]) for k in ('model','task','condition','frozen_v1_correct','v1_posthoc_correct','explicit_correct','unresolved'))+'|')
lines+=['','## 限制与判断','',
 '对照文本来自独立训练侧阅读池，未给出参考答案，按长度匹配并排除连续16词重叠、限制内容词Jaccard；未做母语语义隔离认证，可能存在干扰。E−F不能直接称为内部通路的因果效应。',
 '','C/D仅为提示要求诊断，MCF没有生成解释，不能据此认证实际输出语言。单次正确不认证掌握。选项固定主结果与预定循环排列控制分开提供，不能按结果挑排列。',
 '','生成内容判定保留“明确正确、明确错误、未解析”三类；严格Final格式、是否末行与解释语言单列。语言检测对短文本和混合语言不可靠。',
 '','尚未训练能力访问修复算法。是否值得设计训练侧选择性监督，应结合E−B、F−B、E−F在阅读/知识及排列控制下的一致性讨论，不自动追加实验。',
 '',f"GPU上下文成本：{sum(r['seconds'] for r in costs)/3600:.4f} GPU小时，仅作记录，无停止阈值。",
 '','逐题原始MCF位于 SOURCE_AND_IDENTITY.json 所列私有路径；含题目与回答的材料不公开发布。']
(REPORT/'README_ZH.md').write_text('\n'.join(lines)+'\n')
write(REPORT/'COMPLETION_VERIFIED.json',dict(passed=True,MCF_rows=3840,old_generation_rows=1920,no_training=True,new_generation_answers=0,
 model_count=3,task_pairs=128,finished_unix=time.time()))
print('v2 completed:3840 newMCF +1920 retrospective generation rows, all denominators checked')
