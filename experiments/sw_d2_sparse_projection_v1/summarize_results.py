"""Read-only, verified SW D2 method comparison; no GPU task is launched."""
import csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
EXPERIMENTS=ROOT.parent
sys.path.insert(0,str(EXPERIMENTS/'reproduction'))
sys.path.insert(0,str(EXPERIMENTS.parent/'ssu-main/scripts/sw_recipes_v4'))
from verified_cache import verify
from report import cluster_comparison,reading_details

MODELS={
 'Base':'/root/models/Qwen2.5-1.5B-Instruct',
 'D2_历史FFT':'/root/autodl-tmp/d2_bilingual_v1/checkpoints/sw/final',
 **{m:str(Path('/root/autodl-tmp/sw_d2_sparse_projection_v1/checkpoints')/m/'final') for m in (
  'element_protected','element_unprotected','edge_projected_rows','edge_rows_no_projection','fft_fp32_matched')}
}
METRICS=[('belebele_sw','SW Belebele','%'),('sum_sw','SW XL-Sum','chrF++'),('mt_en2sw','EN→SW','chrF++'),
 ('mt_sw2en','SW→EN','chrF++'),('belebele_en','EN Belebele','%'),('sum_en','EN XL-Sum','chrF++'),
 ('ifeval_prompt_strict','IFEval prompt strict','%'),('ifeval_instruction_strict','IFEval instruction strict','%'),
 ('gsm8k_strict','GSM8K strict 5-shot','%'),('gsm8k_flexible','GSM8K flexible 5-shot','%')]
DIAG=['en_ppl','sw_ppl','en_nll','sw_nll']
PAIRS=[('element_protected','element_unprotected'),('edge_projected_rows','edge_rows_no_projection'),
 *[(m,'D2_历史FFT') for m in ('element_protected','element_unprotected','edge_projected_rows','edge_rows_no_projection','fft_fp32_matched')],
 ('D2_历史FFT','Base'),*[(m,'Base') for m in ('element_protected','element_unprotected','edge_projected_rows','edge_rows_no_projection','fft_fp32_matched')]]

def atomic(path,text):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(text);tmp.replace(path)

def checked(name,model):
 if name not in ('Base','D2_历史FFT'):
  out=Path(model).parent
  done=out/'TRAINING_DONE.json';evaldone=ROOT/'evaluation'/name/'DONE.json'
  if not done.exists():return None,'训练未完成'
  if not evaldone.exists():return None,'测评未完成'
  train=json.loads(done.read_text());ev=json.loads(evaldone.read_text())
  from hashlib import sha256
  assert ev['training_done_sha256']==sha256(done.read_bytes()).hexdigest()
  assert train['global_step']==22146 and train['input_tokens']==15360372
  assert ev['model']==str(Path(model).resolve())
  assert ev['config_sha256']==train['config_sha256']
 evidence=verify('sw',model)
 if not evidence['complete']:return None,'缺少已核验任务：'+','.join(evidence['missing'])
 scores={};details=None
 for unit in evidence['verified_units']:
  attempt=Path(unit['done']).parent
  x=json.loads((attempt/'scores.json').read_text())
  for key,value in x.items():
   assert key not in scores,(name,key)
   scores[key]=float(value)
  if unit['task']=='belebele_sw':details=reading_details(attempt)
 assert details and all(k in scores for k,_,_ in METRICS) and all(k in scores for k in DIAG)
 return {'scores':scores,'details':details,'evidence':evidence},None

def main():
 results={};missing={}
 for name,path in MODELS.items():
  value,why=checked(name,path)
  if why:missing[name]=why
  else:results[name]=value
 raw=[];base=results.get('Base',{}).get('scores',{});d2=results.get('D2_历史FFT',{}).get('scores',{})
 for name,res in results.items():
  for metric,label,unit in METRICS+[(k,k,'diagnostic') for k in DIAG]:
   score=res['scores'][metric];b=base.get(metric);h=d2.get(metric)
   raw.append({'model':name,'metric':metric,'benchmark':label,'unit':unit,'score':score,'base_score':b,
               'delta_base':score-b if b is not None else None,'relative_base_pct':100*(score/b-1) if b else None,
               'd2_score':h,'delta_d2':score-h if h is not None else None,'relative_d2_pct':100*(score/h-1) if h else None})
 with (ROOT/'raw_scores.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['model','metric','benchmark','unit','score','base_score','delta_base','relative_base_pct','d2_score','delta_d2','relative_d2_pct']);w.writeheader();w.writerows(raw)
 comparisons=[]
 for left,right in PAIRS:
  if left in results and right in results:
   result=cluster_comparison(results[left]['details'],results[right]['details'],10000,42)
   comparisons.append({'left':left,'right':right,**result})
 with (ROOT/'reading_pairs.csv').open('w',newline='') as f:
  fields=['left','right','delta_pp','ci95_low_pp','ci95_high_pp','one_sided_cluster_signflip_p','questions','passages','resamples','seed']
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for row in comparisons:w.writerow({k:({**row,'ci95_low_pp':row['ci95_pp'][0],'ci95_high_pp':row['ci95_pp'][1]})[k] for k in fields})
 aggregates={}
 for name,res in results.items():
  if not base:continue
  s=res['scores']
  aggregates[name]={k:sum(100*(s[x]/base[x]-1) for x in keys)/3 for k,keys in {
   'EN3':['belebele_en','sum_en','mt_sw2en'],'Target3':['belebele_sw','sum_sw','mt_en2sw']}.items()}
 costs={}
 for name in MODELS:
  if name in ('Base','D2_历史FFT'):continue
  p=Path('/root/autodl-tmp/sw_d2_sparse_projection_v1/checkpoints')/name/'TRAINING_DONE.json'
  if p.exists():costs[name]=json.loads(p.read_text())
 lines=['# SW D2 同数据方法对照','',
  '仅列出固定 seed42 任务中通过 DONE、产物哈希和模型身份核验的成绩。D2 旧 FFT 是历史参考；同优化器密集 FFT 与两组机制对照为本轨道严格对照。',
  'MMLU/G-MMLU 延期。PPL/NLL 仅作诊断，不进入 EN3/Target3 或能力结论。','',
  '待完成：'+json.dumps(missing,ensure_ascii=False),'','## 下游原始分数与相对 Base 变化','',
  '|方法|Benchmark|原始分数|相对 Base|相对历史 D2|','|---|---|---:|---:|---:|']
 for row in raw:
  if row['metric'] in DIAG:continue
  d='待 Base' if row['delta_base'] is None else f"{row['delta_base']:+.2f}"
  h='待 D2' if row['delta_d2'] is None else f"{row['delta_d2']:+.2f}"
  lines.append(f"|{row['model']}|{row['benchmark']}|{row['score']:.2f}|{d}|{h}|")
 lines+=['','## SW 阅读逐文章配对差值','','|比较|差值 pp|95% CI pp|单侧配对检验 p|','|---|---:|---:|---:|']
 for c in comparisons:lines.append(f"|{c['left']} − {c['right']}|{c['delta_pp']:+.2f}|[{c['ci95_pp'][0]:+.2f}, {c['ci95_pp'][1]:+.2f}]|{c['one_sided_cluster_signflip_p']:.4f}|")
 lines+=['','该区间按文章分组、配对重抽样10000次；只反映固定模型与固定测试集的抽样不确定性，不代表跨训练 seed 稳定性。',
         '','## EN3 与 Target3（相对 Base 百分比等权平均）','',
         '|方法|EN3|Target3|','|---|---:|---:|']
 for name,v in aggregates.items():lines.append(f"|{name}|{v['EN3']:+.2f}%|{v['Target3']:+.2f}%|")
 lines+=['','## PPL/NLL 诊断','','|方法|EN PPL|SW PPL|EN NLL|SW NLL|','|---|---:|---:|---:|---:|']
 for name,res in results.items():
  s=res['scores'];lines.append(f"|{name}|{s['en_ppl']:.3f}|{s['sw_ppl']:.3f}|{s['en_nll']:.3f}|{s['sw_nll']:.3f}|")
 lines+=['','## 训练成本与范围','','最终权重只在预算终点比较。当前完成模型的训练成本、输入与监督 token 记录于 training_costs.json。',
         '目标是判断 SW 阅读收益与 EN/通用保持之间的取舍；阅读 CI 为正也不能覆盖其他任务的下降。','']
 atomic(ROOT/'REPORT_ZH.md','\n'.join(lines))
 atomic(ROOT/'training_costs.json',json.dumps(costs,ensure_ascii=False,indent=2)+'\n')
 atomic(ROOT/'SUMMARY_STATUS.json',json.dumps({'missing':missing,'verified_models':list(results),'reading_pairs':comparisons,
  'aggregates':aggregates,'evidence':{k:v['evidence'] for k,v in results.items()}},ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'verified_models':list(results),'missing':missing,'reading_pairs':len(comparisons)},ensure_ascii=False))
if __name__=='__main__':main()
