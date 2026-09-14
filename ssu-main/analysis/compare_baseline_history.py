"""Decompose historical/current aggregate differences using verified published task scores."""
import csv,json,io
from pathlib import Path
R=Path(__file__).resolve().parents[1];S=R.parent/'experiments/ig_baseline_suite_v1'
keys=['belebele_en','sum_en','mt_ig2en','sum_ig','belebele_ig','mt_en2ig']
history={m:dict(zip(keys,v)) for m,v in {'base':[82,21.87,17.43,17.12,30.22,11.86],'legacy_full_fft':[78.78,20.71,16.28,24.42,30.67,17.11],'legacy_ssu_en_freeze50':[82.11,22.05,20.12,20.01,29.78,20.34]}.items()}
def generate():
 current={};source={}
 for row in csv.DictReader((S/'formal_scores_by_seed.csv').open()):
  if row['model'] in history and row['metric'] in keys:
   k=(row['model'],row['metric']);assert k not in source,'duplicate score'
   current.setdefault(row['model'],{})[row['metric']]=float(row['score']);source[k]=row['source']
 records=[]
 for m in history:
  for k in keys:
   old=history[m][k];new=current.get(m,{}).get(k);b=current.get('base',{}).get(k);hb=history['base'][k]
   oldrel=(old/hb-1)*100;newrel=None if new is None or b in (None,0) else (new/b-1)*100
   records.append({'model':m,'metric':k,'group':'EN3' if k in keys[:3] else 'Target3','historical_raw':old,'current_raw':new,'raw_difference':None if new is None else new-old,'historical_relative_percent':oldrel,'current_relative_percent':newrel,'aggregate_difference_contribution_pp':None if newrel is None else (newrel-oldrel)/3,'current_score_source':source.get((m,k),'')})
 f=io.StringIO();w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records);(S/'historical_current_task_differences.csv').write_text(f.getvalue())
 lines=['# 历史表与当前统一协议的逐任务差异','','历史数值来自用户截图，只有两位小数；当前数值来自通过DONE校验的逐任务汇总。当前评测保持原模型权重；分数差异不应解释为重新训练的效果。当前未完成项留空。','','EN3/Target3分别对三项相对同条件Base变化等权平均。单任务对综合差异的贡献=(当前相对变化−历史相对变化)/3，单位为百分点；它同时包含模型分数与Base分母变化。','','|模型|任务|历史分数|当前分数|原始分数差|综合差异贡献（百分点）|','|---|---|---:|---:|---:|---:|']
 def fmt(v):return '—' if v is None else f'{v:.4f}'
 for x in records:lines.append('|'+ '|'.join([x['model'],x['metric'],fmt(x['historical_raw']),fmt(x['current_raw']),fmt(x['raw_difference']),fmt(x['aggregate_difference_contribution_pp'])])+'|')
 lines+=['','已证实：当前摘要500条采用SSU选样规则及Qwen tokenizer，样本ID、顺序、正文和参考摘要均通过源数据重建比对。当前统一采样参数及seed42；历史表全部任务的有效生成设置未完全追溯，不能把全部差异直接归因于某一个参数。','GSM8K等通用任务不进入这两项综合；当前GSM8K为固定8-shot CoT。翻译数据已对固定版本FLORES-200源数据核验：测试500对及dev示例997对的英语/Igbo文本与顺序全部一致；测试取样为seed42随机500条。未声称与不可访问的当前FLORES+版本逐字相同。']
 (S/'HISTORICAL_CURRENT_DIFFERENCES_ZH.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':generate()
