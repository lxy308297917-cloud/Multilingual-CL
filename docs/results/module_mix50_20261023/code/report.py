from settings import *
import csv

def main():
 assert json.loads((REPORT/'COMPLETION_VERIFIED.json').read_text())['passed'];decision=json.loads((REPORT/'MANDATORY_DECISION.json').read_text());protocol=json.loads((REPORT/'PROTOCOL_FROZEN.json').read_text());short={'en_to_sw':'EN→SW','sw_to_en':'SW→EN','sw_reading':'SW阅读F1','en_reading':'EN阅读%','en_instruction':'EN指令strict%'};lines=['# 模块预算分配：S-Mix50与两种训练顺序','', '**状态：四份4293步短训、提示诊断、可用新开发任务及旧开发追踪完成；原始状态与结果核验通过。**','', '主要候选将每步77,070,336个元素名额平均分给Attention与MLP，分别38,535,168；其余配置与同顺序LR077一致。Attention-only沿用原5e-5配方，和Mix50比较同时涉及学习率与配额，不能解释为仅改变模块位置。','', '**新翻译确认集缺口：**按共同原文档排除历史Igbo/SW评测及训练侧曝光后，仅剩1个未使用FLORES开发文档，不足128。没有使用正式测试或替换题源；新确认仅覆盖SW阅读64、EN阅读64、英语指令代理128。原定EN→SW两顺序升级标准不可验收，结论须标为证据不足。','', '**预定决定：** '+decision['decision'],'', '## 分开的开发结果','']
 for split in ('old','confirmation'):
  r=json.loads((REPORT/(split.upper()+'_RESULTS.json')).read_text());lines+=['### '+('旧开发追踪' if split=='old' else '新来源隔离开发（缺少翻译）'),'','|模型|EN→SW chrF++|SW→EN chrF++|SW阅读F1|EN阅读%|EN指令strict%|','|---|---:|---:|---:|---:|---:|']
  for n,ts in r['models'].items():lines.append('|'+n+'|'+ '|'.join(f"{ts[t]['score']:.3f}" if t in ts else '缺口' for t in short)+'|')
  lines+=['','|同顺序比较|任务|差值|来源配对95%区间|','|---|---|---:|---|']
  for comp,ts in r['comparisons'].items():
   for t,v in ts.items():lines.append(f'|{comp}|{short[t]}|{v["difference"]:+.3f}|[{v["low"]:+.3f},{v["high"]:+.3f}]|')
  lines+=['','两顺序平均：按同一批来源同步重采样，再平均两套固定模型的差值；未将样本量翻倍。','', '|任务|两顺序平均Mix50−LR077|配对95%区间|','|---|---:|---|']
  for t,v in r['mean_order_comparisons'].items():lines.append(f'|{short[t]}|{v["difference"]:+.3f}|[{v["low"]:+.3f},{v["high"]:+.3f}]|')
  lines+=['','相对Base的英语损伤另列在'+split.upper()+'_RESULTS.json的English_vs_Base字段；超过LR077不代表保持解决。','']
 prompt=json.loads((REPORT/'PROMPT_SENSITIVITY.json').read_text());lines+=['## 单一加强提示诊断','', '从原128题以random.Random(42)抽取64题；四模型共256条新回答，旧提示复用对应缓存。原提示语言名称、chat template、源文本边界及截断检查未发现确定实现错误。不按模型输出筛子集，不更换正式主提示。','', '|模型|原chrF++|新chrF++|原/新检测SW|原/新截断|','|---|---:|---:|---:|---:|']
 for n,v in prompt['models'].items():
  a=v['old']['en_to_sw'];b=v['new']['en_to_sw'];lines.append(f'|{n}|{a["score"]:.3f}|{b["score"]:.3f}|{a["languages"].get("sw",0)}/{b["languages"].get("sw",0)}|{a["capped"]}/{b["capped"]}|')
 v=prompt['gap_change_new_minus_old'];lines+=['',f'Attention−LR077差距的提示变化（新−旧）为{v["difference"]:+.3f}，来源配对95%区间[{v["low"]:+.3f},{v["high"]:+.3f}]。它是提示敏感性诊断，不是语言控制与内容质量的完整因果分解。所有语言、重复、截断指标保存在PROMPT_SENSITIVITY.json。','', '## 实际更新、数值与成本','', '|模型|累计实际BF16路径|最终Base净位移L2|非零写回事件|物理显存峰值MiB|','|---|---:|---:|---:|---:|']
 audit=json.loads((REPORT/'COMPLETION_VERIFIED.json').read_text())
 for n,v in audit['new_scientific_runs'].items():lines.append(f'|{n}|{v["cumulative_path_length"]:.6f}|{v["final_base_delta_L2"]:.6f}|{v["nonzero_write_events"]}|{v["peak_physical_mib"]}|')
 lines+=['',f'整轮持卡上下文成本{audit["total_GPU_context_hours"]:.4f} GPU小时，上限16，包含预检、加载、保存、失败尝试与全部新评测。逐阶段成本与输入/监督/生成Token在COMPLETION_VERIFIED.json及运行日志。','', '每份均有2146/4293完整checkpoint，权重、FP32 Adam动量、scheduler、RNG和配额可恢复。每配方最长真实batch预检及4→8保存恢复逐位核验；正式每步LR、K、模块名额、顺序与Token核验。优化器继承原实现且哈希一致。','', '数据：Aya SW80%＋双向翻译各10%的同一冻结前缀，每份1,536,351主输入/599,469监督Token。seed43只打乱原4293个两样本更新块；同seed各方法完全同序。两种训练顺序不代表训练随机性的总体分布。','', '检查点统一入口：`/srv/multilingual/models/checkpoint/module_mix50_20261023`，实体在`/srv/multilingual/checkpoint-storage/module_mix50_20261023`；日志和TensorBoard：`/srv/multilingual/logs/module_mix50_20261023`。','', '## 解释边界与停止条件','', '本轮仅检验简单模块配额分配，未建立算法创新性；没有强行匹配实际更新路径，不能声称收益独立于更新强度。旧新开发分表，不合并扩大样本量。数学、正式测试、LoRA、表示对齐、子空间、首尾层、第二语言均未运行。','', '新翻译确认缺失意味着本轮不能完整判定原升级标准；无论旧题得分如何，都不自动追加比例、学习率、seed或训练长度。来源ID与词面近重复检查不能证明完全语义去污染；阅读/指令为训练侧代理，不是正式Belebele/IFEval。','', '公开保留代码、配置、聚合与核验，题文、逐条回答及权重留服务器。私有逐题比较为work/module_mix50_v1/PAIRED_ITEMS.jsonl。','']
 (REPORT/'README_ZH.md').write_text('\n'.join(lines))
 print('MIX50 CHINESE REPORT COMPLETE',flush=True)
if __name__=='__main__':main()
