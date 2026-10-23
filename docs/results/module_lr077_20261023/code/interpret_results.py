"""Five direct answers from complete frozen results; no new criteria or scores."""
from settings import *

def main():
 assert json.loads((REPORT/'COMPLETION_VERIFIED.json').read_text())['passed'];d=json.loads((REPORT/'DOWNSTREAM_COMPARISON.json').read_text());s=json.loads((REPORT/'UPDATE_STRENGTH.json').read_text());dec=json.loads((REPORT/'MANDATORY_DECISION.json').read_text());a=d['comparisons'][STUDY_NAME+'-minus-S-All'];b=d['comparisons']['S-Attn-minus-'+STUDY_NAME];tasks={'en_to_sw':'EN→SW翻译','sw_to_en':'SW→EN翻译','sw_reading':'SW阅读','en_reading':'英语阅读','en_instruction':'英语指令strict'}
 def stat(v):return f'{v["difference"]:+.3f}，来源配对95%区间[{v["low"]:+.3f},{v["high"]:+.3f}]'
 def evidence(v):return '区间支持提升' if v['low']>0 else '区间支持下降' if v['high']<0 else '区间包含0，差异不确定'
 langs={n:d['models'][n]['en_to_sw'] for n in ['S-All',STUDY_NAME,'S-Attn']};old_sw=langs['S-All']['languages'].get('sw',0);new_sw=langs[STUDY_NAME]['languages'].get('sw',0)
 remaining=[tasks[t] for t in tasks if b[t]['low']>0];lowerlr_positive=[tasks[t] for t in tasks if a[t]['low']>0]
 if not s['path_matched']:attribution='总体路径未匹配，更新强度混杂仍未排除；本轮无法把剩余差距纯归因于模块位置，也不追加学习率重训。'
 elif remaining and lowerlr_positive:attribution='降低学习率和模块选择均保留局部信号：降低LR在'+ '、'.join(lowerlr_positive)+'上有提升证据，Attention在'+ '、'.join(remaining)+'上仍更高。单一累计路径指标不能解释全部差距；其他位移与训练轨迹未完全匹配，不能确立纯模块因果效应。'
 elif remaining:attribution='累计路径接近时，Attention在'+ '、'.join(remaining)+'上仍有局部优势，与模块位置作用相容；但净位移、支持集合和各模块变化未完全匹配，不能据此确认纯位置效应。'
 elif lowerlr_positive:attribution='降低LR在'+ '、'.join(lowerlr_positive)+'上有提升证据，Attention相对新对照没有区间明确支持的优势；这支持更新强度能够解释部分旧差距，但不能断言模块位置无作用。'
 else:attribution='累计路径接近，但降低LR和剩余Attention差异均缺乏明确的局部优势证据；当前不足以确定哪种因素主导，不从不显著推断等效。'
 lines=['# S-All-LR077：五个问题的直接回答','', '本轮仅新增一份4293步训练，其余模型复用已验收缓存；单seed开发诊断，数学与正式测试未运行。','', '**1．实际更新强度接近S-Attn了吗？**','',f'{"按预定总路径标准接近" if s["path_matched"] else "没有达到预定总路径匹配标准"}。LR077累计路径{s["new_path_length"]:.6f}，S-Attn参考213.379786，比例{s["ratio_to_Attn_exact"]:.4f}，允许范围[192.0418074,234.7177646]。最终净位移比为{s["net_displacement_ratio_to_Attn"]:.4f}；不代表所有更新特征都匹配。','', '**2．降低LR修复了目标语输出或英语保持问题吗？**','', f'EN→SW相对原S-All：{stat(a["en_to_sw"])}，{evidence(a["en_to_sw"])}。检测为SW的回答由{old_sw}/128变为{new_sw}/128；新模型截断{langs[STUDY_NAME]["capped"]}/128。语言标签不等于翻译正确率。', '', f'英语阅读变化：{stat(a["en_reading"])}，{evidence(a["en_reading"])}；英语指令strict变化：{stat(a["en_instruction"])}，{evidence(a["en_instruction"])}。这些代理任务不能代表全面通用能力保持。','', '**3．Attention哪些优势仍然存在？**','', '|指标|S-Attn − LR077|解释|','|---|---|---|']
 for t,label in tasks.items():lines.append(f'|{label}|{stat(b[t])}|{evidence(b[t])}|')
 lines+=['','旧差距与新差距的点估计变化、全部冻结门槛及其他参考模型见[完整报告](README_ZH.md)。区间包含0表示证据不足，不表示优势已被证明消失。','', '**4．更支持模块位置还是更新强度？**','',attribution,'', '**5．值得直接进入子空间或扩大规模吗？**','', '累计路径接近且Attention联合门槛通过，可以讨论进一步模块验证，但仍需独立复测与更完整的混杂控制；当前不自动扩展。' if dec['conditional_module_signal'] else '当前不足以按预定规则直接进入子空间、首尾层或长训。本轮到此保留结果和不确定性，不自动追加学习率或seed。','', '行为案例固定取冻结题序最先3道EN→SW，不按新结果选题；完整回答仅留服务器。来源配对区间10000次bootstrap，seed42；不使用跨任务均分选冠军。']
 (REPORT/'INTERPRETATION_ZH.md').write_text('\n'.join(lines)+'\n');print('FIVE FROZEN QUESTIONS ANSWERED',flush=True)
if __name__=='__main__':main()
