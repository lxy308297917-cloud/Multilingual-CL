# 跨语言能力访问诊断v2

本轮仅评测已有三个快照，无训练、无新生成。主要指标为统一四标签MCF；旧1920条生成结果的解析单独列为回顾性分析。题目已暴露，不是独立确认。

历史保护与LR077的配方和训练预算不同，不能作为匹配算法排名；英语辅助条件不是无辅助SW能力成绩。

## 主MCF结果

|模型|任务|EN-only|SW-only|SW+EN|SW+Control|
|---|---|---:|---:|---:|---:|
|Base|belebele|51/64 (79.69%)|18/64 (28.12%)|45/64 (70.31%)|21/64 (32.81%)|
|Base|knowledge|34/64 (53.12%)|16/64 (25.00%)|23/64 (35.94%)|18/64 (28.12%)|
|LR077-s42|belebele|48/64 (75.00%)|20/64 (31.25%)|44/64 (68.75%)|18/64 (28.12%)|
|LR077-s42|knowledge|26/64 (40.62%)|17/64 (26.56%)|20/64 (31.25%)|19/64 (29.69%)|
|Historical-element_protected|belebele|48/64 (75.00%)|30/64 (46.88%)|44/64 (68.75%)|21/64 (32.81%)|
|Historical-element_protected|knowledge|28/64 (43.75%)|14/64 (21.88%)|20/64 (31.25%)|19/64 (29.69%)|

## 正确英语与无关英语控制的差异

- Base/belebele：E−B +42.19pp；E−F +37.50pp，来源配对95%区间[20.3125, 53.125]。
- Base/knowledge：E−B +10.94pp；E−F +7.81pp，来源配对95%区间[-5.084745762711865, 21.21212121212121]。
- LR077-s42/belebele：E−B +37.50pp；E−F +40.62pp，来源配对95%区间[25.0, 56.25]。
- LR077-s42/knowledge：E−B +4.69pp；E−F +1.56pp，来源配对95%区间[-10.9375, 14.754098360655737]。
- Historical-element_protected/belebele：E−B +21.88pp；E−F +35.94pp，来源配对95%区间[18.75, 51.5625]。
- Historical-element_protected/knowledge：E−B +9.38pp；E−F +1.56pp，来源配对95%区间[-10.44776119402985, 13.794429708222792]。

## 原生成评分差异

|模型|任务|条件|v1冻结正确数|v1事后正确数|v2明确正确数|v2未解析|
|---|---|---|---:|---:|---:|---:|
|Base|belebele|A|50|52|52|0|
|Base|belebele|B|13|16|16|6|
|Base|belebele|C|25|25|25|0|
|Base|belebele|D|42|49|49|3|
|Base|belebele|E|14|28|28|12|
|Base|knowledge|A|33|34|34|0|
|Base|knowledge|B|15|22|22|1|
|Base|knowledge|C|21|21|21|0|
|Base|knowledge|D|29|34|34|4|
|Base|knowledge|E|12|21|21|14|
|LR077-s42|belebele|A|17|49|49|1|
|LR077-s42|belebele|B|6|17|17|0|
|LR077-s42|belebele|C|7|17|17|6|
|LR077-s42|belebele|D|17|46|46|2|
|LR077-s42|belebele|E|9|44|44|4|
|LR077-s42|knowledge|A|17|29|29|0|
|LR077-s42|knowledge|B|4|14|14|5|
|LR077-s42|knowledge|C|5|12|12|7|
|LR077-s42|knowledge|D|12|28|28|0|
|LR077-s42|knowledge|E|13|24|24|3|
|Historical-element_protected|belebele|A|46|49|49|0|
|Historical-element_protected|belebele|B|19|30|30|0|
|Historical-element_protected|belebele|C|29|30|30|0|
|Historical-element_protected|belebele|D|12|46|46|1|
|Historical-element_protected|belebele|E|15|34|34|0|
|Historical-element_protected|knowledge|A|31|32|32|4|
|Historical-element_protected|knowledge|B|10|17|17|1|
|Historical-element_protected|knowledge|C|17|17|17|1|
|Historical-element_protected|knowledge|D|6|24|24|0|
|Historical-element_protected|knowledge|E|11|18|18|0|

## 限制与判断

对照文本来自独立训练侧阅读池，未给出参考答案，按长度匹配并排除连续16词重叠、限制内容词Jaccard；未做母语语义隔离认证，可能存在干扰。E−F不能直接称为内部通路的因果效应。

C/D仅为提示要求诊断，MCF没有生成解释，不能据此认证实际输出语言。单次正确不认证掌握。选项固定主结果与预定循环排列控制分开提供，不能按结果挑排列。

生成内容判定保留“明确正确、明确错误、未解析”三类；严格Final格式、是否末行与解释语言单列。语言检测对短文本和混合语言不可靠。

尚未训练能力访问修复算法。是否值得设计训练侧选择性监督，应结合E−B、F−B、E−F在阅读/知识及排列控制下的一致性讨论，不自动追加实验。

GPU上下文成本：0.0565 GPU小时，仅作记录，无停止阈值。

逐题原始MCF位于 SOURCE_AND_IDENTITY.json 所列私有路径；含题目与回答的材料不公开发布。

## 交接要点

能力访问诊断v2已完成：3个历史快照，原64阅读+64知识题，3840条新MCF判断，1920条旧生成回答重解析；无训练、无新生成。

主指标是统一单Token A/B/C/D标签条件概率argmax；原始logP及四标签内归一化概率均保留。4主条件原选项，2辅助解释语言提示条件；4主条件另有预定循环排列控制。C/D不解释为实际输出语言控制。

阅读主结果（SW单独 / SW+对应EN / SW+无关EN，正确数/64）：Base18/45/21；LR077-s42为20/44/18；历史英语保护30/44/21。三者E-B及E-F来源配对区间下界均>0，在预定循环排列控制下同方向且区间仍>0。

知识主结果：Base16/23/18；LR07717/20/19；历史英语保护14/20/19。主排列中E-F区间均跨0；循环排列中也跨0，因此不支持跨任务普遍收益。

旧生成冻结评分明确正确557/1920，旧事后解析879，新v2解析也是879；新解析未解析75条。多出322条可明确解析的正确选择说明格式问题确实影响原结论；未解析不是内容错误。新语言检测仍为启发式。

LR077与历史保护在英语辅助阅读均44/64，旧生成解析里的反超未在本MCF复现。两快照训练预算/配方不同，禁止当作算法胜负。

控制材料来自独立训练侧英语阅读池，不给出参考答案，做长度匹配、16词重叠筛查和内容词Jaccard限制；文本Token差的绝对值中位数1，最大差-49/+41，相对差最高70.27%，因此不是严格等长或因果控制。无母语者语义隔离认证，无关英语可能干扰。正确英语辅助提高的是该上下文下答题表现，不能声称SW独立能力提升或内部能力通路已定位。题目已暴露，本轮是探索性复核。

技术预检发现最后位置优化logits与标准完整前向的BF16舍入差；保留失败记录并改用全体模型统一标准完整前向，在recovery01重做。全部正式任务通过；技术失败成本计入COSTS.json。

详见README_ZH.md、MCF_RESULTS.csv、MCF_PAIRED_TRANSITIONS.json、FORMAT_SUMMARY.csv及逐题CSV。下一步仍先完成完整预算baseline；本轮不训练能力访问修复算法。

