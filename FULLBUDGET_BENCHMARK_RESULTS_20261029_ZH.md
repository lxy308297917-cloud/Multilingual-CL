# 10月29日完整预算基线：已验收阶段结果

本轮统一Qwen2.5-1.5B-Instruct、Aya SW指令80%＋双向翻译各10%（主输入Token口径），每个完整训练端点42,465步、15,360,935主输入Token、5,984,371监督Token。当前已验收10/42个非暂停模型的完整分任务，其余保持空白。MoFO按用户要求暂停。

[完整阶段主表](docs/results/sw_fullbudget_benchmark_stage_20261029/README_ZH.md) · [CSV](docs/results/sw_fullbudget_benchmark_stage_20261029/AVAILABLE_BENCHMARKS.csv) · [配对区间](docs/results/sw_fullbudget_benchmark_stage_20261029/PAIRED_COMPARISONS.json) · [阅读明细与图](docs/results/sw_fullbudget_reading_stage_20261029/README_ZH.md) · [表示漂移当前状态](docs/results/representation_drift_functional_forgetting_20261028/STATUS_20261029_ZH.md)

|模型|SW Belebele ACC|EN Belebele ACC|EN→SW chrF++|SW→EN chrF++|
|---|---:|---:|---:|---:|
|Base|37.56%|81.78%|21.18|21.79|
|Dense FFT|43.33%|63.67%|44.29|28.18|
|SSU Token加权15%匹配变体|43.56%|82.11%|待完成|待完成|
|HFT整矩阵15%匹配变体|40.78%|80.11%|待完成|待完成|
|无保护元素稀疏U|待完成|待完成|待完成|待完成|
|英语敏感度保护P|待完成|待完成|待完成|待完成|

阅读每语言900题，修正同源文档示例排除后的3-shot选项似然ACC；翻译为完整1,012题/方向的语料级chrF++。知识、IFEval与GSM8K尚待完整任务验收，不能推测其成绩。

现阶段可用于汇报的发现：

- FFT相对Base：SW阅读+5.78个百分点，配对95%区间[2.23,9.32]；EN阅读−18.11个百分点，区间[−21.33,−14.92]。
- FFT相对Base：EN→SW翻译+23.11 chrF++，区间[22.31,23.89]；SW→EN +6.39，区间[5.29,7.50]。新配方下翻译学习与英语阅读损伤同时存在。
- SSU变体SW阅读相对Base +6.00个百分点，区间[2.63,9.49]；EN阅读+0.33，区间跨0。相对FFT，EN阅读高18.44个百分点，区间[15.25,21.59]；SW点估计接近，但未证明相等或非劣。
- 我们的英语保护P是否优于无保护U，仍没有结果；SSU的成绩不能当成P的验证。

这些区间反映固定模型上的题目抽样不确定性，不代表跨训练种子稳定性，也不构成多任务整体最优证明。SSU当前实现使用Token加权激活聚合，非严格样本等权参考实现；HFT是整矩阵预算匹配变体，非HFT50。稀疏候选范围和额外英语校准曝光在报告中分别披露。

旧FineWeb主导D2、近期短训代理结果与本轮分表；不能从跨配方数字直接推断更换数据来源的因果收益。已暴露benchmark属于补充比较，不包装成未见最终确认。表示漂移是无训练机制分析，当前Base与FFT提取完成，U/P未齐，尚无四模型完整关联或激活替换结论。
