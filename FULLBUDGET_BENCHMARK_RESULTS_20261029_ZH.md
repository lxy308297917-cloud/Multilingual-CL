# 2026-10-29 完整预算基线：benchmark阶段结果

这是阶段入口，非整轮最终表。已完成Base、HFT15与SSU变体的EN/SW Belebele各900题正式任务；其他方法及任务仍待完成。

|模型|SW阅读ACC|EN阅读ACC|
|---|---:|---:|
|Base|37.56%|81.78%|
|HFT15|40.78%|80.11%|
|SSU变体|43.56%|82.11%|

SSU变体相对Base：SW +6.00个百分点，来源配对95%区间[+2.63,+9.49]；EN +0.33个百分点，区间[-0.89,+1.56]。当前证据支持该固定模型在这批SW阅读题上的收益；不能推广为所有benchmark或多训练seed的稳定优势，也不是英语保护P有效的证据。

相同模型与完整Aya80/10/10配方，42,465步、15,360,935主输入Token；阅读使用统一3-shot MCF、选择题ACC。HFT15是整矩阵预算匹配变体；SSU使用Token加权激活，非参考样本等权实现。两者每步名额均196,608,000，占196候选矩阵15.0060024%。额外英语校准曝光不同，不能称为相同总数据或计算。

[完整阅读阶段报告与区间](docs/results/sw_fullbudget_reading_stage_20261029/README_ZH.md) · [CSV](docs/results/sw_fullbudget_reading_stage_20261029/READING_SCORES.csv) · [差值图](docs/results/sw_fullbudget_reading_stage_20261029/READING_DIFFERENCES.svg)

MoFO按用户要求暂停，原定七模型整轮未完成。Dense/U/P及知识、翻译、指令、数学均不能填入未完成成绩。历史FineWeb主导D2与本轮Aya主导配方分表；不能从跨配方差值单独推断算法收益。

[已完成能力访问诊断v2](docs/results/access_gap_diagnostic_v2/README_ZH.md)与[新表示漂移诊断](docs/results/representation_drift_functional_forgetting_20261028/README_ZH.md)分开报告；后者目前仅Base表示提取完成，尚无完整跨模型漂移或激活替换结论。代理指标不替代目标benchmark。
