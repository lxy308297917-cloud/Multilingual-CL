# 2026-10-29 完整预算基线：benchmark阶段结果

已验收Base、Dense FFT、SSU变体、HFT15的EN/SW Belebele各900题，共8项完整阅读评测。其他模型和任务尚未齐全，不是整轮最终表。

|模型|SW阅读ACC|EN阅读ACC|
|---|---:|---:|
|Base|37.56%|81.78%|
|Dense FFT|43.33%|63.67%|
|SSU-token-weighted-15matched|43.56%|82.11%|
|HFT-whole-matrix-15matched|40.78%|80.11%|

FFT相对Base：SW +5.78个百分点，来源配对95%区间[+2.23,+9.32]；EN −18.11个百分点，区间[-21.33,-14.92]。SSU变体相对FFT：EN +18.44个百分点，区间[+15.25,+21.59]；SW点估计仅高0.22个百分点，差异区间跨0，不能称严格相等或已证明非劣。
这支持当前阅读任务上较好学习—保持取舍的线索，不证明所有benchmark或多个训练seed的稳定优势，也不是P英语保护有效的证据。

[完整阅读报告与区间](docs/results/sw_fullbudget_reading_stage_20261029/README_ZH.md) · [CSV](docs/results/sw_fullbudget_reading_stage_20261029/READING_SCORES.csv) · [差值图](docs/results/sw_fullbudget_reading_stage_20261029/READING_DIFFERENCES.svg)

同一Qwen2.5-1.5B-Instruct；适配模型均使用完整Aya80/10/10配方、42,465步、15,360,935主输入Token。阅读为统一3-shot MCF选择题ACC，不是生成F1。Dense为全参数；HFT15为整矩阵预算匹配变体，SSU使用Token加权激活而非参考样本等权聚合。HFT15/SSU/U/P共同名额196,608,000，占196候选矩阵15.0060024%。额外英语校准曝光不同，不称相同总数据或计算。
MoFO按用户要求暂停，原定七模型整轮未完成。U/P仍待完整端点评测；知识、双向完整翻译、指令和数学不能填入未完成分数。历史D2与本轮配方及部分评测协议不同，分表报告。

[已完成能力访问诊断v2](docs/results/access_gap_diagnostic_v2/README_ZH.md)与[新表示漂移诊断](docs/results/representation_drift_functional_forgetting_20261028/README_ZH.md)分开报告；漂移主分析等待完整U/P端点，尚无全套关联或激活替换结论。机制代理不替代目标benchmark。
