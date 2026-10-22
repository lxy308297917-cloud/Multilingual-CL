# 多语言适配实验进度：2026-10-22最终汇总

本提交包含两条已完成轨道；原始逐题输出、模型与可恢复状态留在服务器，公开为代码、冻结协议、聚合结果、图和核验凭证。

|轨道|已完成内容|结论入口|
|---|---|---|
|增强InfoNCE D1|16批训练侧系数校准；同Base4293步；冻结512条生成及独立1000句对诊断|[完整报告](docs/results/infonce_strength_20261022/README_ZH.md)|
|相同元素预算的模块选择|S-All／S-Attn／S-MLP各4293步；每步名额77,070,336；同一数据前缀及冻结终点评测|[完整报告](docs/results/module_budget_sparse_20261022/README_ZH.md)|

D1的当前EN↔SW独立检索由D0的8.45%升至63.00%，但EN→SW翻译27.024未超过同池翻译CE的27.621；配对差值区间[-2.498,+1.350]。对齐信号确实增强，尚无超过简单监督的任务收益；按冻结规则停止当前固定原型。

模块对照中，S-Attn的EN→SW为23.021，S-All为15.064；英语指令strict为25.000%与16.406%。但反向翻译未通过非劣门槛，英语阅读优势不确定，且Attn实际更新路径仅为All的77.14%；不能直接归因于模块位置或升级子空间方案。

模块选择的预定Attention优先联合筛查：未通过。保留局部优势与不确定性，不直接升级Attention-only或子空间方案。具体任务、置信区间、实际BF16位移和稀疏写回统计见对应报告，不能用固定K代替实际更新强度匹配。

两轨都为单seed开发筛查，不确认普遍有效或无效。没有使用LoRA、OPD或正式测试；数学评分未验收而未纳入。原D2历史与这些新Aya80＋双向翻译20实验分开，Dense BF16动量与Sparse FP32动量也分开。

## 可核验入口

- [D1原始服务器完成验核](docs/results/infonce_strength_20261022/COMPLETION_VERIFIED.json)
- [D1冻结判定](docs/results/infonce_strength_20261022/MANDATORY_DECISION.json)
- [模块三组服务器完成验核](docs/results/module_budget_sparse_20261022/COMPLETION_VERIFIED.json)
- [模块三组冻结判定](docs/results/module_budget_sparse_20261022/MANDATORY_DECISION.json)

不自动追加第二seed、系数／层位扫描、子空间约束或位移匹配对照。负结果与不确定性完整保留。
