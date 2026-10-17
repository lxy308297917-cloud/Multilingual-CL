# 冻结 Base 的 SW/EN 模块梯度诊断

这是训练侧只读诊断：Qwen2.5-1.5B-Instruct Base 不写回权重。D2 开发池的 Aya、EN→SW、SW→EN 各取 16 例，独立英语 instruction／reading／math 池也各取 16 例；EN→SW 样本避开此前 448 个开发集 ID 和正式测试源。原始逐例数值、输入身份和代码哈希见本目录 `RAW.jsonl`、`manifest.json`、`PROBE_RELEASE.json`、`DONE.json`，汇总见 `RESULTS.json`。GPU 已释放。

EN→SW 梯度能量逐模块的 8/8 分半排序 Spearman 为 Attention 0.748、MLP 0.945；前七组重叠分别为 4、7。SW→EN 对应为 0.824、0.902，前七组重叠 4、6。EN→SW 最靠前的 MLP 层为 1、2、27、3、24、5、26；原 LP 名额表仅覆盖其中的 3、5。Aya 分半稳定性较弱（Attention 0.563、MLP 0.477）。

这说明训练侧任务损失对若干模块的局部敏感度可测，但**梯度能量并非更新这些模块后的下游收益**，不能据此宣称发现了有效训练层。后续 `sw_specificity_budget_probe_20261003` 用该代理构造连续预算，并与同训练顺序、同精确名额的 LP 和均匀分配做 2048 步短训和新的独立开发集比较；结果若不稳定或不优于均匀分配，就停止该预算方案。
