# 多语言顺序学习 V2 实验

MASSIVE 为主基准。MasakhaNER 2.0 与 MTOP 各自使用五种语言和固定开发集 Single 排序。TyDiQA-GoldP 和 XNLI 的已有完整轨迹用于跨任务、提示与标签读出诊断。

三个主基准均已完成 LoRA 和 Full-FT 的 Sequential、Replay 与 Joint 六组基线。下表是 5e-5 设定下官方测试集的五语等权平均 F1（%）。MASSIVE 和 MTOP 使用槽位 F1，MasakhaNER 2.0 使用实体 F1。

| 基准 | LoRA Sequential | LoRA Replay | LoRA Joint | Full-FT Sequential | Full-FT Replay | Full-FT Joint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MASSIVE | 61.05 | 61.81 | 63.85 | 28.41 | 44.57 | 49.86 |
| MasakhaNER 2.0 | 67.37 | 67.87 | 71.41 | 57.89 | 61.22 | 67.65 |
| MTOP | 59.71 | 61.39 | 68.63 | 50.57 | 56.07 | 55.25 |

全参回放在三项基准上均提高了最终均分。MASSIVE 的 Full-FT Sequential 在 1e-5 达到 62.66%，同学习率 Joint 达到 63.87%，说明更新幅度会显著改变基线。逐语习得与保留、正迁移和配对区间见[完整结果](RESULTS_ZH.md)。平行数据和输出语种分析单列为探索性结果。

| 查看内容 | 文件 |
| --- | --- |
| 11 月 23 日中午前的基线与研究工作目标 | [修订目标](GOAL_ZH.md) |
| 数据、固定顺序、全参及 LoRA 训练配置、共同指标定义 | [冻结协议](PROTOCOL_ZH.md) |
| 带完整标签表、聊天模板和监督答案的实际训练样例 | [训练输入原文](PROMPTS_ZH.md) |
| 旧实验实际学习率、步数与优化器接续 | [配置和检查点审计](CONFIG_AUDIT_ZH.md) |
| 32 条已完成历史轨迹的 AP、FWT、BWT、习得与遗忘 | [历史指标](CL_HISTORICAL_ZH.md) |
| MasakhaNER 五语扩充与 MTOP 的 Single 排序 | [课程冻结记录](COURSES_ZH.md) |
| 本轮新阶段矩阵、控制组及配对比较 | [V2 结果总表](RESULTS_ZH.md) |
| LoRA 与全参、回放与顺序训练、学习率及课程变化的逐语比较 | [同题配对对照](MATCHED_COMPARISONS_ZH.md) |
| MASSIVE 全参顺序与回放的第一个分支 | [阿语阶段五语测评](MASSIVE_AR_FIRST_ZH.md) |
| 进程、完成回执及评分覆盖 | [运行状态](RUN_STATUS_ZH.md) |
| 样本、词元、卡内微批、参数量、存储字节、耗时和峰值显存 | [测得训练预算 CSV](V2_TRAIN_BUDGETS.csv) |
| 上述计量口径在复用检查点上的实际记录 | [历史训练预算 CSV](HISTORICAL_TRAIN_BUDGETS.csv) |
| 合法输出、类型和边界等错误计数 | [输出诊断 CSV](V2_ERROR_COUNTS.csv) |
| 同题共同有效子集上的配对变化 | [格式影响 CSV](V2_COMMON_VALID.csv) |
| Sequential 和 Replay 在共同有效题上的同阶段差 | [方法间格式影响 CSV](V2_COMMON_VALID_METHODS.csv) |
| 意图、整句匹配、输出合法率与跨度校验 | [各语辅助指标 CSV](V2_SECONDARY_SCORES.csv) |
| MASSIVE 全参回放的逐语习得与保留效应，包括整句匹配的配对区间 | [MASSIVE 回放配对细表](MASSIVE_REPLAY_TRADEOFFS.csv) |
| 平行回放与语言生成的调研、试验和研究切入点 | [研究报告](RESEARCH_DIRECTIONS_ZH.md) |
| 目标语标题干预及末语交换的实际翻译表现 | [FLORES-200 试验](FLORES_LANGUAGE_RESULTS_ZH.md) |
| 两例提示复测、内容候选及语言选择的诊断 | [两例提示](FLORES_FEWSHOT_RESULTS_ZH.md) · [候选与脚本](FLORES_CANDIDATE_ZH.md) |
| 等预算平行答案训练小实验 | [配对方法协议](TRANSLATION_REPAIR_PROTOCOL_ZH.md) · [实验结果](TRANSLATION_REPAIR_RESULTS_ZH.md) |
| 平行题能否在更新前预测旧语失分 | [风险筛查协议](PARALLEL_RISK_PROTOCOL_ZH.md) · [开发集结果](PARALLEL_RISK_RESULTS_ZH.md) |
| 仅检查语种或同时检查重复等错误后回退到底座 | [回退协议](FLORES_FALLBACK_PROTOCOL_ZH.md) · [配对结果](FLORES_FALLBACK_RESULTS_ZH.md) |
