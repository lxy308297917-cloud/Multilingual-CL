# 2026-10-28：能力访问诊断v2与完整预算基线进度

本入口将**已完成诊断**与**正在执行的正式基线**分开。尚无本轮完整预算方法排名。

## 已完成：跨语言能力访问诊断v2

- 3个既有快照、64道同源阅读题和64道同源知识题；3840条新MCF判断全部核验，1920条旧生成回答完成回顾性重解析。
- 阅读中，SW-only / SW+对应英语 / SW+无关英语的正确数分别为：Base 18/45/21；LR077-s42 20/44/18；历史英语保护30/44/21（各64题）。三个模型的E−F配对区间下界均大于0，预定循环选项排列下方向保持。
- 知识题的E−F区间均跨0，不能声称跨任务普遍有效。
- 原冻结生成评分明确正确557/1920，新解析879，未解析75；这是回顾性评分诊断，不能当作新的独立确认。
- 这说明对应英语信息能帮助这批阅读题的判断，不证明SW独立能力提升或内部因果通路。题目已暴露、控制长度并非完全匹配，无母语者语义隔离认证。不同历史快照的训练预算/配方不同，不能据此作算法胜负比较。

[完整v2报告](docs/results/access_gap_diagnostic_v2/README_ZH.md) · [交接摘要](docs/results/access_gap_diagnostic_v2/HANDOFF_TO_CHATGPT_ZH.txt) · [汇总表](docs/results/access_gap_diagnostic_v2/MCF_RESULTS.csv)

## 执行中：统一完整预算SW基线

- Qwen2.5-1.5B-Instruct；Aya SW 80%＋EN→SW10%＋SW→EN10%，按非padding主输入Token。
- 六组训练：Dense FFT、MoFO、SSU、HFT15、元素稀疏无保护、元素稀疏英语敏感度保护；从同一Base、同一顺序开始，各42,465次更新、15,360,935主输入Token、5,984,371监督Token。
- HFT15/SSU/两元素稀疏组，每步名额精确196,608,000，占196个候选矩阵15.0060024%；不是全模型15%。HFT15是整矩阵预算匹配版本，不是原论文50%。MoFO采用官方阈值并列和全候选衰减，实际写回范围不能伪称与其余方法相同。
- 六组真实模型预检、最长batch与恢复检查均通过；四组已启动，剩余两组依队列接续。首个正式2000步checkpoint的文件哈希已核验；完整训练和终点评测尚未完成。
- Base与六个端点统一测：EN/SW Belebele各900题ACC、EN MMLU/SW Global-MMLU各14,042题ACC（另报学科宏平均）、本地FLORES-200派生翻译1012题/方向chrF++、IFEval541提示/834指令strict、GSM8K1319题。
- 阅读采用选择题ACC；不使用KenSwQuAD词面F1替代。所有方法在同一完整预算终点评测，不按测试分数选checkpoint。

[冻结协议](docs/research/sw_fullbudget_baselines_20261028/FULL_BUDGET_BASELINE_PROTOCOL_ZH.md) · [训练配置](docs/research/sw_fullbudget_baselines_20261028/TRAINING_CONFIG_FROZEN.json) · [评测协议](docs/research/sw_fullbudget_baselines_20261028/EVALUATION_FROZEN.json)

## 已接续：完整预算模型统一v2诊断

按原始用户提示词要求，在正式基线主评测完成后，对六个新模型复用相同冻结v2提示进行MCF诊断，每模型1280条，共7680条；Base复用已有核验结果。当前等待上游完成，无新训练、无新生成解释、无新增条件或选参。其分析脚本已在3840条旧v2记录上复核，全部60个汇总单元一致；这不是新模型已出分。

[接续协议](docs/research/access_gap_fullbudget_followup_20261028/FOLLOWUP_PROTOCOL_FROZEN.json)

## 执行边界与交付

全账号最多四张GPU，不设置累计GPU小时或墙钟停止上限；成本只作记录。不中断其他用户任务。检查点实际落盘于本账号/data1目录，保留统一入口与完整恢复状态。一次性协调器不设开机启动、故障自动重启。

终点将输出统一CSV/中文主表、逐题原始结果路径、配对差值和区间、更新量与成本。单训练顺序及已暴露benchmark的限制会单列；当前不声称新方法已稳定优于baseline。原始题目、回答及权重仅留服务器，公开发布聚合证据与可审阅代码。
