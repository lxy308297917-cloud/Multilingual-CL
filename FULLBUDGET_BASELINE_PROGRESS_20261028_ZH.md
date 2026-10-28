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
- 六组真实模型预检、最长batch与恢复检查均通过。用户于17:33明确要求立即暂停MoFO：最后日志3940步，保留2000步完整checkpoint，不自动恢复。Dense/SSU/HFT三组继续，元素稀疏U/P在其余三卡排队，GPU1用于新表示诊断；完整训练和终点评测尚未完成。
- Base与六个端点统一测：EN/SW Belebele各900题ACC、EN MMLU/SW Global-MMLU各14,042题ACC（另报学科宏平均）、本地FLORES-200派生翻译1012题/方向chrF++、IFEval541提示/834指令strict、GSM8K1319题。
- 阅读采用选择题ACC；不使用KenSwQuAD词面F1替代。所有方法在同一完整预算终点评测，不按测试分数选checkpoint。

[冻结协议](docs/research/sw_fullbudget_baselines_20261028/FULL_BUDGET_BASELINE_PROTOCOL_ZH.md) · [训练配置](docs/research/sw_fullbudget_baselines_20261028/TRAINING_CONFIG_FROZEN.json) · [评测协议](docs/research/sw_fullbudget_baselines_20261028/EVALUATION_FROZEN.json)

## 已接续：完整预算模型统一v2诊断

按原始用户提示词要求，在正式基线主评测完成后，对六个新模型复用相同冻结v2提示进行MCF诊断，每模型1280条，共7680条；Base复用已有核验结果。当前等待上游完成，无新训练、无新生成解释、无新增条件或选参。其分析脚本已在3840条旧v2记录上复核，全部60个汇总单元一致；这不是新模型已出分。

[接续协议](docs/research/access_gap_fullbudget_followup_20261028/FOLLOWUP_PROTOCOL_FROZEN.json)

## 执行边界与交付

全账号最多四张GPU，不设置累计GPU小时或墙钟停止上限；成本只作记录。不中断其他用户任务。检查点实际落盘于本账号/data1目录，保留统一入口与完整恢复状态。一次性协调器不设开机启动、故障自动重启。

终点将输出统一CSV/中文主表、逐题原始结果路径、配对差值和区间、更新量与成本。单训练顺序及已暴露benchmark的限制会单列；当前不声称新方法已稳定优于baseline。原始题目、回答及权重仅留服务器，公开发布聚合证据与可审阅代码。

## 新增：表示漂移与功能遗忘诊断（仅推理）

已阅读上传论文§4.4/§5.4与原图，冻结1152条诊断输入、来源分组发现/确认划分、28层位置、指标及进入激活替换的门槛。主比较仅Base与匹配完整预算FFT/U/P，SSU/HFT/MoFO未加入补充分析。

Base已完成并核验1152条输入、28层×3位置表示及全部逐题ID/Token/文件哈希，消耗0.408 GPU小时。FFT/U/P的跨模型分析等待完整端点，尚无漂移—遗忘关联或激活干预结论。数学原####格式解析全部未解析；保存原记录，另冻结内容解析v2：58题明确数值正确、23题明确数值错误、47题仍未解析；不能把未解析当成知识遗忘。

本诊断使用固定零样本提示，分数不与正式3/5-shot benchmark混合。材料已暴露，仅作机制开发诊断。t-SNE只用于辅助展示，不决定层位或方法去留。论文跨层均值协方差与本轮逐层跨样本协方差分开。

[新诊断入口](docs/results/representation_drift_functional_forgetting_20261028/README_ZH.md) · [冻结协议](docs/results/representation_drift_functional_forgetting_20261028/PROTOCOL_FROZEN.json) · [SSU与MoFO审查](docs/results/baseline_correctness_audit_20261028/README_ZH.md)

## 本日晚间阶段更新：首份完整预算训练验收完成

HFT15完成42,465步，主输入15,360,935 Token、监督5,984,371 Token，每步精确196,608,000元素名额。权重/状态文件哈希、全部步数与计数、1,347,106,304冻结元素逐位不变、FP32动量有限及二阶动量非负等检查通过；实际GPU成本7.512小时。CPU RNG真实恢复通过，CUDA RNG字节与此前已通过真实GPU恢复的预检checkpoint一致；没有声称在正式终点重新做了GPU训练恢复。

U已接续训练；Dense和SSU继续，P仍排队。MoFO保持用户暂停，未自动恢复。正式benchmark尚未出分，当前不作方法胜负判断。其余五组与Base完成后可先交付明确排除MoFO的阶段表，不冒充原定七模型整轮完成。

[HFT15终点验收](docs/results/baseline_correctness_audit_20261028/HFT15_FULLBUDGET_TRAINING_ACCEPTANCE.json) · [当前执行政策](docs/research/sw_fullbudget_baselines_20261028/ACTIVE_EXECUTION_POLICY_20261028.json)
