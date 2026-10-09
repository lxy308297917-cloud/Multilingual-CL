# 给 ChatGPT 的研究评议入口

更新2026-10-09（已补充国庆完整结果）。**历史实验统一入口是[仓库根目录全局总览](../../../EXPERIMENT_RESULTS_OVERVIEW_ZH.md)**，完整分数/分项报告见[假期结果索引](../../results/holiday_20261008/README.md)。请先读 [METHOD_ZH.md](METHOD_ZH.md)、[RELATED_WORK.md](RELATED_WORK.md)、[EXPERIMENT_PLAN_ZH.md](EXPERIMENT_PLAN_ZH.md)，再读 [core.py](../../../experiments/on_policy_sparse_cl_v1/core.py) 与 [demo_cpu.py](../../../experiments/on_policy_sparse_cl_v1/demo_cpu.py)。本包未做Qwen/GPU实验，无方法成绩。

## 已有可信项目进展（历史结果，非候选成绩）

- SW 15%保护/去保护seed43正式EN→SW为35.20/33.42，SW→EN44.12/38.23，EN阅读80.78/77.78，IFEval prompt28.84/22.00，GSM8K18.50/16.22。
- seed42/43双向翻译和EN阅读配对描述区间均为正；GSM seed43区间跨零，不宣称稳定数学保护。
- 同预算5.798%均匀保护EN→SW seed42/43为33.87/33.04；LP为16.34/16.56，不沿旧LP集中预算作为主方法。
- 固定mask与EMA未达独立开发短训门槛，EMA正式日程前缀复核仍失败；梯度方向与撤销定位信号未达到可用稳定性。
- 10月8日独立开发集保护减去保护：EN阅读+10.55个百分点，指令prompt+6.25；SW翻译/摘要差异未明确。数学####格式受限。
- Igbo seed43现已有最终报告及四模型44项正式DONE完整性审计。首尾−分散目标摘要/双向翻译两轮同向；GSM8K两轮分别−4.70/−15.69个百分点。原IFEval保留，语言检测固定seed的重评分另列诊断，不覆盖正式成绩。
- SW开发基线已补齐MoFO/SSU/HFT及seed42去保护。seed42指令约束保护差+5.19个百分点、区间为正，英语阅读−0.39；seed43阅读+10.55。开发到正式差值不稳定：两轮SW→EN开发约零/负，正式+4.49/+5.89，不能单独凭当前翻译dev决定新方法长训。详见已同步的DEVELOPMENT_TRANSFER_AUDIT报告。

这些值来自服务器分项报告的汇编，不替代原始输出。旧GitHub部分报告使用过早的分数或方案口径，当前简报与旧稿冲突时请要求核对模型身份、协议和正式DONE，不混用两者。历史训练seed改变也伴随输入步顺序改变。

## 希望你重点批评

1. 在SDFT、iSDFT、Dense Supervision Sparse Updates及GAST等先例下，“当前轨迹＋实际Adam候选更新风险＋精确逐矩阵top-k”究竟有多少剩余新颖性？请给具体重合机制与最接近论文。
2. u=-g_new*d是否过度依赖同batch，风险正部＋经验score-square proxy是否有可测优势，初始化KL梯度为0如何正确处理？不用代理分数替代任务表现。
3. 本地KL stop-gradient-prefix的semi-gradient解释是否准确？缺失状态分布导数、teacher提示位置、采样温度、EOS与陈旧度有哪些偏差？
4. 三阶段CL流、双参考与固定prompt记忆是否公平？是否应先用标准CL benchmark，再加低资源语言验证？
5. 核心四格消融是否能拆开on-policy、选参、刷新频次与额外记忆？还缺哪个最有说服力的强基线？
6. 当前翻译开发差值未可靠预示正式集，请评议如何建立训练侧的更可信准入证据，避免重复用正式测试挑方法。
7. 哪一项最便宜的前置实验能最快否决这个方案？如果无增益，如何写有价值的机制分析而不制造“有效新方法”？

## 可直接复制的提问

> 请审阅这个研究分支的 docs/research/on_policy_sparse_cl/CHATGPT_REVIEW_BRIEF_ZH.md、METHOD_ZH.md、RELATED_WORK.md、EXPERIMENT_PLAN_ZH.md 与 experiments/on_policy_sparse_cl_v1/core.py。目标是提出可发表的持续学习方法，但当前只有14项CPU测试和合成演示，没有真实模型结果。请先与SDFT、iSDFT及Dense Supervision Sparse Updates等最近工作比较新颖性，再指出数学/实现缺陷，提出最小可否证实验与必要基线。不要把历史SW保护版成绩当作新候选结果。如果你无法读取某文件，请明确指出，不要假定已读。
