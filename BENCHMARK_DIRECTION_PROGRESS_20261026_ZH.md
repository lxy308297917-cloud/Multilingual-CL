# Qwen2.5-1.5B SW适配：阶段完成与下一轮队列（2026-10-28）

**旧LR077诊断、七模型benchmark补评、U/P/R/N/N_abs五组匹配短训和统一开发筛查均已完成。方向风险候选N未通过预定门槛，停止扩展，不新增第二顺序或SSU/HFT训练。后续已授权纯评测诊断排队，等待本轮发布验收完成后执行。**

[本轮完整中文报告](docs/results/direction_risk_20261028/README_ZH.md) · [完整分数CSV](docs/results/direction_risk_20261028/SCREEN_TABLE.csv) · [配对区间](docs/results/direction_risk_20261028/SCREEN_COMPARISONS.json) · [逐项去留判定](docs/results/direction_risk_20261028/SCREEN_DECISION.json) · [产物验收](docs/results/direction_risk_20261028/COMPLETION_VERIFIED.json)

## 本轮五组结果

|模型|SW Belebele ACC%|EN→SW chrF++|SW→EN chrF++|EN Belebele ACC%|EN MMLU ACC%|SW Global-MMLU ACC%|IFEval prompt strict%|
|---|---:|---:|---:|---:|---:|---:|---:|
|Base|37.56|17.65|16.88|81.78|63.68|30.00|39.06|
|U 无保护|34.33|18.28|28.30|79.11|60.00|29.12|26.56|
|P 敏感度保护|37.67|20.55|29.54|81.33|63.68|28.95|32.81|
|R 同池回放|34.89|17.73|28.13|77.89|58.77|28.42|24.22|
|N 正方向风险|34.89|23.86|28.76|72.44|52.81|31.75|22.66|
|N_abs 绝对风险|37.22|24.40|27.46|81.67|63.16|30.18|35.16|

- **N没有实现保护目标。** 相对P，EN阅读−8.89个百分点，配对95%区间[−11.66,−6.26]；EN MMLU−10.88，[−14.56,−7.19]；IFEval prompt strict−10.16，[−18.75,−1.56]。EN→SW虽提高3.30，[+1.98,+4.66]，仍未通过保持门槛。
- **无符号消融存在取舍，不能事后改成成功主方法。** N_abs相对P，EN→SW+3.85 chrF++，[+2.40,+5.30]；SW→EN−2.08，[−3.68,−0.57]。其更新幅度也更小，不能单独归因于风险符号。
- **P仍是值得保留的简单保护基线。** 阅读和EN知识点估计接近Base，翻译改善，但未解决所有英语损伤。单顺序短预算不能推出全面能力保持。

五组均4293步、1,536,351主输入Token、599,469主监督Token；逐矩阵15%共196,529,312元素名额/步。P/R/N/N_abs各额外访问389,435输入及101,635监督Token。没有LoRA、OPD、表示对齐或子空间约束。R的0.1间歇回放只代表本轮配方，不能外推回放普遍无效。

## 评分与证据范围

- Belebele为选择题ACC，每语言900题；KenSwQuAD自由回答F1不参与本轮门槛。
- MMLU/Global-MMLU每语言570题，IFEval为官方材料的固定128提示子集；不冒充全量benchmark分数。
- 所有模型采用同题、同提示、同生成/选项评分。17,640次选择判断（含复用Base 2,940）、2,304条生成回答（含Base384）已验收；选择分数由原始四选项logprob重新计算，输入Token与金标跨模型一致。
- 已暴露材料属于开发补评，不是独立最终测试。配对区间反映固定模型的题目不确定性，不代表训练seed总体。
- 全部最终模型与恢复state.pt文件哈希已重新计算；配置、模型身份、更新统计、TensorBoard及逐题结果可追溯。原题、回答和权重留在服务器。

[配置与模型身份](docs/results/direction_risk_20261028/IDENTITIES_AND_CONFIG.json) · [实际更新统计](docs/results/direction_risk_20261028/TRAINING_UPDATE_STATS.json) · [生成行为](docs/results/direction_risk_20261028/OUTPUT_BEHAVIOR.json) · [分项GPU成本](docs/results/direction_risk_20261028/GPU_COST_ACCOUNTING.json)

## 此前已完成

[旧LR077顺序诊断](https://github.com/lxy308297917-cloud/Multilingual-CL/blob/5ddd6d73fdc39abf4bf0ab89e5a72b03be82b5e4/ORDER_DIAGNOSTIC_PROGRESS_20261025_ZH.md)：512条中间点回答及逐题重评分完成，未定位明确实现错误或足够的任务聚集解释。

[七模型benchmark接轨及完整结果](https://github.com/lxy308297917-cloud/Multilingual-CL/blob/2ded993eb9bdf20cd355c1818eb96b9f09441946/BENCHMARK_DIRECTION_PROGRESS_20261026_ZH.md)：20,580次选择判断，另128题速度预检。Attention-only与Mix50未显示跨两训练顺序稳定优于LR077的整体证据，停止这些固定配方。

[历史计划与恢复记录](https://github.com/lxy308297917-cloud/Multilingual-CL/blob/deba86386dce4d330264611110e54e7208f389b0/BENCHMARK_DIRECTION_PROGRESS_20261026_ZH.md)保留旧16/20小时条款作追溯。**最新用户政策已覆盖这些条款：不设置累计GPU小时或墙钟上限，不因时间阈值终止既定任务，仍全账号最多四卡。** 本轮实测闭合GPU上下文10.794小时，中断时间估算4.660小时，合计15.454小时；另列1小时不确定性余量，不称为实耗。七模型补评另1.375小时。

原中断与同秒Codex会话Shutdown记录吻合，未发现他人杀进程的证据。恢复改为手动启动的独立一次性用户服务，无自启或故障自动重启；历史失败日志保留。

## 下一轮：能力访问缺口诊断（纯评测）

已手动排队`access-gap-after-direction-20261027.service`：等待本阶段报告发布验收，不在等待时占GPU。用Base、LR077-s42、历史英语保护模型，复用/补齐同源EN/SW选择题缓存，统计四种对错状态与适配前后转移，再完成A–E输入/解释语言干预1920条回答。

本轮不新增训练，不把英语对/SW错直接当作内部能力访问障碍的证明，不根据开发题选训练样本；旧D2与新短训预算不同，只作分别相对Base的机制描述。方法设计须等待这些行为证据，不因现有原型失败继续堆叠组件。
