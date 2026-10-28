# 表示漂移与功能遗忘诊断：执行入口

2026-10-28启动准备。仅推理与统计，不新增训练。

## 已落实

- 按用户要求立即停止MoFO，最后日志3940步，保留2000步完整checkpoint；不自动恢复MoFO。
- 原FFT、SSU、HFT训练继续，U/P在其余三卡继续排队。GPU1交给本轮诊断；不运行占卡空进程。
- 阅读上传论文§4.4、§5.4、Table5、Future Work，并检查Figure2/3原图及协方差公式。论文跨层均值协方差与本轮逐层跨样本协方差分开，不声称其模糊逐层曲线已严格复现。
- 固定1152条真实诊断输入：EN/SW阅读各256、EN/SW知识各256、EN数学128；来源分组发现/确认各半。材料均来自已暴露benchmark池，只作开发机制诊断。
- 主指标、28层边界、位置、协方差定义、统计控制、t-SNE参数、进入激活替换的门槛均在PROTOCOL_FROZEN.json中预先固定。

## 重要评测边界

本诊断使用固定零样本提示及MCF选择题准确率；正式大表仍使用已冻结的Belebele3-shot、知识5-shot及GSM8K5-shot协议。两套分数不合并。机制功能状态仅指本诊断固定协议中的对错。数学未解析答案单列，不直接算知识遗忘。

Base先提取。只有完整42465步、15,360,935主输入Token的FFT/U/P端点就绪后，才做跨模型比较。短预检、历史不同配方或中间checkpoint不进入主分析。

## 数据与执行位置

- 冻结协议：本目录PROTOCOL_FROZEN.json。
- 代码：outputs/representation_drift_functional_forgetting_20261028/code/。
- 私有题目、逐题预测、池化表示：work/representation_drift_functional_forgetting_20261028/。
- 每条仅存28层×3位置/池化×1536维FP32，约594MB/模型；不保存所有Token的激活。
- 主分析仅Base/FFT/U/P；SSU/HFT/MoFO未加入补充分析队列。

## 当前未完成

Base已完成1152条提取与逐题功能诊断，真实模型层边界、全部缓存哈希、输入ID及有限数值已核验；FFT/U/P完整端点未就绪。尚无漂移结果、t-SNE图或因果结论。激活替换仅在至少两个方法通过预定确认门槛后执行，不做层位或强度扫描。

最终将交付DRIFT_LAYERWISE.csv、FUNCTIONAL_TRANSITIONS.csv、DRIFT_FORGETTING_ANALYSIS.json、COSTS.json、SOURCE_IDENTITIES.json及真实数据图。未执行patching时明确记录原因，不制造PATCHING_RESULTS.csv正结果。

## Base阶段验收（2026-10-28）

- 1152×28×3×1536 FP32缓存全部有限，逐题ID、Token哈希与文件哈希均通过。
- GPU成本0.408小时，物理显存观测峰值3707MiB；没有训练更新。
- 数学原始v1格式解析128题全部未解析，不能称为0%数学能力。原因是自然输出使用boxed/加粗最终答案，未遵守要求的####格式。
- 保留v1不覆盖，新增MATH_CONTENT_V2_PROTOCOL.json，对所有模型的相同原始回答统一重解析；当前Base为58题明确数值正确、23题明确数值错误、47题仍未解析。它们是本轮零样本诊断计数，不是正式GSM8K5-shot成绩。单位表达、矛盾或不明确答案继续保留未解析；不无条件取末尾数字。
- v2受Base输出格式启发，因此如实标为事后评分修订；没有使用适配模型结果定规则，数学也不进入主要MC预测／patching门槛。
- 统计及真实数据绘图流程已排队等待四份匹配缓存；尚未进行漂移—遗忘比较或激活替换。
- 层级诊断不能单独定位Attention/MLP责任，三个适配端点也不足以用参数更新范数建立跨模型因果解释。
