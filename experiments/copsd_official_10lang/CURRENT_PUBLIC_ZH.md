# COPSD 当前实验进展

更新：2026-10-20 00:36（Asia/Shanghai）。目标保持为十语言 single OPD，依据独立开发集选择三语言，再做正反顺序 OPD 与逐阶段遗忘评测。服务器进程、绝对路径和内部指纹保留本机。

| 项目 | 已核实状态 |
|---|---|
| SW single | 100更新完成，checkpoint100、最终adapter及恢复文件保存，正常退出 |
| TH 四卡 single | 100更新完成，checkpoint100、最终adapter及恢复文件保存，正常退出 |
| TE single | vLLM正式训练运行，已完成10更新并保存checkpoint10；继续训练 |
| JA single | 新vLLM正式训练运行，从原始Base独立开始；00:20完成1更新、44个microstep |
| BN single | 批量rollout正式100更新任务已实际启动，原始Base独立初始化，仍在启动阶段 |
| ZH / RU / DE / FR / ES | 尚未新启动 |
| 开发集 | TE Base已完成4/100；TH20运行；JA Base已手动排队，账号四卡上限下等待 |
| 批量rollout工程验证 | SW两更新完整长度GPU检查通过并正常退出，断点恢复验证完成并正常退出；已用于新BN正式任务 |
| 三语言顺序 OPD | 未达到稳定开发集收益与语言选择条件，未启动 |

## 完成的训练与实际预算

SW：3188题次、5,717,658生成token，2146条触及2048上限，训练时钟约97.09 GPU小时。TH：3188题次、6,273,851生成token，2732条触及2048上限；两段训练时钟约28.68小时，分配约114.73 GPU小时，不含交接、测速、排队与未提交工作。跨epoch末组仅20题，不能按100×32报告实际曝光。详见 `sw_completed100_20261019.PUBLIC.json`、`th_completed100_20261019.PUBLIC.json`。

训练核心沿用固定COPSD版本、LoRA、原始冻结Base教师、在线学生轨迹及原始beta=0带分量裁剪蒸馏损失；模型规模、数据池、预算和设备兼容改动单独记录，不宣称论文表格数值复现。

## 独立开发集：100题，同一HF协议

| 模型 | Base | checkpoint20 | checkpoint65 | checkpoint100 |
|---|---:|---:|---:|---:|
| SW 正确题数 | 2 | 11 | 9 | 7 |
| TH 正确题数 | 37 | 运行中 | 未测 | 20 |

SW后期低于早期checkpoint，不能称单调提升。TH最终比Base少17题正确，配对新增8题、失去25题；2048截断从67/100升至91/100，提示需要排查，但尚无因果结论。题目、答案、提示和逐题seed已核对一致，仅adapter不同。正式100更新模型作为固定预算参考，中间checkpoint用于开发曲线，不根据正式测试挑选checkpoint。详见 `sw_dev100_learning_curve_20261019.PUBLIC.json`、`th_dev100_checkpoint100_comparison_20261019.PUBLIC.json`。

## SW 官方发布脚本评测：AfriMGSM test250

原固定源码1.7B发布脚本协议：thinking、每题12条、每条最多1024 token、temperature1.0；使用原提示、解析与数学评分函数。Base与SW checkpoint100均正常完成，配置除adapter与输出目录外一致。

| 指标 | Base | SW checkpoint100 |
|---|---:|---:|
| Pass@12 | 34/250（13.6%） | 40/250（16.0%） |
| Average@12 | 90/3000（3.00%） | 74/3000（2.47%） |
| 第一个回答正确率（同1024协议） | 6/250（2.4%） | 4/250（1.6%） |
| 生成token | 2,797,919 | 2,787,358 |
| 纯生成时间 | 13.70分钟 | 16.30分钟 |

Pass@12增加6题，平均回答正确率下降；单seed、低绝对水平，不能判定稳定有效。此1024-token官方对照与项目2048-token主评测分开。AfriMGSM SW与MGSM-Rev2有207道相同目标语言题，不能视作独立证据。详见 `sw_afrimgsm_vllm_pair_20261019.PUBLIC.json`、`VLLM_EVALUATION_ZH.md`。

项目MGSM-Rev2的SW Base／checkpoint100同协议正式对照已手动排队；尚无分数。PolyMath四档补评及英语GSM8K保持测评尚未完成。正式测试只作冻结报告，不用于选择训练题、语言或checkpoint。

## vLLM 提速与下一步

单题在线生成约146 token/s，原HF约16 token/s；不同语言完整训练不是严格等长测速。TE前5次完整更新平均约570.3秒，按当时速率外推剩余95更新约15.05小时，另计评测与后续波动。独立恢复、固定Base不变、学生推理同步、保存和正常退出已通过；早期probe析构失败及非逐位重放均保留记录。详见 `VLLM_RUNTIME_ZH.md`、`te_vllm_checkpoint5_20261019.PUBLIC.json`。

新的批量rollout适配仅在同一次optimizer更新内批量生成，随后逐microstep执行原官方拼接、mask、教师前向与KD反传。CPU用实际官方生成方法、真实tokenizer和固定模拟输出核对32题组与20题尾组，52次逐条结果完全一致，错序与跨更新缓存会拒绝；不证明随机轨迹相同或GPU加速。现有TE、JA的冻结v1运行不变。两更新GPU检查已正常完成：64条轨迹、113,588生成token，含诊断同步的完整训练277.05秒，约138.52秒／更新；外部显存采样峰值19,099 MiB。固定Base未变、推理权重随两次更新变化、checkpoint1／2恢复文件完整、最终adapter一致，父队列exit0。短跑线性外推100更新约3.85小时，仅供初步排期，不能当完整100更新实测。checkpoint1恢复至2的独立验证已完成32次microstep，159.36秒，恢复的初始学生与推理权重均与父checkpoint1一致，optimizer／scheduler恢复，Base未变、保存完整、父队列exit0；恢复后最终adapter与原未中断step2非逐位相同，不声称确定性重放。新BN正式100更新已实际启动，从原始Base开始，不接续验证权重；现有TE／JA不迁移。所有队列仍遵守账号四卡总上限，不占第五张卡，不自动重试。详见 `VLLM_UPDATE_BATCH_ZH.md`。

顺序数据接口已准备967／967／966三个不重叠原题池，十语言英文问题、参考与答案对齐，与开发集无交集。原数据无难度标签，字符长度仅作代理。尚未选择三语言，顺序训练与匹配single参考仍需实施。详见 `SEQUENTIAL_POOLS_ZH.md`。

TE Base独立开发集100题完成：4题正确、2题正常终止，204,143生成token，约5679秒。TE训练后尚无配对分数。详见 `te_dev100_base_20261020.PUBLIC.json`；批量GPU检查详见 `vllm_update_batch_probe_20261020.PUBLIC.json`。

批量恢复证据见 `vllm_update_batch_resume_20261020.PUBLIC.json`。BN配套Base开发集已手动排队；JA Base及SW MGSM-Rev2主评仍在四卡上限下等待。工程验收通过不代表数学收益通过，仍需配对开发集和正式主评结果。
