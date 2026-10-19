# COPSD 当前实验进展

更新：2026-10-19 13:49（Asia/Shanghai）。本页仅公开实验摘要，服务器进程、绝对路径与内部指纹保留本机。目标仍为十语言 single OPD，之后依据独立开发集选择三语言进行正反顺序 OPD 与遗忘测量。

| 项目 | 实际状态 |
|---|---|
| SW 正式 single | 95/100 更新，已保存 checkpoint95，继续运行 |
| TH 四卡正式 single | 77/100，新运行最新完整保存为 checkpoint75 |
| 另外八种语言 | 未新启动；等待加速路径验证后扩展 |
| 三语言顺序 OPD | 尚未达到开发集选择条件，未启动 |
| 三个开发集评测 | 已恢复手动排队，尚无新分数 |
| 单卡 vLLM 完整训练检查 | 已手动排队，GPU probe 尚未开始 |

SW checkpoint95已核验保存：实际3028题次、5,449,997生成token，2067条回答触及2048上限；训练时钟92小时30分44秒，分配成本92.51 GPU小时。跨epoch时最后一个累积组只有20题，因此不能用95×32当作实际曝光。adapter、optimizer、scheduler、trainer state与RNG文件均非空，数值日志有限；仅剩5次更新，仍未完成100步或取得新评测分数。见 `sw_checkpoint95_progress_20261019.PUBLIC.json`。

TH 最近完整保存的checkpoint75记录了从checkpoint70恢复后的5次更新、160题次、319,431生成token；145条回答触及2048上限。新训练时钟共90分46秒，均值18分9秒／更新。checkpoint75保存时，按这五次更新粗估当时剩余25次训练约7小时34分，另计评测与耗时波动。checkpoint75的adapter、optimizer、scheduler、trainer state和四rank RNG文件均非空；未重复加载全部optimizer，也未声称新checkpoint75已在GPU上做过恢复测试。证据见 `th_checkpoint75_progress_20261019.PUBLIC.json`。

当前正式训练仍为固定COPSD源码基础上的1.7B／LoRA／固定教师／HF生成适配。独立同卡batch2测速的vLLM生成吞吐约为HF的6.22倍，**不是完整OPD加速倍数**。待运行probe另外验证非破坏性LoRA同步、初始学生同步及长梯度累积中的sleep生命周期，保留官方损失、2048长度与有效batch32，详见 `VLLM_COLLOCATE_PROBE_ZH.md`。

新增probe和待评测均按账号四卡上限排队；既有SW加四卡TH继续使用本轮临时五卡授权，不额外占用新卡。工程运行成功与数学效果分开判断：目前仍只有早期SW独立开发集Base2/100→checkpoint20的11/100，绝对水平较低；没有新增TH或正式测试结果，不能据此宣称十语言或顺序方法有效。
