# COPSD 当前实验进展

更新：2026-10-20 01:23（Asia/Shanghai）。目标仍为十语言独立single OPD，再按独立开发集选择三语言，完成正反顺序OPD与逐阶段遗忘矩阵。以下是当前快照，历史配置、失败与结果保存在各独立记录中。

## 当前任务

| 任务 | 实际状态 |
|---|---|
| SW single | 100/100更新完成，checkpoint100及最终adapter保存，正常退出 |
| TH 四卡single | 100/100更新完成，checkpoint100及最终adapter保存，正常退出 |
| TE single | 已核实16/100更新，已保存checkpoint15，原逐题vLLM训练继续 |
| BN single | 已核实21/100更新，已保存checkpoint20，批量rollout训练继续 |
| JA single | 卡住：4次更新、142个microstep后停滞，尚无checkpoint；进程仍活着，不能计为有效训练 |
| ZH / RU / DE / FR / ES | 批量single配置已准备并CPU预检，未启动 |
| 新vLLM开发评测cohort | 已实际运行，当前语言及英语Base已完成，继续SW／TH checkpoint20／100 |
| SW MGSM-Rev2主评 | Base／固定100更新模型对照已手动排队，尚无正式分数 |
| 三语言顺序OPD | 未达到稳定开发收益与语言选择条件，未启动 |

按账号总计最多四张GPU调度，JA卡住任务仍计入占卡。没有占第五张卡，也不因等待或观察超时重启任务。

## 已完成的同协议开发结果

新协议`project_dev_single_vllm_v1`：固定100道开发题、每题一次、2048生成token、temperature1.0、top_p0.95、无top-k过滤、逐题seed；沿用原官方提示、thinking前缀与数学评分。模型之间问题、答案、prompt、seed及配置除adapter／输出目录外一致。

| 语言 | Base | 已测训练checkpoint | 配对变化 |
|---|---:|---:|---|
| BN | 8/100 | checkpoint5：8/100 | 新增4题、失去4题，净变化0 |
| TE | 2/100 | checkpoint10：0/100 | 新增0题、失去2题，净变化−2 |
| SW | 6/100 | 评测中 | 未完成 |
| TH | 41/100 | 评测中 | 未完成 |
| JA | 39/100 | 无已保存checkpoint | 未完成 |
| 英语 | 40/100 | 保持评测待补 | 未完成 |

BN／TE早期评测父队列已正常退出；不同更新预算不能作正式语言排名，当前没有早期稳定收益证据。BN checkpoint5有96/100条截断，TE checkpoint10有100/100条截断，需要保留这一限制。详见 `bn_dev100_vllm_early_pair_20261020.PUBLIC.json`、`te_dev100_vllm_early_pair_20261020.PUBLIC.json`。

历史HF开发协议另行保留，不能与新vLLM分数混合比较：

| 语言 | Base | checkpoint20 | checkpoint65 | checkpoint100 |
|---|---:|---:|---:|---:|
| SW | 2/100 | 11/100 | 9/100 | 7/100 |
| TH | 37/100 | 29/100 | 未测 | 20/100 |

TH的正常终止从33→22→9，2048截断从67→78→91；已有两处训练checkpoint都低于Base。GlotLID辅助诊断中TH思考区泰语top1为99→100，而缺少结束思考标记从44→84；SW思考区目标语言top1为62→79→97，但数学2→11→7。语言ID不是流利度或数学正确性，不作因果归因。详见 `th_dev100_learning_curve_20261020.PUBLIC.json`、`sw_th_dev100_language_20261020.PUBLIC.json`。

## SW 官方发布脚本对照

AfriMGSM SW test250，原1.7B发布脚本协议：每题12条、每条最多1024 token，原始提示／采样／数学评分函数；Base和SW checkpoint100均正常完成。

| 指标 | Base | SW checkpoint100 |
|---|---:|---:|
| Pass@12 | 34/250（13.6%） | 40/250（16.0%） |
| Average@12 | 90/3000（3.00%） | 74/3000（2.47%） |
| 纯生成耗时 | 13.70分钟 | 16.30分钟 |

Pass@12增加6题，但平均回答正确率下降，单seed不能宣称稳定有效。该1024预算官方对照与项目2048主评分开。AfriMGSM和MGSM-Rev2 SW共享207道目标语言题，不是独立证据。见 `sw_afrimgsm_vllm_pair_20261019.PUBLIC.json`、`VLLM_EVALUATION_ZH.md`。

## 提速、训练预算与验收

固定COPSD源码commit `4f4410730e9b6775a30d20f4b6e141cba2e87590`，第一批固定Qwen3-1.7B revision与tokenizer，LoRA64、有效batch32、AdamW、100更新、2048生成预算。原始Base禁用adapter作固定教师，保持原beta=0带分量裁剪的分布蒸馏损失。1.7B／共同2900题池及兼容适配均单独记录，不声称论文表格数值精确复现。

SW完成3188题次、5,717,658生成token、约97.09 GPU小时。TH完成3188题次、6,273,851生成token，原70更新与恢复30更新共约28.68小时训练时钟、114.73 GPU小时分配成本；不含交接、排队和未提交工作。跨epoch末组20题，因此实际曝光不是3200。见 `sw_completed100_20261019.PUBLIC.json`、`th_completed100_20261019.PUBLIC.json`。

批量rollout只合并同一次optimizer更新内的生成，逐microstep执行原官方拼接、mask、教师前向和完整KD反传；不缩短回答，不更换损失。两更新及checkpoint1恢复验证均已正常退出，Base不变、学生推理同步、保存与恢复通过。批处理／恢复随机轨迹不保证逐位重放。现用于BN正式训练。

BN前10更新共320题次、653,044生成token，checkpoint10恢复文件完整；Trainer时钟22分59秒，约137.9秒／更新，线性粗估100更新约3.83小时，另计冷启动、评测与后续波动，并非最终完工时间保证。教师前向已含在完整训练成本中，尚未单独计时。见 `bn_vllm_update_batch_checkpoint10_20261020.PUBLIC.json`、`VLLM_UPDATE_BATCH_ZH.md`。

## JA异常与授权边界

JA日志停在00:44:28，第4次更新后第142个microstep；检查时GPU利用率0%、显存15,473 MiB，Python约占一个CPU核心，尚无checkpoint。原因未定位。只读栈采集被系统ptrace_scope1拒绝，没有修改系统设置。

已准备并CPU预检从原始Base重新启动JA批量训练的新队列，保留旧日志与目录，尚未启动。AGENTS.md要求取消运行中的队列须获明确授权，因此已请求取消卡住JA后重启；此前“等JA下个checkpoint迁移”的提议被此异常恢复方案替代。TE保存后迁移的授权也仍待答复，原TE继续；BN及评测独立推进。详见 `ja_vllm_stall_20261020.PUBLIC.json`。

## 后续执行

先完成当前配对开发评测及SW MGSM-Rev2正式主评，再逐项使用资源推进ZH／RU／DE／FR／ES；其配置已预检但未启动，见 `remaining_single_batch32_grid_20261020.PUBLIC.json`。未启动全部后续阶段，没有自动失败重试或开机启动。

PolyMath四档补评、英语GSM8K正式保持评测和剩余single正式结果仍未完成。顺序数据接口已有967／967／966个互不重叠原题池，与开发集隔离，十语言英文问题／参考／答案对齐；字符长度不是难度标签。尚未选择三语言或训练顺序模型。后续依据独立开发集、必要的训练seed复测和成本选择，不能用正式测试挑语言、训练题或checkpoint，也不因目前缺少收益而强行进入顺序阶段。
