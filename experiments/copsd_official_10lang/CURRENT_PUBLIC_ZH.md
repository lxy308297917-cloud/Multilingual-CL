# COPSD 当前实验进展

更新：2026-10-20 04:57（Asia/Shanghai）。目标仍为十语言独立single OPD，再按独立开发集选择三语言，完成正反顺序OPD与逐阶段遗忘矩阵。以下是当前快照，历史配置、失败与结果保存在各独立记录中。

## 当前任务

| 任务 | 实际状态 |
|---|---|
| SW single | 100/100更新完成，checkpoint100及最终adapter保存，正常退出 |
| TH 四卡single | 100/100更新完成，checkpoint100及最终adapter保存，正常退出 |
| TE single | 已核实39/100更新，已保存checkpoint35；原逐题vLLM训练继续 |
| BN single | 100/100更新正常完成；最终开发Base8/100→训练后0/100 |
| JA single | 卡住：4次更新、142个microstep后停滞，尚无checkpoint；进程仍活着，不能计为有效训练 |
| ZH single | 已核实82/100更新，已保存checkpoint80；批量rollout训练继续 |
| RU / DE / FR / ES | RU等待监控已停止，先排查BN退步；DE／FR／ES仅配置，未启动 |
| 新vLLM开发评测cohort | 六种Base及SW／TH checkpoint20／100全部完成，父队列exit0 |
| SW MGSM-Rev2主评 | Base8/250（3.2%）→100更新9/250（3.6%），父队列exit0 |
| 三语言顺序OPD | 未达到稳定开发收益与语言选择条件，未启动 |

按账号总计最多四张GPU调度，JA卡住任务仍计入占卡。没有占第五张卡，也不因等待或观察超时重启任务。

## 已完成的同协议开发结果

新协议`project_dev_single_vllm_v1`：固定100道开发题、每题一次、2048生成token、temperature1.0、top_p0.95、无top-k过滤、逐题seed；沿用原官方提示、thinking前缀与数学评分。模型之间问题、答案、prompt、seed及配置除adapter／输出目录外一致。

| 语言 | Base | 已测训练checkpoint | 配对变化 |
|---|---:|---:|---|
| BN | 8/100 | checkpoint5：8/100；100：0/100 | 最终新增0题、失去8题，净变化−8 |
| TE | 2/100 | checkpoint10：0/100 | 新增0题、失去2题，净变化−2 |
| SW | 6/100 | checkpoint20：11/100；100：5/100 | 分别+5、−1题 |
| TH | 41/100 | checkpoint20：26/100；100：17/100 | 分别−15、−24题 |
| JA | 39/100 | 无已保存checkpoint | 未完成 |
| 英语 | 40/100 | 保持评测待补 | 未完成 |

两批新vLLM开发评测父队列均已正常退出；不同更新预算不能作正式语言排名，当前没有早期稳定收益证据。BN checkpoint5有96/100条截断，TE checkpoint10有100/100条截断，需要保留这一限制。详见 `bn_dev100_vllm_early_pair_20261020.PUBLIC.json`、`te_dev100_vllm_early_pair_20261020.PUBLIC.json`。完整新协议cohort见 `vllm_dev_active_cohort_20261020.PUBLIC.json`：共1000条回答、1,838,900生成token，纯生成1036.92秒；SW早期增益未保持到100更新，TH两处checkpoint均低于Base，当前不满足稳定三语言选择条件。

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

当前配对开发评测及SW MGSM-Rev2正式主评已完成。新v2入口已补上初始化前seed和有效token／教师成本记录；ZH已从Base独立启动，RU／DE／FR／ES后续逐项推进；其余四种配置已预检但未启动，见 `remaining_single_batch32_grid_20261020.PUBLIC.json`。未启动全部后续阶段，没有自动失败重试或开机启动。

PolyMath四档补评和剩余single正式结果仍未完成。Base／SW100／TH100英语GSM8K1319题0-shot正式保持对照已全部完成，后续SW／TH主评与PolyMath补评按同一单卡队列继续；冻结报告用途，不用于选语言或checkpoint。顺序数据接口已有967／967／966个互不重叠原题池，与开发集隔离，十语言英文问题／参考／答案对齐；字符长度不是难度标签。尚未选择三语言或训练顺序模型。后续依据独立开发集、必要的训练seed复测和成本选择，不能用正式测试挑语言、训练题或checkpoint，也不因目前缺少收益而强行进入顺序阶段。

## 新完成的正式主评与教师诊断

MGSM-Rev2 SW test250，固定100更新参考与Base均按`project_single_vllm_v1`完成：8/250→9/250，净增1题（新增7题、失去6题），正常结束54→190，截断196→60。回答结束改善没有形成明显数学收益。题目／答案／prompt、逐题seed规则及配置除adapter／输出外相同，父队列exit0；正式测试只作报告，不用于挑checkpoint、语言或训练题。见 `sw_mgsm_rev2_vllm_pair_20261020.PUBLIC.json`。

真实SW教师信号两更新诊断已正常完成：64条在线轨迹、113,068有效回答loss位置，Base不变、保存与退出通过。教师argmax项被裁剪占4.43%，被裁剪教师概率质量约3.57%；这些局部统计不能证明终点退步原因。教师forward64次，CUDA事件合计16.09秒，学生forward15.38秒，均不含生成、KL和反传。诊断仍返回原损失，没有启动无裁剪消融。见 `teacher_signal_probe_20261020.PUBLIC.json`、`LOSS_DIAGNOSTICS_ZH.md`。

复现记录补充：两次同样声明Trainer seed42的短跑，初始LoRA参数指纹不同，但初始推理策略均等于同一Base。固定库在创建PEFT后才进入Trainer设置seed，需在新入口中显式提前设置；现有运行及旧结果不改写。该修正已在新v2入口实施并CPU验证，完整GPU初始策略已核对仍等于原始Base；不宣称完整训练逐位重放。见 `lora_initial_seed_observation_20261020.PUBLIC.json`。

## ZH新v2正式链路

ZH Base开发100题25题正确，185,308生成token、纯生成72.15秒。随后从原始Base独立开始100更新，实际GPU初始推理权重与此前核验的原始Base一致，已有有限loss的真实microstep。新入口在模型／PEFT初始化前调用实际seed42，读取原shifted labels记录有效回答位置，分别记录teacher／student forward CUDA时间；不更换损失或优化器。600秒watchdog只打印堆栈，不杀进程或自动重试。训练正常完成后在同一分配GPU上接最终100题开发评测。见 `zh_seeded_cost_launch_20261020.PUBLIC.json`、`SEEDED_COST_RUNTIME_ZH.md`。

ZH checkpoint5成本文件已实际验收：160次loss调用，295,590有效回答位置（等于本段生成token），教师forward39.20 CUDA秒、学生36.50秒，恢复文件齐全。根据保存参数与固定sampler重建的160题曝光前缀，其四项实际forward位置总量均吻合；原题ID清单保留本机私有文件，不把重建清单称为逐条运行时记录。见 `zh_seeded_cost_checkpoint5_20261020.PUBLIC.json`。

## 03:03 调度与成本补充

BN checkpoint65已保存完整恢复文件，ZH checkpoint25的800题次已记录1,446,755有效回答位置、772.49秒整组生成、197.72秒teacher forward CUDA时间；计时范围不同，不能将其相减推算全部剩余成本。两项训练仍运行，尚无其100更新终点评测结果。

RU已手动启动单次排队，仍等待账号四卡上限内的空位，没有分配第五张GPU。取得资源后依次执行Base开发评测、从原始Base独立100更新、最终开发评测；DE／FR／ES仍仅配置，未启动。等待中的英语保持队列已接入BN完成条件检查，满足时先评BN最终开发集，再做原英语保持对照。见 `ru_seeded_single_queue_20261020.PUBLIC.json`、`zh_seeded_cost_checkpoint25_20261020.PUBLIC.json`、`EVALUATION_HANDOFF_ZH.md`。此前GitHub传输异常已恢复，这两项记录已补传。

## 04:20 BN完整训练已验收

BN批量vLLM正式100更新正常完成，父队列exit0：实际3188题次、6,523,858生成token，训练时钟13,402.43秒（3.72小时），平均134.02秒／更新，整组生成累计4,191.21秒。训练时钟包含教师／学生计算、反传、优化器和诊断同步，不含冷启动与下游评测；不是单题短跑外推，也不是跨语言受控加速比。每秒采样的单卡显存峰值20,537 MiB。

固定Base检查通过，101个策略同步点齐全，学生与推理权重逐更新变化；3188个microstep损失均有限，checkpoint100恢复文件完整，最终adapter与checkpoint100一致，运行时有序释放及父队列正常退出均通过。旧v1初始化前未显式设置PEFT seed的限制保留，不能与新v2混称确定性初始化。最终BN开发结果尚未产生，工程完成不能当作数学提升。原评测队列仍活跃等待资源，入口会先接BN最终开发评测。见 `bn_completed100_20261020.PUBLIC.json`。

## BN最终开发结果及正式评测交接

BN100更新开发评测已完整生成和评分，并成功返回原父脚本，后续英语GSM8K Base1319题正式评测已实际开始生成。BN最终开发与Base的题目、答案、prompt、逐题seed及配置除adapter／输出目录外全部一致：8/100→0/100，新增0题、失去8题；训练后100条均触及2048上限，100条均无可提取的boxed答案，只有2条含结束思考标记（Base22条）。这是固定协议下的退步与未正常给出最终答案，不能据此断言每一步内部推理均错误，也不能凭截断统计断定原因。见 `bn_dev100_vllm_final_pair_20261020.PUBLIC.json`。

SW、TH、BN已有的最终开发结果均未提供稳定增益证据，目前不选择三语言或强行启动顺序训练。ZH及TE按冻结配置继续，RU等待；数学效果与已验证的3.72小时工程速度分开报告。

## 04:35 BN生成内容诊断与下一步

对既有逐题输出的CPU复核显示，Base／checkpoint5／checkpoint100含Unicode十进制数字的回答为94／93／0条，数字总数5234／4596／0，含所选LaTeX数学命令为49／46／0条。最后100条不仅未给出可提取答案，生成内容也发生明显变化；这些统计不是独立数学或流利度分数，不证明因果，也不能覆盖用文字表达的数值。结果与可复算脚本见 `bn_dev100_output_content_20261020.PUBLIC.json`、`audit_dev_math_content.py`。

为先排查而不是继续铺开退步实验，尚未分配GPU的RU等待监控已停止；未取消任何运行中的训练。原始冻结1.7B在官方学生前缀／英文参考特权教师前缀下的100题条件生成检查已手动排队，最多四卡，目前waiting、没有GPU结果。输入直接由固定官方collator构造，CPU已核对100对前缀均未截断，源码和私有输入另行冻结。教师条件看到参考解答，不能当公平benchmark分数，也不代表所有on-policy位置的教师信号；不制作SFT标签、不暗改损失。见 `BN_FAILURE_ANALYSIS_ZH.md`、`bn_teacher_context_prepared_20261020.PUBLIC.json`。

## 英语GSM8K正式保持对照已完成

固定1319题、0-shot、每题一次、2048生成上限及`project_single_vllm_v1`协议；三模型全部完成并成功返回原父脚本，已接SW PolyMath正式评测。

| 模型 | 正确题数 | 正确率 | 相对Base |
|---|---:|---:|---:|
| Base | 1113/1319 | 84.38% | — |
| SW100 | 1056/1319 | 80.06% | −4.32个百分点 |
| TH100 | 1068/1319 | 80.97% | −3.41个百分点 |

SW相对Base新增58题、失去115题；TH新增58题、失去103题。题目、答案、prompt及配置除adapter／输出目录外相同，冻结评分源码与逐题seed规则相同；正式ITEMS没有独立保存sample seed字段，因此这里不声称从样本字段直接验过实际seed值。两项训练均存在英语数学下降，单seed保持对照不等于顺序遗忘实验，不用于挑语言或checkpoint。见 `english_gsm8k_completed_singles_20261020.PUBLIC.json`。ZH与TE仍运行，教师条件检查仍等待四卡上限内资源，RU等待监控已停止。
