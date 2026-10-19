# 同一次更新内批量 rollout：独立工程适配

状态：2026-10-20，CPU对照及完整长度GPU两更新通过，父队列exit0；独立断点恢复验证通过并正常退出，已从原始Base启动BN正式100更新。现有TE／JA v1训练使用原冻结配置。

## 实现边界

Transformers Trainer原本在`get_batch_samples`预取本次累积所需的microbatch。新适配读取其中的学生prompt；同一次更新的第一条生成调用让vLLM批量生成整组，后续调用依次取回CPU输出。每题仍n=1，2048生成上限，temperature1.1、top_p0.95、top_k20；教师特权上下文不进入生成prompt。学生权重在该组32次累积反传期间不更新；跨更新缓存、未消费输出与提示错序均拒绝。

原官方生成函数仍逐题执行prompt重新分词、completion拼接、attention mask和labels构建，教师前向、causal shift、beta=0分量裁剪损失、LoRA、AdamW和累积缩放不变。跨epoch尾组20条同样处理。vLLM整组生成后休眠，余下KD阶段不重复唤醒。结束时先拆除控制器绑定引用，再执行已经通过验证的CUDA池释放流程。

这是项目工程扩展，并非声称官方训练入口已内置此功能；批处理可改变随机数消耗和生成轨迹，不能由seed相同推断逐位一致。完整GPU检查通过前不用于新的正式100更新训练。

## 验证与计时

`check_vllm_update_batch_cpu.py`抽取实际冻结官方方法，使用真实Qwen3 tokenizer与确定性的模拟推理输出，比较32和20组共52次input_ids、attention_mask、labels、prompt和completion文本；全部一致。包括2048长度与短输出，检查错序和跨更新失败。此项不执行模型训练，不证明教师参数、GPU显存、吞吐、恢复或方法效果。结果见 `vllm_update_batch_cpu_20261020.PUBLIC.json`。

`UPDATE_BATCH_GENERATION`才是新适配的整组真实生成计时。官方逐microstep打印的缓存命中时间不再代表该题真实推理成本，不可汇总成提速结论。完整训练更新时间、外部每秒显存采样和checkpoint验收另行记录。

待验收：64条完整长度在线轨迹、两次KD更新、当前学生推理同步、固定Base不变、checkpoint保存、正常退出及从checkpoint1恢复。与现有v1比较实际tokens/s和完整更新耗时；未通过前保留为工程检查，不混入正式语言排名。

入口：`run_vllm_update_batch_17b.sh LANG NEW_RUN_ID 2|100 [CHECKPOINT]`。运行环境和原始源码与v1一致，模型与源码、运行适配和数据的实际身份锁保留本机。输出使用独立实验目录，账号最多四卡。SW两更新验证完成后，手动启动了checkpoint1恢复至2的独立验证；不同时铺开新的正式语言实验。

## 两更新实测

64条在线轨迹共113,588 token，两次整组生成分别30.29秒／54,755 token与31.34秒／58,833 token；包括教师／学生前向、反传与诊断同步的训练总计277.05秒，约138.52秒／更新。外部每秒显存采样峰值19,099 MiB。固定Base不变，推理权重逐更新变化，64个microstep损失有限，checkpoint1／2及最终adapter已保存，顺序释放后父队列exit0。独立恢复验收已通过，BN正式100更新已实际启动。短跑线性100更新约3.85小时，仍需正式运行校准；不能把缓存命中打印的数千万token/s当成实测吞吐。见 `vllm_update_batch_probe_20261020.PUBLIC.json`。

## 独立恢复验收

从真实checkpoint1恢复optimizer、scheduler和学生权重，完成step1→2的32次microstep，耗时159.36秒；加载的学生及推理权重均与父step1一致，固定Base未变，checkpoint2及最终adapter一致，顺序清理后父队列exit0。恢复后最终adapter与未中断probe step2非逐位相同，保留此限制；验收范围为功能恢复与在线策略同步，不是确定性逐轨迹重放。见 `vllm_update_batch_resume_20261020.PUBLIC.json`。

BN正式任务使用同一冻结批量适配、相同1.7B revision与tokenizer、共同2900原题池及100更新／2048长度预算，从原始Base重新初始化，输出与验证任务独立。BN Base开发评测已排队；现有TE／JA v1任务不变。
