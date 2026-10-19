# 同一次更新内批量 rollout：独立工程适配

状态：2026-10-20，CPU对照通过；完整长度GPU两更新任务已手动排队，尚无GPU验收或速度结论。现有TE／JA v1训练使用原冻结配置。

## 实现边界

Transformers Trainer原本在`get_batch_samples`预取本次累积所需的microbatch。新适配读取其中的学生prompt；同一次更新的第一条生成调用让vLLM批量生成整组，后续调用依次取回CPU输出。每题仍n=1，2048生成上限，temperature1.1、top_p0.95、top_k20；教师特权上下文不进入生成prompt。学生权重在该组32次累积反传期间不更新；跨更新缓存、未消费输出与提示错序均拒绝。

原官方生成函数仍逐题执行prompt重新分词、completion拼接、attention mask和labels构建，教师前向、causal shift、beta=0分量裁剪损失、LoRA、AdamW和累积缩放不变。跨epoch尾组20条同样处理。vLLM整组生成后休眠，余下KD阶段不重复唤醒。结束时先拆除控制器绑定引用，再执行已经通过验证的CUDA池释放流程。

这是项目工程扩展，并非声称官方训练入口已内置此功能；批处理可改变随机数消耗和生成轨迹，不能由seed相同推断逐位一致。完整GPU检查通过前不用于新的正式100更新训练。

## 验证与计时

`check_vllm_update_batch_cpu.py`抽取实际冻结官方方法，使用真实Qwen3 tokenizer与确定性的模拟推理输出，比较32和20组共52次input_ids、attention_mask、labels、prompt和completion文本；全部一致。包括2048长度与短输出，检查错序和跨更新失败。此项不执行模型训练，不证明教师参数、GPU显存、吞吐、恢复或方法效果。结果见 `vllm_update_batch_cpu_20261020.PUBLIC.json`。

`UPDATE_BATCH_GENERATION`才是新适配的整组真实生成计时。官方逐microstep打印的缓存命中时间不再代表该题真实推理成本，不可汇总成提速结论。完整训练更新时间、外部每秒显存采样和checkpoint验收另行记录。

待验收：64条完整长度在线轨迹、两次KD更新、当前学生推理同步、固定Base不变、checkpoint保存、正常退出及从checkpoint1恢复。与现有v1比较实际tokens/s和完整更新耗时；未通过前保留为工程检查，不混入正式语言排名。

入口：`run_vllm_update_batch_17b.sh LANG NEW_RUN_ID 2|100 [CHECKPOINT]`。运行环境和原始源码与v1一致，模型与源码、运行适配和数据的实际身份锁保留本机。输出使用独立实验目录，账号最多四卡。当前只排队SW两更新验证，不同时铺开其他新实验。
