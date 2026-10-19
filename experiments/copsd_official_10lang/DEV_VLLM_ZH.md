# 独立开发集的vLLM评测协议

协议名：`project_dev_single_vllm_v1`。固定COPSD原始提示、thinking／目标语言前缀、原数学解析与评分函数；同一模型revision和tokenizer，每题一次、2048生成token、temperature1.0、top_p0.95、无top-k过滤，逐题SHA256(seed:language:id:0)导出seed。

该协议仅用于原来隔离的100道开发题，不含MGSM-Rev2、PolyMath正式测试题，也不从正式测试选题。开发数据中其他字段保留，但学生只得到目标语言问题和官方提示。英语使用相同原题ID的原英文problem，不使用英文参考解答。来源与先前HF开发集一致。

vLLM批处理可能改变随机采样输出；这里重新运行Base与已完成SW／TH模型的配对参考，不把历史HF分数混入新协议排名。旧HF结果、正在运行的TH20评测均保留。JA／BN两个尚未取得GPU的旧HF Base等待队列已停止，新的开发cohort覆盖其Base，避免重复慢评；不取消运行中的训练或评测。

当前一次性cohort按SW、TH、TE、JA、BN、EN顺序评Base，然后按固定顺序评SW checkpoint20／100、TH checkpoint20／100；独立新进程与独立输出目录，每项相同生成和评分协议。尚未运行的其他语言随后配套新增Base参考。所有任务最多合计四张GPU、手动启动、遇错停止且不自动重试。队列等待不代表已生成分数。

CPU对照核对六语言共600个实际开发prompt，与已有HF prompt逐字相同；目标语言数据字段与旧loader一致，英语从原problem读取，问题ID及顺序跨语言一致。此检查不证明随机输出或分数相同，也不证明GPU吞吐。见 `dev_vllm_cpu_compat_20261020.PUBLIC.json`。

代码：`evaluate_dev_vllm.py`、`run_dev_vllm_17b.sh`、`run_dev_vllm_active_cohort.sh`。继续导入冻结的 `evaluate_vllm.py` 中官方函数装载器；私有身份锁验证全部运行适配、官方源码、模型身份与开发数据。所有实际路径和指纹保留本机。

每次保存CONFIG、逐题token／答案／原输出、阶段状态及DONE。DONE分别报告正确率、正常结束、截断、答案提取失败、生成token、模型加载和纯生成耗时。正式测试标记始终为false；不要把开发分数当正式benchmark结果。

## 已准备的TE／JA迁移

`run_vllm_update_batch_resume_checked.sh`和两个独立续训队列已CPU预检，仅配置、未启动。入口要求完整checkpoint及父任务对应step的真实推理同步记录，恢复时强制比对加载的学生／推理权重，同时恢复optimizer、scheduler及RNG。不会把调试权重作为正式起点。

TE／JA原训练继续运行。根据AGENTS.md，停止运行中的队列任务需要用户明确授权，已请求在下一完整checkpoint后迁移；未获授权不停止。若授权，续训写新目录、保留旧产物与分段成本；批处理及恢复的随机轨迹不保证逐位一致，四卡上限和交接竞争仍适用。该授权与开发评测队列推进相互独立。
