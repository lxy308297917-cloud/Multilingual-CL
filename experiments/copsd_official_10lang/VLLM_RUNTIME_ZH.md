# vLLM 完整 OPD 运行验证

2026-10-19。本项目运行适配，训练核心仍为已固定的 COPSD 官方源码及已记录的设备兼容补丁。

## 首次真实运行结果

SW／Qwen3-1.7B，micro batch1、梯度累积32、有效batch32、2048生成上限，实际完成2次AdamW更新和64次学生在线生成、固定教师评分及完整蒸馏反传。生成115,788 token，52条达到2048上限，64个microstep损失均有限。原始冻结参数逐位未变，初始化及两次更新后的推理权重同步均通过，长累积中的sleep/wake权重检查通过。

训练部分耗时1053.44秒，包含诊断哈希和同步，均值8.78分钟／更新；仅用这两个早期更新线性外推100更新约14.63小时，**不含模型／引擎冷启动和评测，不能当正式完工承诺或严格控制变量的加速倍数**。

两份checkpoint和最终adapter已保存，但进程在Python退出、CUDA MemPool析构时发生`CUDAPluggableAllocator raw_delete`异常，队列exit=-6。因此首次完整运行验收仍未通过，不能把更新完成marker当作正常完成。详见 `vllm_colocate_probe_20261019.RESULTS.PUBLIC.json`。Torch allocator计数不能用作物理显存峰值，后续验证新增独立nvidia-smi采样。

## 独立修复与恢复验证

新入口 `vllm_colocate_entry.py` 复用原不可变probe中的非破坏性LoRA导出、初始同步、保留权重的sleep/wake及官方训练核心；额外支持真实Trainer checkpoint恢复和100更新运行。它核验恢复后的optimizer step、scheduler step，以及从首次probe checkpoint1加载后的adapter与推理引擎权重身份。最终曝光数按epoch尾部不足batch计算，已对照实际SW／TH的3188题次、TH恢复段948题次核对。

`vllm_ordered_shutdown.py`在训练和最终模型保存结束后显式释放引擎，再按MemPool先于其allocator所有者的顺序释放引用，最后释放分布式资源。参考[vLLM当前源码的release_pools](https://github.com/vllm-project/vllm/blob/main/vllm/device_allocator/cumem.py)；相似退出故障见[上游问题16993](https://github.com/vllm-project/vllm/issues/16993)。安装的0.11已有上游早期callback强引用修复，因此不能声称重新应用该修复即可解决此次异常。独立SW恢复运行已通过本机真实GPU更新及正常退出验证，不修改已安装包、不跳过Python析构、不把异常退出码改为成功。

恢复验证已在独立输出目录完成，从首次probe的checkpoint1恢复到step2，实际32次microstep、59,133生成token，训练579.22秒，checkpoint2保存后exit code0。1秒间隔外部采样的物理显存峰值17,263 MiB（约16.9 GiB）；可能漏掉亚秒峰值。见 `vllm_resume_exitfix_20261019.RESULTS.PUBLIC.json`。恢复时student／推理引擎权重、optimizer step及scheduler step均已核对，但恢复后最终adapter与原probe step2并非逐位相同，因此不声称完整随机轨迹逐位重放。它只验证恢复、权重同步和正常退出；沿用两步调试学习率日程，不能冒充100步正式训练。原失败目录和日志保留。

启动入口：`run_vllm_colocate_17b.sh LANG NEW_RUN_ID 2|100 [CHECKPOINT]`。固定官方损失、AdamW、LoRA64／alpha128、学习率5e-6、有效batch32、2048长度及种子42；源代码、模型锁、各语言数据和运行适配的实际身份清单保留服务器私有配置。每次新输出目录，开启TensorBoard、外部显存采样，按账号四卡上限手动排队，无失败自动重试。

真实恢复更新、checkpoint保存、顺序资源释放及exit code0门槛均已通过，现已用此入口从原始Base启动TE独立100更新正式训练，另行运行TE Base开发集评测。正式训练不接续调试权重。开发集与正式测试仍独立进行；工程通过不代表数学效果有效。
