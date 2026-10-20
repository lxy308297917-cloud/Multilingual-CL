# TE90退出：原生None返回引用缺失与进程局部兼容修复

**进展更新：TE已按用户授权从checkpoint90恢复并完成100更新。** 新段10更新／308题次，恢复状态、最终权重与有序释放通过，父队列exit0；开发仍2→0/100。此修复解决运行缺陷，未解决数学效果。见[恢复验收与成本](te_resume100_completed_20261020.PUBLIC.json)。下面故障定位与初始验证过程保留作历史证据，不再表示正在等待授权。

## 现场与可恢复状态

TE旧逐题vLLM任务在90更新后退出134，错误为 `none_dealloc: deallocating None`，栈落在 `CuMemAllocator.unmap_and_release`／`engine.sleep`。没有日志证据表明此次是CUDA OOM或输入坏题。

已保留原日志和目录，checkpoint90的adapter、optimizer、scheduler、RNG文件完整；CPU实际读取确认optimizer各参数step与scheduler.last_epoch均90，RNG载荷可读。最后一次同步的adapter／推理权重身份保留于私有恢复参考。总分配14.1514 GPU小时包含90次可复用更新，不能全算成空闲浪费，也不能计为完成100更新。

## 已复现的扩展缺陷

本机vLLM0.11.0的 `cumem_allocator.abi3.so` 在 `python_create_and_map`、`python_unmap_and_release` 成功路径返回None时没有增加引用计数。反汇编保留于本机；在CPython3.10上，实际GPU测试的5次unmap／map循环使None引用计数减少10，符合每次原生成功返回少一个引用的机制。

上游[PR49950](https://github.com/vllm-project/vllm/pull/49950)报告了同类abi3编译头文件／运行解释器差异及None返回问题；该PR的示例版本不是本机0.11.0。本机结论依据自身二进制与实际GPU计数测试，不仅依赖上游报告。

该缺陷解释的是长时间反复sleep／wake下的运行中止，**不解释或修复数学准确率退步**。没有故意耗尽None引用以再次崩溃，故障定位时尚未用修复入口恢复TE模型训练；后续实际结果见顶部更新。

## 修复方式与边界

`cumem_none_ref_compat.py`只在新进程内，对已核验哈希的特定二进制和CPython3.10补偿这两个成功返回的None引用。不修改.so、已安装包、共享环境或已运行任务；未知二进制或已有其它patch时拒绝安装。原生异常直接传播，不补偿未成功返回的调用。

验证：

- CPU原生C夹具：未修复5次调用减少5个引用，修复后10,000次调用变化0，异常仍传播。
- 真实vLLM GPU分配器：未修复5次remap循环减少10个引用；修复后100次循环变化0。
- 真实allocator额外3次sleep／wake后，测试tensor值保持一致；显式释放内存池后父队列正常exit0。
- GPU测试只用262,144个float32元素，不加载模型、不运行优化器。账号最多四卡，不修改其它用户任务。

这是小规模原生分配器兼容验证，不能代替完整OPD训练／恢复稳定性或数学效果证据。见 `cumem_none_ref_CPU_20261020.PUBLIC.json`、`cumem_none_ref_GPU_20261020.PUBLIC.json`。

## TE恢复入口

`run_te_resume90_with_dev.sh` 已准备并预检：复用已有TE Base开发结果，从完整checkpoint90恢复权重、optimizer、scheduler、Trainer RNG与采样步位；使用同次更新批量rollout及上述进程局部兼容修复，补齐100更新并执行同协议最终开发评测。

恢复后先核对step90学生adapter和推理权重与原记录一致，optimizer／scheduler仍为90，再执行余下10更新、308个有效训练题次。保持原官方clip0.05、beta0、固定Base教师、AdamW、学习率日程、有效batch32和2048预算；不接入正在运行的无裁剪对照，也不重跑前90更新。

新段单独记录成本、配置、TensorBoard和完整checkpoint100。批量生成与新vLLM进程会改变采样流，不能宣称与旧逐题运行逐位等价；旧初始化限制继续保留。该切换是工程适配参考，不能伪装成从头统一框架的受控训练seed重复。

初次准备时依AGENTS.md请求恢复授权；用户随后明确允许从90接续，已实际完成恢复。当前没有待确认的TE恢复操作；原90步与新10步的成本及采样流差异分别记录。
