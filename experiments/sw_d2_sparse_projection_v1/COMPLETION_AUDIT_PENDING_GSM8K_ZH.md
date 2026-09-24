# SW D2 五方法完成度审计（仅待密集FFT GSM8K）

协议：`sw_d2_sparse_projection_seed42_v1`。本审计以当前DONE、哈希、只读核验器和真实进程/GPU状态为准。

## 要求逐项核验

|要求|证据|结论|
|---|---|---|
|五个模型使用冻结D2数据与配置|`TRAINING_RELEASE.json`锁定五配置、代码、Base、数据与校准/投影基|通过|
|五个模型完成15.36M输入token训练|五个`TRAINING_DONE.json`均为22,146步、15,360,372输入token|通过|
|方案一及去保护对照完整评测|各自`evaluation/*/DONE.json`与核验缓存|通过|
|方案二及无投影对照完整评测|各自`evaluation/*/DONE.json`与核验缓存|通过|
|同协议密集FFT完整评测|语言六项、IFEval、PPL完成；GSM8K缺失|未通过：唯一缺项|
|PPL/NLL诊断|五模型均有分数，且报告中单列、不进入能力聚合|通过|
|配对SW阅读差值与CI|四完整方法及密集FFT已有900题/488文章、10,000次重抽样结果|通过|
|最终中文全量报告|阶段报告已交付；正式报告尚不能纳入缺GSM8K的密集FFT|待GSM8K后封口|

## 当前确切状态

- 五个模型训练全部完成。
- 前四个新方法全套测评完成。
- 密集FFT已完成：SW/EN摘要、双向翻译、SW/EN Belebele、IFEval、PPL。
- 唯一缺项：密集FFT的正式GSM8K 5-shot。
- `nvidia-smi`当前返回`No devices were found`；旧PID 12881不存在。

## 恢复流程

GPU恢复后执行：

```bash
/root/miniconda3/envs/cl/bin/python /root/resume_sw_d2_after_reboot.py
```

恢复控制器使用实时核验器，已完成单元会幂等跳过，只补GSM8K；随后自动运行`summarize_results.py`。最终完成必须再确认：密集FFT `evaluation/DONE.json`存在、核验器`complete=true`、五方法均进入正式CSV/报告且无FAILED。

机器可读证据：`COMPLETION_AUDIT_PENDING_GSM8K.json`。
