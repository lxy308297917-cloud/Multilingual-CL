# SW D2 五方法实验：GPU持续阻塞审计

截至2026-09-23 21:26（UTC+8），连续三个目标回合的真实资源复核均得到`nvidia-smi: No devices were found`。密集FFT原评测PID已经不存在，不能把旧状态文件中的`running`解释为仍在执行。

已完成：五个模型训练；方案一、去保护对照、方案二、无投影对照的全套测评；密集FFT的六项语言任务、IFEval和PPL。

唯一计算缺项是密集FFT的正式GSM8K 5-shot。没有CUDA GPU时按冻结协议无法继续，因此目标进入阻塞状态。GPU恢复后执行：

```bash
/root/miniconda3/envs/cl/bin/python /root/resume_sw_d2_after_reboot.py
```

控制器会通过核验缓存跳过已完成单元，只补GSM8K，然后重建最终五方法报告并执行完成度审计。
