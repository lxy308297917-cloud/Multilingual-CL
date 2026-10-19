# 训练完成后的评测交接（2026-10-20）

已授权且等待资源的英语保持评测队列改用 `run_english_retention_with_ready_dev.sh`；配置与 CPU 预检通过，没有重新启动监控或中断现有训练。

取得单卡后，入口检查 BN 训练父队列是否正常完成、checkpoint100 的 global step 与 adapter 是否齐全。满足时先运行固定100题 BN 最终开发评测，再执行原 Base／SW100／TH100 英语 GSM8K1319题保持对照；BN 尚未完成则直接执行原英语评测，不占着 GPU 等训练。英语正式结果只用于冻结报告，不用于选语言或 checkpoint。BN 的 Base 已完成8/100，最终开发结果尚未产生。

新增 `verify_cached_development.py` 核验缓存 Base 的协议、100道题与答案顺序、逐题 seed、代码与数据身份、依赖版本及分数／token 总量。JA 的既有 Base39/100已实际通过，不重复生成。`run_seeded_single_reuse_base.sh` 将此核验接入新的 seeded-cost 批量训练及最终开发评测。JA 新队列仅配置，尚未启动；旧 JA 停滞任务尚未取消，仍待此前所请求的明确授权。

`run_seeded_cost_resume_checked.sh` 保留完整 checkpoint 和父策略同步记录检查，切换到新成本记录入口；未用于迁移正在运行的 TE。恢复后的成本仅覆盖新进程片段，不能补称旧段已有统计，也不保证随机轨迹逐位重放。

03:00 左右核对：BN、ZH、TE 的真实训练进程存活；BN 已显示63次更新、ZH24次、TE26次，日志继续更新。JA 进程仍活但没有新的训练进展。全账号仍按最多四张卡调度。未启动顺序训练，未改动官方蒸馏损失。

## 正式补评入口

等待中的原英语队列已扩展为 `run_completed_singles_formal_suite.sh`，仍只分配一张GPU。先执行上述BN完成条件检查与英语保持对照，再补SW Base／100更新的PolyMath500题、TH Base／100更新的MGSM-Rev2及PolyMath500题。PolyMath每种语言low／medium／high／top各125题，保留难度字段以分别报告。SW已完成MGSM-Rev2不重跑。均使用既定`project_single_vllm_v1`单次回答协议，不混称官方pass@12。

配置前已确认队列仍waiting、无运行中的冻结job；shell语法及输出目录不重复检查通过，队列validate通过。没有新增GPU、启用开机启动或故障重试，也没有修改运行中的训练代码。新增正式补评尚未实际开始；任何阶段失败即停止并保留此前产物，不自动重跑。
