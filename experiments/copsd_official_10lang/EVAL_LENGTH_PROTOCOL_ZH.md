# 训练生成长度与评测长度：协议差异和配对诊断

固定官方commit `4f4410730e9b6775a30d20f4b6e141cba2e87590`：

| 来源 | 生成上限 | 采样 |
|---|---:|---:|
| PolyMath发布评测脚本 `polymath_eval/run_eval_4b_all_checkpoints.sh` | 8192 | pass@12 |
| AfriMGSM 1.7B发布评测脚本 | 1024 | pass@12 |
| 当前项目共同开发／正式报告 | 2048 | n=1 |
| 官方训练发布设置 | 2048 | 每题一次在线rollout |

PolyMath评测器Python CLI默认38912，与发布shell脚本也不同。本次长度诊断采用发布shell的8192，不能把训练2048直接当成官方PolyMath评测预算。

此前八语言正式表一直标记项目n=1／2048。相同项目协议下的下降是真实观测，但高截断率可能影响模型比较，不能直接外推为官方8192／pass@12下方法有效或无效。原正式输出和成绩保留，不能静默替换。

## 已启动的独立开发长度诊断

固定SW／BN／FR开发清单前32题，分别加载原Base和原100更新adapter。每个模型在同一进程内重新跑2048与8192两种预算；两者均为32题，提示、模型、采样与逐题seed相同。重跑2048对照，避免把缓存100题批量与新32题批量的差异当成长度作用。

保存逐题token，检查长预算输出是否保留短预算的生成前缀；不能仅凭seed相同假定逐位一致。报告正确数、正常结束、截断、提取失败、token和耗时。三语言使用独立手动单卡队列，与TE合计最多四卡；不新增训练、不改TE正在使用的原评测文件。

这是开发子集上的探索性长度敏感性分析，n=1，尚不构成官方pass@12复现或新的正式排名。下一步正式协议须依据该诊断明确冻结，不能同时混用两种预算比较模型。

入口：[配对评测](evaluate_dev_length_sensitivity.py)、[启动脚本](run_dev_length_sensitivity.sh)。

官方来源：[PolyMath发布脚本](https://github.com/cisnlp/COPSD/blob/4f4410730e9b6775a30d20f4b6e141cba2e87590/polymath_eval/run_eval_4b_all_checkpoints.sh)、[PolyMath评测器](https://github.com/cisnlp/COPSD/blob/4f4410730e9b6775a30d20f4b6e141cba2e87590/polymath_eval/evaluate_math.py)。
