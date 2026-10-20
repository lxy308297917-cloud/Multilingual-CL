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

## 固定开发子集结果（16:40）

三项父队列均exit0，原始题目／prompt／seed、计数、同一长轨迹前缀重评分及完整解码评分已核验。所有数字分母32：

| 语言 | Base2048 | OPD1002048 | Base8192 | OPD1008192 |
|---|---:|---:|---:|---:|
| SW | 1 | 2 | 1 | 6 |
| BN | 1 | 0 | 8 | 0 |
| FR | 9 | 4 | 21 | 12 |

SW长预算下观察到目标语收益；BN在8192仍坍塌，FR仍低于Base。不能用“短预算”统一解释全部退步，也不能由32题单seed宣称SW稳定有效。各模型原训练不变。

**两次生成会提前分叉：** 尽管逐题seed相同，2048／8192生成的公共前缀中位长度仅18.5—160.5 token，不能将两次采样当成严格同一条轨迹延长。为直接测量截断，另对同一8192输出的前2048token评分：SW Base1→1、OPD4→6；BN Base1→8、OPD0→0；FR Base11→21、OPD3→12。箭头表示同一长轨迹的前缀分数到完整分数，均为同32题；该分析不依赖两次采样逐位相同。

全部384条新回答、1,382,226生成token、2569.30秒纯生成、0.75389累计GPU小时；无新训练。早期结果与原正式主表不改写。原评测源码身份为本地Git commit `2aae742`，对应官方来源／实际代码哈希保留在各私有CONFIG；精确源码可由该版本重建。

结果：[SW](sw_length_sensitivity_dev32_20261020.PUBLIC.json)、[BN](bn_length_sensitivity_dev32_20261020.PUBLIC.json)、[FR](fr_length_sensitivity_dev32_20261020.PUBLIC.json)。

下一步将既有损失消融checkpoint按相同8192预算重新评测，确认此前“去裁剪／改方向未改善”的结论是否受短预算影响；在该证据完成前不根据旧短预算对照推广新训练配置。

## 已有消融的8192开发100复核已启动

仅复用现有Base、SW20裁剪、SW20无裁剪、SW20反向KL／无裁剪权重；四组均为同100道独立开发题、同8192上限、n=1及相同逐题seed。统一max_num_seqs16以适配实测约155k token的KV缓存；没有新增训练。四个手动单卡队列已预检并启动，总上限四卡，实际运行与结果待后续验收。见[固定设置及启动](sw_saved_ablation_dev8192_started_20261020.PUBLIC.json)。

## 同一SW训练的长预算阶段曲线已完成

固定独立开发100题、8192预算、n=1，原同一SW训练run的0／20／50／100更新正确数为 **6／16／21／10**。所有父队列exit0，逐题输入／prompt／seed／配置及原始计数均核验，三个checkpoint保存的科学训练参数一致。50→100更新新增2题、失去13题，净下降11题；它说明长预算下仍有后期下降，同时100更新仍高于Base，不能把SW描述为完全未获得能力。最终完整配对数量以结果JSON为准。

这条旧训练曲线与现代SW20裁剪消融是不同run，不混作单条曲线或严格seed对照；原2048结果仍保留。当前20更新独立seed43和BN裁剪开／关诊断继续，尚未证明稳定三语言收益，不进入顺序训练。

见[同一run完整曲线、配对与成本](sw_same_run_8192_curve_20261020.PUBLIC.json)、[验收源码](summarize_sw_8192_curve.py)。本项无新训练，仅新增三个已保存checkpoint的推理，Base缓存复用。

## 完整正式套件的8192入口已实现，仅完成CPU预检

新入口[evaluate_formal8192.py](evaluate_formal8192.py)及[启动脚本](run_formal8192.sh)保留官方模型加载、提示词／目标语前缀、boxed答案提取和评分函数。继续使用原冻结测试输入：十语各MGSM-Rev2全250题、PolyMath全500题（四档各125），英语GSM8K全1319题。21个输入组合、共8819题的CPU预检通过；TE实际checkpoint100的adapter路径／step／身份亦通过CPU预检。未启动此新入口的GPU任务，不能将CPU预检称为GPU验证或新正式成绩。见[实际预检清单](formal8192_CPU_preflight_20261020.PUBLIC.json)。

协议名为`project_formal_single_vllm8192_v1`：8192生成上限、n=1、temperature1、top_p0.95、top_k−1、逐题seed42规则、max_num_seqs16、context40960。8192来自官方PolyMath发布脚本；n=1／逐题seed／本项目数据适配单独标记，不称官方pass@12。Base、single和后续阶段使用同一新协议。旧2048正式结果与官方AfriMGSM1024／pass@12均保留，不与本协议混合。

先在独立开发题冻结配置和checkpoint，再生成正式报告；不得依据正式分数挑checkpoint或选语。这些仍是先前使用过的测试输入，不能把改变预算后的报告描述为一套从未查看过的新独立测试集。当前开发筛选及JA训练优先继续，尚未开启全十语8192正式评测。

只做CPU预检的命令示例：

```bash
COPSD_FORMAL_PREPARE_ONLY=1 bash experiments/copsd_official_10lang/run_formal8192.sh mgsm_rev2 sw sw_formal8192_new_unique_id
```

实际执行须由单卡队列提供GPU UUID及账号合计四卡限制，并保留日志、GPU遥测、逐题原始token及配置。Base无需checkpoint参数；adapter在run ID后增加checkpoint目录和已冻结step。输出使用独立`evaluation_vllm8192`目录，不覆盖开发或旧正式结果。
