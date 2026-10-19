# 正式数学评测的 vLLM 入口

2026-10-19。新增评测入口已经实际启动SW官方协议Base／checkpoint100对照，训练核心和正在运行的训练配置未修改。

## 协议分开保存

| 任务 | 数据与范围 | 采样／预算 | 协议标记 |
|---|---|---|---|
| SW官方评测对照 | 固定AfriMGSM SW test250题 | pass@12，1024生成token | `afrimgsm_release_vllm_v1` |
| 项目主评测 | MGSM-Rev2十语，每语250题 | 单次回答，2048生成token | `project_single_vllm_v1` |
| 项目数学补充 | PolyMath十语，每语500题 | 单次回答，2048生成token；分别报告四档难度 | `project_single_vllm_v1` |
| 英语保持 | GSM8K test1319题，0-shot；可补MGSM-Rev2 EN250题 | 单次回答，2048生成token | `project_single_vllm_v1` |

已有100题独立开发集的Transformers结果保留原协议，不与新增正式测试结果混算。正式测试尚未用于挑选训练题或语言；SW本轮固定使用100更新终点作为同预算参考，20／65更新只作开发集学习曲线。AfriMGSM与MGSM-Rev2的题目重叠情况沿用既有审计，不能当作两份独立测试证据。

## 官方来源与适配范围

官方commit固定为`4f4410730e9b6775a30d20f4b6e141cba2e87590`。对两份官方评测源码和1.7B AfriMGSM启动脚本，已将本机文件的Git blob ID与从该固定commit取得的上游文件逐一比对，三份均一致。本机源目录是无`.git`元数据的固定导出，实际身份锁保留本机。

SW官方协议采用[发布的1.7B AfriMGSM脚本](https://github.com/cisnlp/COPSD/blob/4f4410730e9b6775a30d20f4b6e141cba2e87590/african_langs_eval/run_afrimgsm_one_lang_all_ckpts_1.7b.sh)的参数：n12、1024生成token、temperature1.0、top_p0.95；其余沿用原入口默认top_k=-1、min_p0、presence_penalty0、thinking开启、TP1、显存比例0.9和40960上下文上限。CLI的默认生成预算38912不用于此轮；官方PolyMath脚本另为8192／pass@12，不将项目PolyMath的2048／单次结果称为该配置复现。

当前1.7B／2900训练题／100更新模型属于已记录的项目适配。发布1.7B AfriMGSM脚本标记opsd_500并遍历至50更新，不能把本项目100更新模型称为原论文表格的完全数值复现。

`evaluate_vllm.py`直接从固定源码载入原始prompt、模型加载、答案提取及评分函数。SW官方协议额外直接调用原`evaluate_afrimgsm`、`run_generation`及`process_outputs_and_save`。只跳过源码中的缓存路径占位和CLI main，将原`load_dataset`调用接到冻结本地快照，补充原始token、终止原因与成本记账。LoRA缺失时直接失败，避免误把Base计为训练模型。

官方输出函数返回Average@n，包装器从官方结果中的`pass_at_n`字段取得pass@12；两项指标分别保存。原评分结果复用，不重复运行全部3000条回答的评分。项目单次评测使用同一原始prompt／评分函数，增加本地数据字段适配及固定逐题seed；所有正式模型使用相同新协议。

官方分支保留SamplingParams的seed=None和模型引擎默认seed0。通用CLI的seed42仅用于项目单次分支；官方分支的实际随机策略以配置中的`seed_policy`为准，不把未使用的CLI默认值当作生成seed。

## 已完成与待完成

- CPU核对：十语及英语共11种prompt与原冻结Transformers入口逐字、逐token一致；官方提取／评分函数对SW最终模型已保存的100条回答得到相同7/100，未重新生成。见 `vllm_eval_cpu_compat_20261019.PUBLIC.json`。
- 运行前锁定23份现有benchmark输入、原始源码、模型身份及新评测入口；实际内部指纹仅保存在本机。
- 已手动启动SW AfriMGSM Base→checkpoint100两次独立进程，在同一张分配GPU上顺序运行。模型各自重新初始化，前项失败即停止，无自动重试。当前仍待完整分数与退出状态。
- 其余十语主评测、PolyMath及英语保持使用已实现通用入口，按训练完成与资源逐项推进，不声称已经运行或完成。

## 输出与命令

每次新输出目录，保存`CONFIG.json`、`PHASE.json`、逐题`ITEMS.jsonl`和`DONE.json`；SW官方协议额外保存原格式`OFFICIAL_RESULTS.json`。记录模型加载时间、生成时间、评测总时间、生成token、终止／截断、提取失败及1秒间隔外部显存采样。数学分数与思考／回答语言保持的后续诊断分开报告。

单模型入口：

```bash
bash run_vllm_math_eval.sh BENCHMARK LANGUAGE base|adapter NEW_RUN_ID [CHECKPOINT]
```

固定SW官方对照入口为`run_sw_afrimgsm_vllm_pair.sh`。通过手动GPU队列分配UUID及环境，所有账号任务合并最多四卡。原始逐题输出和完整运行锁留在服务器日志目录；公开记录只发布配置说明、汇总分数与状态。
