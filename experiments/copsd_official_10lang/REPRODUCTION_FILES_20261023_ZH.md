# COPSD 实验源码与配置索引（2026-10-23）

本次在用户明确允许发布实验源码和配置后，补齐此前仅本地提交的文件。发布范围和文件哈希见 [PUBLICATION_SOURCE_MANIFEST_20261023.PUBLIC.json](PUBLICATION_SOURCE_MANIFEST_20261023.PUBLIC.json)。未包含凭据、PRIVATE 身份清单和服务回执、原始数据、逐题生成文本或模型权重。

## 来源与实验边界

- 官方源码固定于 [cisnlp/COPSD@4f441073](https://github.com/cisnlp/COPSD/tree/4f4410730e9b6775a30d20f4b6e141cba2e87590)。已有兼容补丁位于 [patches/](patches/)，环境冻结文件保留在本目录。
- 原裁剪 BN4B20：固定原始 Base 教师、LoRA、AdamW、beta=0、词表分量 clip=0.05、完整2048回答预算。双角色设备、micro batch 与发布启动器有已记录的适配，不称严格硬件复现。
- 无裁剪 BN4B20：使用官方支持的 `--jsd_token_clip 0`，入口将其映射为 `None`；作为单因素干预单列，不能混称发布默认结果。
- 新合并服务评测与旧 native16 评测各自匹配 Base 和学生；既有输出不同，不跨协议混比，不把推理提速当作数学收益。

## 主要入口

| 用途 | 源码／配置 |
|---|---|
| 原裁剪 BN4B20 训练 | [bn4b_server20_entry.py](bn4b_server20_entry.py)、[bn4b_server20_job.py](bn4b_server20_job.py)、[run_bn4b_server20.sh](run_bn4b_server20.sh)、[冻结配置](bn4b_server20_frozen_20261023.PUBLIC.json) |
| 原裁剪固定开发评测 | [evaluate_bn4b_followup200.py](evaluate_bn4b_followup200.py)、[collect_bn4b_followup200.py](collect_bn4b_followup200.py)、[评测配置](bn4b_eval_frozen_20261023.PUBLIC.json) |
| 无裁剪 BN4B20 训练 | [bn4b_unclip20_entry.py](bn4b_unclip20_entry.py)、[bn4b_unclip20_job.py](bn4b_unclip20_job.py)、[run_bn4b_unclip20.sh](run_bn4b_unclip20.sh)、[冻结配置](bn4b_unclip20_frozen_20261023.PUBLIC.json) |
| 无裁剪放行／评测／三组比较 | [release_bn4b_unclip20_checkpoint20.py](release_bn4b_unclip20_checkpoint20.py)、[evaluate_bn4b_unclip20_followup200.py](evaluate_bn4b_unclip20_followup200.py)、[collect_bn4b_unclip20.py](collect_bn4b_unclip20.py)、[评测配置](bn4b_unclip20_eval_frozen_20261023.PUBLIC.json) |
| 手动启动的一次性训练后衔接 | [bn4b_unclip20_series.py](bn4b_unclip20_series.py)，无开机启动或失败自动重试 |
| 新合并服务完整配对评测 | [evaluate_canonical_merged_bn4b.py](evaluate_canonical_merged_bn4b.py)、[canonical_merged_bn4b_eval_job.py](canonical_merged_bn4b_eval_job.py)、[汇总](collect_canonical_merged_bn4b.py)、[配置](canonical_merged_bn4b_eval_frozen_20261023.PUBLIC.json) |
| 单卡服务初始化与吞吐基准 | [trl_server_eval_direct_lora_worker32.py](trl_server_eval_direct_lora_worker32.py)、[benchmark_direct_single_gpu_server4b.py](benchmark_direct_single_gpu_server4b.py) |
| 实际权重加载／前缀概率诊断 | [check_actual_server_student_4b.py](check_actual_server_student_4b.py)、[check_actual_server_merged_hf_4b.py](check_actual_server_merged_hf_4b.py) |
| 顺序数据接口准备 | [prepare_sequential_3pool_interface.py](prepare_sequential_3pool_interface.py)，不表示已选三语言或开始顺序训练 |

## 复现前须重建的本机绑定

这些文件记录实际执行源码与当次冻结参数，包含本机相对目录、模型 revision、数据与运行时源码哈希；不是直接下载即可运行的通用安装包。新机器须从官方发布数据重建身份清单、准备相同模型和环境、生成本机队列与模型绑定，并重新做必要预检。PRIVATE 文件和路径引用仍用于追溯，本次不上传其原始内容。旧输出目录不能直接重用或覆盖。

检查存储时同时查看 `/data0` 与 `/data1/lixiaoyang`，解析入口软链接后检查实际落盘路径。可在统一 checkpoint 入口下使用指向本账号 `/data1` 的软链接；须在新任务启动前配套更新路径预检及冻结配置，不能修改正在运行的冻结任务。

最新效果和执行状态见 [CURRENT_PUBLIC_ZH.md](CURRENT_PUBLIC_ZH.md) 与[今日日报](../../docs/daily/实验日报_2026.10.23.md)。训练完成、保存文件验收、实际进程恢复和方法有效是分别记录的证据。
