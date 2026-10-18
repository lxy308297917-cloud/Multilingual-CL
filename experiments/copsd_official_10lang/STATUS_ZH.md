# COPSD 官方源码十语言实验状态

更新时间：2026-10-18 14:03（Asia/Shanghai）。本轨道执行用户新目标：十语言独立 COPSD，再以独立开发集选择三语言做两条相反顺序的持续 LoRA OPD。此前自写 FFT、SFT 标签批量合成与自写 OPD 均为旧协议，不计入本轨道。以下 10 月 16 日段落保留为历史快照；当前状态以紧接的 10 月 18 日段落为准。

## 2026-10-18 当前进展

- SW 1.7B single 仍在 GPU 0 运行，已保存至 checkpoint-65。14:03 日志中约 2201 条在线 rollout、4,005,418 个生成 token，纯生成累计 68.04 小时；最近 100 条平均 98.8 秒/题，约 71.5% 的全部回答触及 2048 token 上限。100 步目标与正式测试尚未完成。队列 `copsd-sw-17b-project2900-formal-v1` 为 `running`，对应 PID 1933369。
- 同一 100 题独立开发集和同一单次生成协议下，Base 2/100、正常终止 9/100；SW checkpoint-20 11/100、正常终止 43/100。两项队列于 10 月 16 日分别以退出码 0 完成，原始逐题输出、配置和 `DONE.json` 在 `logs/copsd_official_10lang/evaluation/sw_17b_dev100_{base,ckpt20}_v1/`。这是项目开发集早期改进信号，不能代替正式 MGSM-Rev2、PolyMath 或官方 AfriMGSM 成绩，也不能据此单独宣称十语效果。
- 隔离的 micro batch 2 分块 KL 单步探针于 10 月 16 日完成，生成两条各 2048 token 用时 138.61 秒、整步 144.77 秒，未 OOM。日志同时给出 decoder-only **右侧填充警告**；不同长度提示词的批量生成可能受填充影响，因此该步仅证明显存和计算链路能运行，29.6 token/s 不能直接当作可采用的正确提速结果。正式 SW 与新语言继续使用未改 trainer 的 micro 1；左侧填充与教师/学生位置及 mask 修正须在独立探针验证后才能作为新协议运行。
- 用户于 14:00 明确要求合计占用四张空闲 GPU 推进 OPD。启动前确认本账号仅 SW 占 GPU 0，GPU 1/6/7 是其他用户任务，GPU 2/3/4/5 空闲；没有操作他人进程。将 TE、JA、RU 三个先前停止且未训练的命名队列 `max_user_gpus` 从 3 改为默认上限 4，逐个通过 `validate` 并手动 `start`。14:03 确认 TE、JA、RU 分别在 GPU 2/3/4 以 PID 3537260/3537358/3537622 运行，连同 SW 共四张本账号 GPU；GPU 5 留空。三个语言从同一原始 Qwen3-1.7B revision 独立初始化、共享原题集合与 100 步预算，使用原单卡官方 trainer 适配路径，日志与 checkpoint 各自独立。队列不设开机自启、失败自动重试；当前尚无三语有效训练结果。

## 2026-10-16 实时进展

- SW 1.7B single 正式任务继续在 GPU 0 运行，队列 `copsd-sw-17b-project2900-formal-v1` 于 16:15 状态为 `running`，有对应 GPU 计算进程。最新已落盘 checkpoint-20（100 次优化更新目标的 20 次）；日志至 16:16 已完成 730 条在线 rollout，共生成 1,312,571 token，生成阶段累计 22.70 小时；547/730 条触及 2048 token 上限。最后 100 条均值约 107.5 秒/题、1,718.9 token/题。按当前吞吐仅粗估剩余约 3 天生成计算，另有反传、保存和评测；学生轨迹长度可能变化，完成时间不能保证。loss 变化和 4B 八题 smoke 均不是方法有效的证据，尚无 1.7B 开发集完整 Base/训练模型对照分数。
- 单卡 micro batch 2 短测 `copsd-sw-17b-batch2-probe-v1` 于 2026-10-15 20:02 失败，退出码 1。官方完整词表 KL 的 `F.kl_div` 需要再分配 2.32 GiB 时 GPU 仅余 1.30 GiB，CUDA OOM；该配置不能直接用于本机 24GB 单卡正式训练。失败日志保留，不重复运行，也未影响 micro batch 1 正式任务。
- 16:15 手动启动 SW 100 题开发集 Base 与 checkpoint-20 两条相同协议的评测队列；两者当前均等待空闲 GPU。它们使用独立开发题、单次回答和同一 2048-token 上限，不用正式测试集调配置。评测结果完成前不宣称数学提升。
- 用户要求扩展后，TE、TH、BN、ZH、JA、RU、DE、FR、ES 九个 single 队列已全部单独配置、CPU 预检并于 16:15 手动启动；每个队列单卡、相同 Qwen3-1.7B revision、同一官方训练代码和 2900 条对应题、micro 1 × 累积 32、100 次更新、独立日志与 checkpoint 目录。当时九个均处于 `waiting`，不占 GPU：GPU 0 为本账号 SW，其他七卡均有计算进程。九条新训练队列将本账号并发阈值显式设为 3 张，为评测保留第四张卡的账号容量；实际占用仍以 GPU 空闲和队列复查为准，不占或中断其他用户任务。不设开机启动、故障自动重试。
- MGSM-Rev2 十语单次回答正式测评入口 `run_mgsm_rev2_17b.sh` 已补齐，直接调用现有固定提示和评分器；十个本地 TSV 均以同一加载器验证为 250 条、题目与答案非空，shell 语法预检通过。正式测试不会先于开发集选择拿来调参，当前也没有为此额外占用 GPU。
- PolyMath 补充测试输入已从本地四档 `low/medium/high/top` Parquet 固化为每语言 500 题 JSON，十语逐题答案与英文对应项一致；源文件与输出 SHA256 在 `data/copsd_official_20261015/polymath_project_eval_v1/MANIFEST.json`。项目单次回答入口 `run_polymath_project_17b.sh` 已通过 CPU 加载和语法预检。它与官方默认单档、vLLM 多次采样结果分开报告，目前尚未启动 GPU 测评。
- 用户进一步指出现有十语训练墙钟耗时不可接受后，于 16:28–16:29 停止上述九个**尚未占卡**的等待队列，九个状态均为 `stopped`；配置保留，均未发生训练和产出 checkpoint。SW 正式训练与两条 SW 开发集评测等待队列不受影响。742 条 rollout 的实测生成约 23.07 小时、最近 100 条约 15.97 token/s；详细瓶颈、方案和三篇方法比较见 `ACCELERATION_AND_METHOD_REVIEW_ZH.md`。
- 另立 `COPSD-fast-kl-probe-20261016` 源码副本，在官方 `beta=0` 损失中分块计算并用 autograd 重算以降低临时显存；CPU 小张量 loss/梯度对照原代码通过，补丁 `patches/fast_kl_batch2_probe.patch` 可对原适配源码 `git apply --check`。完整 2048 上限的单卡 micro batch 2 单更新短测 `copsd-sw-17b-fastkl-batch2-probe-v1` 于 16:35 手动排队，目前等待 GPU。它仅是吞吐/显存实验，不修改正式 SW 进程，也不在 GPU 实测之前声称已经提速。

## 官方来源与实现边界

- 官方代码：`cisnlp/COPSD`，固定提交 `4f4410730e9b6775a30d20f4b6e141cba2e87590`，树 `10de1fd160b34901c57e80a56f0b082a4727df12`。服务器到 GitHub 的 `git clone` 和 codeload TLS 握手失败；通过 GitHub 连接器取得独立源码快照 `/data0/users/lixiaoyang/COPSD-official-20261015`。19 个文本源码的 Git blob SHA 与该提交递归树逐一相同；两个无关 `.DS_Store` 二进制文件未取回。它是经哈希验证的源码快照，不冒称成功的原生 Git checkout。
- 官方源码 `multilingual_opsd_train.py`、`multilingual_opsd_trainer.py`、`multilingual_data_collator.py`、`language_config.py`、`multilingual_scripts/run_all_opsd_4b_3000.sh` 及两套评测入口已核对。单卡适配副本 `/data0/users/lixiaoyang/COPSD-local-adapter-20261015` 仅改入口中的缓存路径、`torch.dtype` 配置序列化，以及把官方已有的 `resume_from_checkpoint` 参数传入 `trainer.train()` 以便恢复；差异在零上下文补丁 `patches/local_runtime.patch`，可在官方快照上用 `git apply --unidiff-zero --check` 核验。训练损失与 collator 保持官方源码。恢复功能尚待实际短程验证。
- 论文目标为教师分布到学生分布的逐 token 差异；官方脚本传入 `beta=0`，trainer 使用完整词表的 `F.kl_div(student_log_probs, teacher_log_probs, log_target=True)`。此前自写反向 KL/全参数 SGD、AdamW 入口并非官方方法，已暂停。旧单卡 AdamW 探索性 run 在用户指令时停止，仅执行 2 步，未产出 checkpoint，不报告方法效果。
- 官方发布脚本默认 4 GPU、Qwen3-4B、LoRA r64/alpha128、固定教师禁用 adapter、AdamW、LR 5e-6、有效 batch 32、100 步、2048 completion、温度 1.1/top-p .95/top-k 20、每 5 步保存。默认语言循环只有 BN/SW/TE/TH/ZH/ES/RU/JA，漏 DE/FR；本项目十语言列表包含两者。论文报告官方实验使用 8×A100 或 4×H200，当前服务器 24GB 单卡运行仅作设备适配与先导，不把它叫硬件等同复现。
- 官方依赖写在 `environment.yml`。独立 venv `/data0/users/lixiaoyang/envs/copsd-official-20261015` 以已有 `cl` 环境为 system site packages，单独安装 TRL 0.26.0、Accelerate 1.11.0、PEFT 0.17.1、WandB 0.22.3 和 math-verify 0.8.0；复用 Torch 2.5.1+cu121、Transformers 4.57.6，与官方 Torch 2.8.0/Transformers 4.57.1 不完全一致。现有 FlashAttention 2 因 GLIBC_2.32 不兼容无法导入，单卡启动使用 SDPA；vLLM 未安装，使用官方 trainer 的 Transformers 生成路径，均须单独报告吞吐和效果协议差异。WandB 仅离线，TensorBoard 开启。

## 数据和评测

- 训练数据：`yihongLiu/COPSD-PolyMath-TrainDataset` 固定 revision `519ea2be68fec00c8a033cb2040059ff595b06a0`，十个语言 Parquet 位于 `data/copsd_official_20261015/`；各 3000 条、同一 3000 道英文原题、原顺序一致、无英文题精确重复或必要字段空值，`source/problem/solution/Answer` 在十语逐题一致。发布数据字段 `problem_translated` 与官方 trainer 预期的 `problem_<lang>` 不一致；`prepare_polymath_official.py` 仅按原顺序映射字段并保存哈希。完整 3000 条版本用于官方抽样规则参考，`adapted_v2_ordered/project_train_2900_*.json` 与 100 题共享独立开发集是另列的项目适配实验。详细哈希、清单及英文题精确重叠检查在 `adapted_v2_ordered/DATA_AUDIT.json`。
- 对本机锁定的 MGSM-Rev2 250、PolyMath 500、GSM8K-test 1319 道英文测试题做精确题面交叉检查，均为 0。另将发布训练集 3000 英文题对本机合并的 2319 条 benchmark 英文题做字符 3–5 gram TF-IDF 近重复初筛，最高相似度 0.69857，>=0.7 的 0 条，记录 `adapted_v2_ordered/NEAR_OVERLAP_HEURISTIC.json`；该启发式仍不能排除语义等价或翻译重叠，不能称完全无污染。
- 官方复现：SW 的 AfriMGSM、PolyMath 使用官方提示、采样与 pass@12 单独汇报。项目主评测固定 MGSM-Rev2 十语；PolyMath 十语作补充，英语 GSM8K 作保持评测。MGSM-Rev2 原始仓库本地版本 `5f091edeae66ebc605ac4f98dca07a6c345f0060`，`mgsm_{sw,te,th,bn,zh,ja,ru,de,fr,es}.tsv` 各 250 题。官方 PolyMath 评测代码支持十语，但默认批量脚本仅八语；DE/FR 的扩展需单独记录。单次准确率与 pass@12 分开报告，正式测试不用于挑语言或调参。

## SW 短程运行

- 官方源码的 collator 已在 CPU 上以真实 4B tokenizer 验证：学生提示只含 SW 题，英文题/参考均未泄漏；教师包含两者；两侧均出现官方 SW thinking 前缀。
- `copsd-sw-4b-official-smoke1-v1` 16:26 启动后在第 0 步之前失败，原因是 TensorBoard 序列化 `torch.dtype`；队列失败退出码 1，日志 `logs/copsd_official_10lang/sw_4b_smoke1_v1.run.log`，没有更新或 checkpoint。适配副本改为等价的字符串 `bfloat16` 传给模型加载器，不修改算法。
- `copsd-sw-4b-official-smoke1-v2` 16:28 在 GPU 0 手动启动，16:30 **工程短程完成，退出码 0**。完整 2048 completion 上限、LoRA/fixed teacher/官方损失与参数；仅将更新数设为 1、micro/effective batch 设为 1、关闭 vLLM，属于设备先导，不能代替正式 100 步和有效 batch 32。学生一次在线生成 1176 token，93.50 秒（12.6 tok/s），整步 97.97 秒，train loss 0.007201、非零 grad norm 0.2708。日志 `logs/copsd_official_10lang/sw_4b_smoke1_v2.run.log`；checkpoint、optimizer.pt、scheduler.pt、rng_state.pth、trainer_state.json 和 TensorBoard 均位于 `models/checkpoint/copsd_official_sw_4b_smoke1_v2/`。独立 CPU 重载验证：同一短提示下冻结教师 `disable_adapter()` 与新鲜 Base 的全词表末位置 logits 最大绝对差为 0.0，更新后学生与教师最大差为 0.16015625。此证据支持 teacher adapter 禁用与学生权重实际改变；尚未检查 optimizer 恢复、正式数学评测或方法收益。运行时出现 gradient checkpointing 的 `inputs have requires_grad=False` 警告，LoRA 仍得到非零梯度，继续留意完整 run。
- 本机 GPU 为 8×RTX 3090 24GB；现有任务只能核实单张 GPU 0 空闲。将该短程速度粗略外推至官方 100×32 单题更新，约 87 小时，仅作容量规划，不能视为实际正式耗时，因为题目长度、模型尺寸、micro batch 和生成路径会改变吞吐。已获取 `Qwen/Qwen3-1.7B` 固定 revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` 的文件清单与 LFS SHA256，正在下载校验；十语统一 1.7B 轨道要在模型完整且短程吞吐核实后才启动。
- 16:43 手动启动 `copsd-sw-4b-dev8-base-v1` 单卡独立开发集 8 题 Base 探查，使用官方 PolyMath 语言提示字符串与 2048 回答预算，项目 Transformers 生成后端，逐题输出在 `logs/copsd_official_10lang/sw_4b_dev8_base_v1/`。该输出不是正式 MGSM-Rev2、PolyMath 或 AfriMGSM 分数。
- 上述 Base 8 题已完成：单次 0/8、正常终止 2/8、生成 15,573 token、583.2 秒、峰值 8,071.5 MiB，`DONE.json` 与原始逐题输出已保存。6/8 在 2048 token 上限截断且多数出现循环性 SW 文本。相同 8 题与采样协议的一步 checkpoint 队列 `copsd-sw-4b-dev8-smoke1-v1` 16:55 启动，17:09 完成：单次 1/8、正常终止 4/8、生成 12,059 token、861.6 秒、峰值 8,761.4 MiB。两组逐题原文、配置和 `DONE.json` 均保存；样本量、训练步数和评测预算不足以判断 COPSD 方法收益，尤其不能把 1/8 对 0/8 当正式提升。项目评测解析器随后改为与官方 `extract_boxed_answer` 一致，并在这两组共 16 个输出上验证提取结果一致。
- Qwen3-1.7B 固定 revision 的两个模型 shard 与 tokenizer LFS SHA256 均验证通过，`models/Qwen3-1.7B/MODEL_LOCK.json` 已生成。其他七张 GPU 为其他用户计算任务；`copsd-sw-17b-official-smoke1-v1` 于 17:14 在 GPU 0 启动、17:15 完成完整 2048 上限、LoRA/固定教师/官方损失的单步训练，退出码 0。一次在线生成 806 token、49.35 秒（16.3 tok/s），整步 51.71 秒，loss 0.009567、非零 grad norm 0.3859，checkpoint-1 与 optimizer/scheduler/RNG、TensorBoard 已保存。该单样本时长不能代表正式 32 条/更新的稳定吞吐；完整训练期间将继续核实显存峰值与样本成本。
- **SW 1.7B 正式 single 已于 17:17 启动**：命名队列 `copsd-sw-17b-project2900-formal-v1`，GPU 0，当前 `phase=running`。从原始 Qwen3-1.7B 开始，项目共享训练题 2900 条，独立开发题 100 条；官方 LoRA/fixed teacher/在线 rollout/KL，学习率 5e-6、有效 batch 32（micro 1 × 累积 32）、100 步、生成上限 2048、每 5 步保存、TensorBoard 开启。环境适配为单 GPU、SDPA 与 Transformers 生成，不称完全相同的官方 4B/vLLM 硬件复现。数据 SHA256 `59ea3c059d637cfb203c1abe9203764489dca06d66b3afaf624f8c4b917ad403`。完整冻结配置、源码/数据哈希、GPU UUID 和命令保存在 `logs/copsd_official_10lang/sw_17b_project2900_formal_v1.MANIFEST.json`，训练日志同名前缀 `.run.log`，checkpoint 在 `models/checkpoint/copsd_official_17b_single_v1/sw_17b_project2900_seed42_20261015_v1/`。目前只有启动和第一个 rollout 的证据，**尚无正式结果或方法效果判断**；开发集评测将在 checkpoint 产生后按固定协议独立运行。
- 17:30 核验该训练队列仍有真实计算进程，日志已完成 6 次在线生成、共 11,080 生成 token、生成耗时合计 679.1 秒；有效 batch 32，所以尚未到首次参数更新。GPU 0 前 3 分钟 86 次采样的显存使用量为 14,257–15,619 MiB，利用率均值 26.0%；这是采样观察范围，不是完整运行峰值。若许多回答都生成至 2048 上限，实际总耗时会远高于单条 806 token 的 smoke 外推，不能用小步测试声称已完成正式训练。
- 已用同一固定提示和随机种子准备 100 题 SW 开发集 Base 对照；命名队列 `copsd-sw-17b-dev100-base-v1` 手动启动，17:30 为等待状态，不占 GPU。保存逐题输出与 2048 token 上限、终止、答案提取及成本；checkpoint 出现后才启动对应模型的同协议评测。
- 用户要求优先实测加速后，17:33 停止**尚未占卡**的 `copsd-sw-17b-dev100-base-v1` 等待队列，保留配置和数据，正式训练不受影响。`copsd-sw-17b-batch2-probe-v1` 已手动排队：同一 1.7B/数据/2048 上限/官方损失，micro batch 2、一更新工程短测；若显存足够，再比较题目/秒、有效 token/秒及峰值。正式配置若切换，应设累积 16 保持有效 batch 32，重新从原始 Base 开始、另存 run ID，不把现在 micro 1 的权重混入新轨道。其他七卡均为他人任务，不能直接占用；目前无可立即运行短测的第二张卡。多卡数据并行和 vLLM 仅在资源与兼容性实证满足时考虑，不以等待四卡空闲作为当前 SW 正式训练前提。
- 17:35 早期 9 条正式在线轨迹中 7 条达到 2048 token 上限，均值生成约 107.3 秒/题，单按当前均值 × 3200 题曝光外推约 95.4 小时**生成时间**，尚未加上反向传播、保存和评测。这是早期估计，后续学生长度会变；不能用单步 smoke 的 51.7 秒推算 100 步。若 batch2 实测明显提速且显存安全，再决定是否从 Base 重启新的正式 run；当前 run 继续产出，不凭假定收益中断。
- 官方 AfriMGSM 数据集 `masakhane/afrimgsm` 锁定 revision `8e4268d2b94941f18f63f694cf48e4ae26fbec65`，SW test 250 题，文件哈希在 `data/copsd_official_20261015/afrimgsm_8e4268d2b949/MANIFEST.json`。与 MGSM-Rev2 SW test 有 207/250 完全相同的 SW 题面，四处相同题的答案文本仅差千位逗号；它们按两套协议单独报告，但不能当两份独立测试证据。SW 训练/开发题与 AfriMGSM SW test 的题面精确重叠均为 0，详见 `adapted_v2_ordered/BENCHMARK_OVERLAP_AFRIMGSM_SW.json`。官方 African 评测提示在一题上与本项目 Transformers 后端逐字相同；官方脚本的 pass@12、1024 token 入口 `run_sw_afrimgsm_17b.sh` 已备，vLLM 后端差异继续单列。
- 官方入口训练时 `eval_dataset=None`；直接打开 Trainer 的 `eval_strategy` 会调用没有数据集的评测。当前使用独立开发集评测脚本在训练外评估并保存逐题结果，TensorBoard 在训练中开启；正式训练需固定逐 checkpoint 评测调度并记录为本项目适配，不能把它叫官方内置 eval。

## 后续启动条件

先核对上述短程的显存峰值、每步耗时、有效 token、教师冻结及 checkpoint 可加载；若 4B 单卡不可行，固定 Qwen3-1.7B 的模型 revision，十语全部从同一规模重新开始。正式项目实验使用共享 2900/100 题、相同曝光预算与独立评测；官方 3000 题版本若运行，报告为单独的复现轨道。Base、single、顺序所有结果按协议分别保存，不合并不同模型规模或旧自写方法。
