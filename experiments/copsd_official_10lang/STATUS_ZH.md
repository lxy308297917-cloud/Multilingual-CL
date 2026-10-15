# COPSD 官方源码十语言实验状态

更新时间：2026-10-15 17:10（Asia/Shanghai）。本轨道执行用户新目标：十语言独立 COPSD，再以独立开发集选择三语言做两条相反顺序的持续 LoRA OPD。此前自写 FFT、SFT 标签批量合成与自写 OPD 均为旧协议，不计入本轨道。

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
- Qwen3-1.7B 固定 revision 的两个模型 shard 与 tokenizer LFS SHA256 均验证通过，`models/Qwen3-1.7B/MODEL_LOCK.json` 已生成。其他七张 GPU 为其他用户计算任务；`copsd-sw-17b-official-smoke1-v1` 于 17:04 手动启动单卡队列，等待当前评测释放 GPU 后做完整 2048 上限、LoRA/固定教师/官方损失的单步吞吐和显存测试。只有该测试通过后才考虑正式 SW 1.7B 训练。
- 官方入口训练时 `eval_dataset=None`；直接打开 Trainer 的 `eval_strategy` 会调用没有数据集的评测。当前使用独立开发集评测脚本在训练外评估并保存逐题结果，TensorBoard 在训练中开启；正式训练需固定逐 checkpoint 评测调度并记录为本项目适配，不能把它叫官方内置 eval。

## 后续启动条件

先核对上述短程的显存峰值、每步耗时、有效 token、教师冻结及 checkpoint 可加载；若 4B 单卡不可行，固定 Qwen3-1.7B 的模型 revision，十语全部从同一规模重新开始。正式项目实验使用共享 2900/100 题、相同曝光预算与独立评测；官方 3000 题版本若运行，报告为单独的复现轨道。Base、single、顺序所有结果按协议分别保存，不合并不同模型规模或旧自写方法。
