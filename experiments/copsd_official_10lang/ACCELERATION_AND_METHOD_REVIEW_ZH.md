# COPSD 单卡耗时与相关方法的实验建议（2026-10-16）

## 2026-10-18：SW 逐题诊断与用户提速建议的落实

四卡 TH 正式运行的首个完整更新已完成：32 条 rollout、61,685 个生成 token，用时 1048.32 秒（约 17 分 28 秒），按整次更新计算生成吞吐约 58.84 token/s，线性外推 100 更新约 29.12 小时，另计评测和保存。这是单次更新的 TH 记录，不是同题 SW 的配对测速；学生长度和稳定吞吐仍可能变化。原始计数见 `th_17b_four_gpu_first_update_20261018.json`。原 SW 及四卡 TH 均已核对实际训练 PID 和 GPU 计算进程存活。

为后续空卡测速新增 `benchmark_rollout.py` 与 `run_rollout_benchmark.sh`：直接用固定官方 collator 逐题生成未填充的学生 prompt token IDs，CPU 已固化同一组 4 道 SW 训练题（提示长 227/172/119/146 token），两后端读取相同 token 清单、同一模型身份、可选同一 LoRA snapshot 和相同采样参数。支持 batch1/2/4、2048 生成上限，64-token warmup 单列，不计入正式测速；报告真实生成 token、终止原因、逐题文本、包版本以及每 0.5 秒的设备显存／利用率。该工具仅测独立 rollout，不包含教师、反传或在线权重同步，不能据此直接承诺端到端 OPD 加速。输入位于 `logs/copsd_official_10lang/rollout_benchmark/sw_train4_prompts_20261018/PROMPTS.json`；prepare 已在 CPU 运行成功，GPU 后端尚未执行。

独立 Python 3.10 venv `envs/copsd-vllm011-20261018` 已按 `requirements_vllm011_rollout.txt` 安装 COPSD 发布版本对应的 vLLM0.11.0、Torch2.8.0、Transformers4.57.1 等生成依赖；此阶段未装 DeepSpeed/bitsandbytes，不称完整官方训练环境。首次 pip 下载因网络读超时终止，日志保留；随后用 120 秒超时重试，退出码 0。`pip check` 通过，关闭 GPU 可见性的 CPU 检查成功导入 torch、vllm 和编译 CUDA 扩展 vllm._C，实际包版本与完整 freeze 已保存；GPU 内核、生成与在线权重同步尚未验证。本机 driver535.54.03、glibc2.31；采用独立环境避免影响当前 Torch2.5.1 正式训练。官方安装说明与 NVIDIA 的 CUDA minor compatibility 只说明候选兼容路径，实际 vLLM 内核仍须空卡运行验证。来源：[vLLM0.11 安装说明](https://docs.vllm.ai/en/v0.11.0/getting_started/installation/gpu.html)、[固定版本发布页](https://github.com/vllm-project/vllm/releases/tag/v0.11.0)、[NVIDIA 兼容说明](https://docs.nvidia.com/deploy/cuda-compatibility/latest/595/minor-version-compatibility.html)。环境进度、安装命令和日志位置记录于 `logs/copsd_official_10lang/vllm011_environment_20261018.MANIFEST.json`。

准备中的四卡 fast-KL／leftpad 探针已另建源码副本，加入四卡 smoke 验证通过的 teacher-unwrapped 与 `ddp_find_unused_parameters=false` 适配，避免重复旧 DDP reducer 错误；旧源码保留。同时新增单卡完整长度 micro2 探针 `run_sw_17b_fastkl_leftpad_batch2_probe_v2.sh`，以便仅有一张空卡时先验证正确批量生成。两项均未启动，当前五卡用于 SW 单卡与 TH 四卡正式训练；开发集评测继续等待空卡。

已重新核对 SW Base 与 checkpoint-20 的 100 题独立开发集逐题输出：题目、标准答案、完整提示、逐题采样 seed 和其余评测配置一致，只有 adapter 不同。单次正确数为 2→11，新增正确 9 题、丢失 0 题；正常终止 9→43，2048-token 截断 91→57，成功提取 boxed 答案 11→56，平均生成 token 1925.65→1680.57。按小写词／标点序列定义的四元组重复占比（1−不同四元组数／全部四元组数）均值为 83.56%→41.08%；这个数字只作重复诊断，不是数学或语言能力分数。

人工检查的新增正确题中，2015 的算术、1/2 的因式分解极限、sin(4α)/2 的三角化简均包含正确变换，说明存在实际数学输出改善。但有正确极限题转用英语，其他回答也存在明显混乱语句，不能据答案正确认定 SW 推理与语言保持已达标。另有两项评分适用性问题：证明题的 gold 是待证公式，仅匹配公式不验证证明；一道双问题的 gold 只含第一问的 0，模型把第二问也答成 0，第二问实际极限为 e²。原始数据和 11/100 原始分数保留。事后排除这两题的敏感性诊断是 Base 2/98、checkpoint-20 9/98；它不是重新选择的正式 benchmark。

补充 CPU GlotLID 识别（剥离数学表达式，少于 20 个字母记 too_short，输出续文不含强制 SW 提示前缀）：Base／checkpoint-20 思考区的 top1 swh_Latn 为 62／79，eng_Latn 为 1／4；最终回答区可识别的样本为 9／44，其中 top1 swh_Latn 为 6／26，swc_Latn 为 3／8。checkpoint-20 的 11 个答案匹配成功样本中，思考区 10 个预测为 SW、1 个英语；最终回答区为 swh 2、swc 4、英语 1、其他 1、无独立 final 区 3。swc 与 swh 标签分开保留；该模型没有对这些数学输出完成语言与流畅度校准，不能把相关方言标签当确定失败，也不能把 SW top1 当自然语言质量通过。逐题结果在 `sw_dev100_language_diagnostics_20261018.json`，复现工具 `analyze_output_languages.py` 使用本机已有 model_v3.bin，未占 GPU。

续跑已新保存 checkpoint-70；已排队的 checkpoint-65 评测保持固定，目前只有 checkpoint-20 的上述成对开发集成绩，不能推断更多更新必然提高效果。训练日志末 100 条窗口在步骤 2–5、17–20、57–60 的四元组重复占比均值约为 80.2%、54.6%、13.4%，而思考结束比例分别为 18%、35%、25%；窗口题目不同且没有评分，不能把重复下降当作后期推理提升。

用户所贴建议中的分块完整词表 KL、正确批量生成和独立 vLLM 测速均适用，优先推进已有官方源码适配。需要修正文中 KL 方向：冻结发布源码的 beta=0 分支是 `F.kl_div(log_student, log_teacher, log_target=True)`，即教师到学生方向，并含逐词表项 clip；不按贴文的学生到教师公式替换。当前 generate 已显式传 `use_cache=True`，日志的默认配置警告不能证明实际 KV cache 被关闭。

分块 KL 已有单卡 micro2 完整长度一步成功，但之前的右填充批量生成使该测试不能作为正式吞吐依据；独立左填充和 mask 适配已完成 CPU 位置检查。10 月 18 日新增实际源码函数的 CPU AdamW 三更新比较（两个 seed、FP32/BF16、clip 开／关）：梯度、参数和 Adam 一阶动量最大差均为 0；FP32 loss 差不超过 4.8e-7，BF16 无 clip 的一个配置出现 0.03125 标量舍入差。此测试不验证 GPU 吞吐或长期数值稳定性，记录在 `logs/copsd_official_10lang/fastkl_adamw_cpu_equivalence_20261018.json`。

四卡 DDP 完整长度一步已成功，TH 正式 single 于 15:01 在 GPU 2–5 启动；现有 SW 留在 GPU 0，沿用本次临时五卡授权。下一项有空卡时的工程验证是分块 KL＋正确左填充 micro2，配套降低梯度累积保持有效 batch32，然后独立环境比较官方 vLLM 路径与 Transformers 的真实吞吐。现有正式代码和权重冻结，候选加速版本独立保存。SW 诊断原始逐题结果及 JSON 摘要在 `logs/copsd_official_10lang/evaluation/`，不占用新的 GPU。

## 实测瓶颈和当前处置

- SW 1.7B 正式轨道仍按官方核心训练代码运行：LoRA、固定教师、在线 rollout、完整词表 KL，micro batch 1 × 梯度累积 32、100 更新、回答上限 2048。当前为 RTX 3090 24 GB、Transformers `generate`、SDPA；官方发布脚本则使用 4 GPU、vLLM colocate、FlashAttention 2。硬件和生成后端不同，不能称严格官方速度复现。
- 16:36 采样 742 条在线生成、1,333,587 token、纯生成累计 23.07 小时；最近 100 条约 15.97 token/s、106.64 秒/题。若回答长度分布不变，距 3200 条曝光尚有约 72.8 小时**仅生成**。约 74.7% 的轨迹触及 2048 token 上限。按现状十语直接并行排队不经济，故已停止九语未占卡队列；SW 当前正式进程和 checkpoint 保留，开发集 Base/ckpt20 评测仍等待空卡。
- 原样 micro batch 2 在官方完整词表 KL `F.kl_div` 分配时 OOM（还需 2.32 GiB，余 1.30 GiB）。问题并非模型权重单独装不下，而是批量回答位置 × 词表的临时张量峰值。已在独立 `COPSD-fast-kl-probe-20261016` 源码副本中增加仅对 `beta=0` 生效的 32 位置分块和 autograd 重算，未改损失、mask、教师、生成参数。CPU 上 2 个种子、带/不带官方 clip 的 loss 差不超过 1.2e-7，梯度最大差 0；补丁在 `patches/fast_kl_batch2_probe.patch`。探针日志另按各回答首个 EOS 统计实际生成 token；官方 batch>1 日志把较短回答的填充也算入总 token，会高估吞吐。完整 2048 上限的 micro batch 2 单步 GPU 短测队列已挂起，待卡后确认显存、题目/秒与实际生成 token/秒；成功前不推断提速倍数。

## 按证据推进的加速顺序

1. **先测同卡批量生成。** 若上述等价 KL 版本能让 micro batch 2 完整训练，正式候选设为 micro 2 × 累积 16，仍是有效 batch 32；必须比较相同题数和实际 token 的墙钟吞吐、峰值显存与 loss/梯度，另开 run ID 从 Base 初始化。单次回答长度差异可能使 batch2 的最慢回答拖住同批，不能只看峰值显存。
2. **测官方 vLLM 路径。** 官方源码有 `--use_vllm --vllm_mode colocate` 和独立 server 路径，但本机独立环境未装 vLLM；官方环境钉 `vllm==0.11.0`、`torch==2.8.0`，现有复用环境是 Torch 2.5.1。需独立环境安装兼容依赖，再在一张已核实空闲卡上做完整长度的短测；24GB colocate 可能因模型双份和 KV cache 而 OOM。若取得两张空闲卡，可比较一张 rollout server 加一张训练卡，账号总占用仍不超过四卡。vLLM 的实际收益取决于并行请求数；在 micro 1 串行生成时不会保证显著提速。FlashAttention 2 本机旧 wheel 因 GLIBC 不能导入，不能在当前正式进程中热切换。
3. **再测多卡数据并行。** 官方 4 卡脚本是高吞吐部署，不是 COPSD 算法的必要条件。2 卡或 4 卡只在真实空卡且相同有效 batch/回答预算下测试，记录总 GPU 小时与墙钟时间；不能占其他用户 GPU。它会改善墙钟吞吐，但未必降低总计算成本。
4. **预算策略单独成轨。** COPSD 论文指出很多语言在较早更新就提升，且后续可能下降。对 checkpoint-5/10/20/… 用独立开发集测正确率、终止和目标语言保持；若早期模型已明显优于 Base，可另报 20/40 步资源受限实验。缩短 2048 回答上限或更换损失会改变训练协议，只能作为独立对照，不能与完整官方配置混称复现。

## 三篇工作的关系与推荐实验

| 方法 | 核心额外条件 | 对本机的意义 |
| --- | --- | --- |
| [COPSD 论文](https://arxiv.org/abs/2605.09548) / [代码](https://github.com/cisnlp/COPSD) | 同一模型的目标语言学生 rollout；教师多看英文题和参考解答，完整词表逐 token 蒸馏。论文使用 500 道 OpenThoughts 派生题、100 更新、2048 训练生成。 | 作为十语言 single 和顺序学习的主基线。当前十语共享 PolyMath 3000 发布数据中的 2900 训练／100 开发属于项目适配轨，不能混成论文 500 题复现。论文文字称 reverse KL，而发布脚本 `beta=0` 的实际代码执行教师到学生的 KL；以冻结源码及实测梯度作为本轨计算定义，论文差异单独标注。 |
| [RP-OPSD 论文](https://arxiv.org/abs/2608.06347) / [代码](https://github.com/NJUNLP/RP-OPSD) | 比较带/不带英文参考解答的两个教师视图，以其分布差异定位推理枢纽，再加冻结基座语言锚定；额外教师前向与完整词表 KL。 | 更适合作为 **SW 优先的方法对照**，检验正确率和目标语言保持；并非提速方案。其公开代码含 SW 训练入口，训练数据须另供。SW 成功后才考虑 JA/RU 等项目语言，方法差异、数据差异和额外 FLOPs 全部单列。 |
| [多语言 OPD² 论文](https://arxiv.org/abs/2608.05802) / [官方实现](https://github.com/naver-ai/opd2) | 从强教师相对其基座模型的 log-prob 差提取后训练信号；论文用 Qwen3-30B-A3B-2507 教师、Qwen3-1.7B/8B 学生、英/韩/日 100K 题及 100 更新。 | 这是另一种强教师 OPD，不能与 COPSD 的冻结同模型特权教师等同。当前 24GB 单卡和十语问题池不足以直接复现；先借鉴其“正确率提升是否伴随改用英语回答”的诊断，待主基线和资源允许后再做独立扩展。 |

实验顺序建议：先把 SW 的 Base、checkpoint-20 与后续 checkpoint 做同题开发集对照；按吞吐证据选择官方代码的最快可行运行方式，再跑十语 single。三个顺序训练语言最终按独立开发集的增益、绝对分数、目标语言保持和成本确定；仅作为预设观察组，可优先留意 SW（低资源）、JA（不同文字体系）、RU（中资源），不能跳过十语筛选直接宣称它们入选。主项目评测固定十语 MGSM-Rev2 单次正确率、四档 PolyMath 的分档与加权结果、英语 GSM8K 保持；SW 另按官方 AfriMGSM pass@12 对照。MGSM-Rev2 与 AfriMGSM 的 SW 题大幅重叠，不能当两份独立证据。每个阶段固定题、提示、采样和评分器，并报告终止、答案提取失败、思考区/最终回答语言、生成与训练 token 及 GPU 小时。
