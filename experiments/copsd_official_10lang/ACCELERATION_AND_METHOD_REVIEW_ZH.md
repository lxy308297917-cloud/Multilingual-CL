# COPSD 单卡耗时与相关方法的实验建议（2026-10-16）

## 正式运行源码的完整重建（2026-10-18）


已静态核对新装 vLLM0.11 的同步接口：COPSD 指定的 `external_launcher` 会在 `vllm/config/parallel.py` 关闭 V1 multiprocessing；`vllm/v1/engine/llm_engine.py` 在该模式下保留 `model_executor` 兼容属性，worker wrapper 的属性代理也保留了 `model_runner` 访问。由此未发现仅因 V1 默认多进程接口而必须重写同步代码的证据，不额外补丁化官方同步逻辑。此结论仅针对静态接口链，不等同于 GPU 上生成、权重更新、缓存失效与教师冻结已验证。


已从官方固定 commit `4f4410730e9b6775a30d20f4b6e141cba2e87590` 独立获取四个核心原文件，分别重建当前 SW single 与 TH 四卡运行源码，并与各自启动 manifest 的源码哈希核对一致。SW 的实际工作目录由存活进程确认是 `COPSD-local-adapter-20261015`，不是旁边的 pristine `COPSD-official-20261015`；其 trainer 保持官方原文件。TH 的 trainer 只加入无梯度教师绕过 DDP 包装的适配，完整 KL、clip、mask、在线 rollout 和 LoRA 固定教师定义未改。

此前记录以独立增量补丁为主，正式四卡源码还有 `use_cache=False` 的加载设置没有单独覆盖；正式命令开启梯度检查点，和原条件式在此配置下相同，但逐字节源码重建仍应包含它。现已生成并实际在临时目录应用两份完整补丁，重建后的 train、trainer、collator、language_config 四文件哈希全部与运行清单一致：

- `patches/single_complete_from_4f441073.patch`：重建 SW single 的正式源码。
- `patches/formal_ddp_complete_from_4f441073.patch`：重建 TH 四卡的正式源码。
- `CORE_SOURCE_REBUILD_LOCK_20261018.json`：保存官方原文件 URL／哈希、完整补丁哈希、两条运行轨道的目标文件哈希及验证范围。

在新的干净目录 checkout 上述官方 commit 后，按所需轨道**只应用一份完整补丁**，不要叠加旧增量补丁。示例（仅重建新目录，运行中的源码目录保持冻结）：

```bash
git clone https://github.com/cisnlp/COPSD.git COPSD-rebuild-ddp-4f441073
git -C COPSD-rebuild-ddp-4f441073 checkout --detach 4f4410730e9b6775a30d20f4b6e141cba2e87590
patch -d COPSD-rebuild-ddp-4f441073 -p1 < experiments/copsd_official_10lang/patches/formal_ddp_complete_from_4f441073.patch
```

该核验只证明四个训练核心文件可完整重建；模型、数据、依赖、启动命令和评测仍分别按各自的锁和记录复现，不能把源文件一致等同于论文全部配置或方法效果已复现。

## 2026-10-18 续跑：英语保持评测与完整官方依赖

补齐了英语保持评测输入和入口，尚未启动正式测试。GSM8K main test 的 1319 条输入从既有本地 Arrow 缓存导出，并与 [OpenAI 官方固定 commit 的 test.jsonl](https://raw.githubusercontent.com/openai/grade-school-math/b0bb162abedc65e1fdd8e93ed090fd7598ee68bc/grade_school_math/data/test.jsonl) 的题目及数值答案按顺序逐条核对一致；原始文件 SHA256 为 `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`。数值 gold 来自官方 `####` 后缀，参考解答不会进入模型输入。与当前共享 2900 训练题及 100 开发题的英文题面，在 NFKC／空白合并／casefold 下的精确重叠分别为 0；不声称覆盖模糊或语义重复。与 MGSM-Rev2 English 250 题有 228 道相同英文题，不能把二者当两份独立测试证据。

项目英语 GSM8K 协议固定为 0-shot、COPSD 发布 EN 指令与 English thinking 前缀、2048 token 上限、单次采样及现有 math_verify／boxed 评分，和本轨其他项目单次评测保持同一生成参数。它独立于旧 SW D2 5-shot 协议及 COPSD 官方已发表的其他 benchmark 分数。`run_english_math_17b.sh` 支持 `gsm8k|mgsm_rev2 base|adapter RUN_ID [CHECKPOINT]`，由单卡 GPU 队列调用；MGSM-Rev2 English 250 题可作同题阶段对齐补充，完整 GSM8K 保持评测仍单独报告。输入 JSON 复用冻结 evaluator 的通用 `id/question/answer` 读取器，其历史 CLI 别名为 `polymath_project`；实际 benchmark 身份、输入／evaluator／adapter 哈希由独立 `BENCHMARK_PROTOCOL.json` 明确记录，不能按读取器别名误标结果。现有 SW、TH 等待评测的 evaluator 未改。

`prepare_english_gsm8k.py` 可从固定 SHA 的官方 JSONL 重建输入与重叠报告；已在临时目录重建并确认导出字节与冻结文件完全相同，实际旧 evaluator 可读取全部 1319 项，EN 官方提示存在，新增 shell 与内嵌 Python 语法通过。输入位于 `data/copsd_official_20261015/gsm8k_project_en_zeroshot_v1/`，记录见 `english_math_data_20261018.json`。正式测试仅在开发集选定 checkpoint 并冻结协议后运行，不用于选择语言、训练题或超参数。

独立 vLLM 环境进一步按 `requirements_vllm011_train.txt` 补齐 DeepSpeed0.18.2 和 bitsandbytes0.48.2，`environment.yml` 的全部 17 项官方版本与实际安装完全一致。关闭 GPU 可见性并设 `DS_BUILD_OPS=0` 安装成功，`pip check` 通过，固定官方训练入口 `multilingual_opsd_train.py --help` 退出码 0。完整 freeze 与验证记录为 `vllm011_training_environment_freeze_20261018.txt`、`vllm011_training_environment_20261018.json`。这仅证明依赖和入口导入通过；未编译／测试 DeepSpeed GPU 算子，也未验证 vLLM 的显存、生成、权重同步和训练吞吐。当前正式 SW／TH 继续使用原环境，不因准备新环境而改动运行代码。

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


## 19:25 可审阅的四卡恢复入口（未启动）

新增 `run_four_gpu_17b_resume_v2.sh th NEW_RUN_ID ABSOLUTE_CHECKPOINT`。它保留当前正式TH的冻结源码、环境、四rank×micro1×GA8、100次总更新、2048上限与官方损失，仅增加Trainer的resume参数并输出到新目录。预检验证原manifest的源码／模型锁／数据哈希，以及checkpoint step与optimizer、scheduler、四rank RNG文件；checkpoint-15的CPU预检和shell语法检查已通过，实际GPU恢复尚未验证。不会因准备这个入口而取消当前训练。

建议在用户批准后，于完整checkpoint处短暂停TH，保留SW，进行相同SW学生快照的HF batch2与vLLM batch2/4固定四题测速，再恢复四卡训练。预计测速约20–40分钟，尚无实测保证；交接可能遇其他用户取得空卡，须重新排队而不能强占。当前AGENTS.md要求取消运行中的队列任务需明确授权，因此执行暂停前需用户确认；无答复时现有训练继续。独立推理测速不含教师、反传和在线权重同步，不能直接当作完整OPD提速倍数。


## 2026-10-19 01:08 恢复方案更正：旧Torch无法直接加载optimizer

先前19:25准备的恢复入口不能按原环境直接执行：Torch2.5.1下Trainer的安全版本检查拒绝optimizer反序列化，实际错误见 `resume_legacy_runtime_gate_20261019.json`。不能据“文件完整／CPU预检”宣称可无风险暂停再恢复。当前正式SW和TH未受影响，原环境和冻结源码未改。

已修正**尚未启动**的`run_four_gpu_17b_resume_v2.sh`，改用已有隔离`copsd-vllm011-20261018`环境。新增bootstrap仅为旧NumPy RNG的reconstruct模块名及uint32状态类型提供显式兼容，保留weights-only加载和Torch版本检查，然后运行同一冻结官方适配入口。没有改损失、LoRA、固定教师、batch32、100总更新或2048上限。运行时版本变化必须记录：Torch2.5.1→2.8.0、Transformers4.57.6→4.57.1、datasets4.8.5→3.6.0、NumPy1.26.4→2.2.6；TRL0.26.0／PEFT0.17.1／Accelerate1.11.0相同，不称为原环境精确续跑。

`verify_resume_state_cpu.py`对真实checkpoint-35调用Trainer原生optimizer／scheduler／RNG恢复函数，验证392个optimizer状态及四rank的Python、NumPy、CPU RNG；结果在`resume_torch28_cpu_restore_20261019.json`。这只消除了CPU序列化兼容阻塞：没有验证CUDA RNG、实际LoRA参数名与optimizer顺序映射、四卡更新、数据skip位置或新CUDA内核，尚不能保证实际恢复成功。新入口`--help`、shell语法及冻结身份预检通过。未收到暂停确认，现有训练继续。若后续实际恢复，需冻结并记录新环境、父checkpoint及bootstrap哈希；不能隐去环境改变。
