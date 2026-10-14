# 十语言 Qwen3 数学 Single 交接与实时状态

更新时间：2026-10-15 05:05（Asia/Shanghai）。状态：**共享题单 v8 已冻结、SW/ZH 训练链路烟测已完成；标签验收与正式 SFT 未完成，Base IFEval 与英语开发集评测运行中**。

## 代码与环境

- 主仓库远端 `main` 已同步十语言 single 核心代码、v8 审计与语言识别诊断；本工作树 `/data0/users/lixiaoyang/Multilingual-CL-math-single-20261015` 继续提交本次小样本与并行测评交接。原 `/data0/users/lixiaoyang/Multilingual-CL` 有未提交改动，保留原样。远端最新 SHA 以最终同步核验记录为准，不在本文件内固定自引用提交号。
- Python：`/data0/users/lixiaoyang/ssu-migration/envs/cl/bin/python`；torch 2.5.1+cu121、transformers 4.57.6、datasets 4.8.5、accelerate 1.7.0。当前显卡为 8× RTX 3090 24GB；启动前仍需重新核验 GPU 利用率、显存、进程归属和本账号占卡总量。
- 学生模型：`Qwen/Qwen3-4B`，revision `1cfa9a7208912126459214e8b04321603b3df60c`，本地 `/data0/users/lixiaoyang/models/Qwen3-4B`，已完整落盘并通过两项真实 GPU 评测链路小样本。
- 第一候选标签生成器：`Qwen/Qwen3-8B`，revision `b968826d9c46dd6066d109eabc6255188de91218`，本地 `/data0/users/lixiaoyang/models/Qwen3-8B` 已完整落盘。第二候选 `Qwen/Qwen3-14B`，revision `40c069824f4251a91eefaf281ebe4c544efd3e18`，本地 `/data0/users/lixiaoyang/models/Qwen3-14B` 八个权重分片均已对官方哈希验证；两者都尚未通过十语质量验收。
- 训练 checkpoint 根目录 `/data0/users/lixiaoyang/models/checkpoint/math_single_10lang/` 的 `formal_runs` 链接指向本账号私有的 `/data1/lixiaoyang_math_single_10lang_20261015`，解决四十份 FP32 FSDP 权重预计约 657 GiB 的空间需求。日志在 `/data0/users/lixiaoyang/logs/math_single_10lang/`。尚未创建真实训练 run。

## 可复现命令

```bash
PY=/data0/users/lixiaoyang/ssu-migration/envs/cl/bin/python
ROOT=/data0/users/lixiaoyang/data/math_single_10lang_20261015
$PY experiments/math_single_10lang/export_benchmark_questions.py --mgsm "$ROOT/mgsm_rev2/MGSM-Rev2/mgsm_en.tsv" --mgsm-original "$ROOT/mgsm_original/mgsm/mgsm_en.tsv" --polymath "$ROOT/polymath/en" --gsm8k-arrow /data0/users/lixiaoyang/data/sw-d2-eval-inputs/hf_cache/datasets/gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866/gsm8k-test.arrow --output "$ROOT/benchmark_questions_en_v2.jsonl"
$PY experiments/math_single_10lang/audit_math_data.py --data-dir "$ROOT/copsd_train" --output-dir "$ROOT/audit_v2" --benchmark-questions "$ROOT/benchmark_questions_en_v2.jsonl"
$PY experiments/math_single_10lang/near_duplicate_audit.py --inventory "$ROOT/audit_v2/inventory.jsonl" --benchmark "$ROOT/benchmark_questions_en_v2.jsonl" --output "$ROOT/audit_v2/near_duplicates.jsonl"
$PY experiments/math_single_10lang/audit_gold_semantics.py --inventory "$ROOT/audit_v2/inventory.jsonl" --prior-splits "$ROOT/split_v6_answerleak_free/problem_ids.json" --output-dir "$ROOT/audit_v6_worked_solution_quarantine"
$PY experiments/math_single_10lang/audit_internal_near_duplicates.py --inventory "$ROOT/audit_v2/inventory.jsonl" --output-dir "$ROOT/audit_v7_internal_near_unitaware" --threshold 0.8
$PY experiments/math_single_10lang/freeze_problem_split.py --inventory "$ROOT/audit_v2/inventory.jsonl" --audit "$ROOT/audit_v2/audit.json" --near-duplicates "$ROOT/audit_v2/near_duplicates.jsonl" --problem-exclusions "$ROOT/audit_v6_worked_solution_quarantine/excluded_problem_ids.json" --internal-near-duplicates "$ROOT/audit_v7_internal_near_unitaware/excluded_problem_ids.json" --tokenizer /data0/users/lixiaoyang/models/Qwen3-4B --max-prompt-tokens 1024 --output-dir "$ROOT/split_v8_screened"
$PY experiments/math_single_10lang/export_dev_eval_items.py --inventory "$ROOT/audit_v2/inventory.jsonl" --splits "$ROOT/split_v8_screened/problem_ids.json" --language sw --split dev_tune --output "$ROOT/eval_items_dev_v8/dev_tune_sw.jsonl"
```

标签小样本入口：

```bash
CUDA_VISIBLE_DEVICES=GPU-UUID $PY experiments/math_single_10lang/build_math_sft_data.py --inventory "$ROOT/audit_v2/inventory.jsonl" --splits "$ROOT/split_v8_screened/problem_ids.json" --model /data0/users/lixiaoyang/models/Qwen3-14B --generator-id Qwen/Qwen3-14B --generator-revision 40c069824f4251a91eefaf281ebe4c544efd3e18 --device-map auto --student-tokenizer /data0/users/lixiaoyang/models/Qwen3-4B --output-dir "$ROOT/labels_v8_probe" --languages sw zh --smoke-train 4 --smoke-dev 4
```

该入口只生成 `pending_review`，绝不自动验收。可加 `--english-only` 先做英文数学推理质量门槛；正式训练入口只接受 `accepted` 的全量同 ID 标签；`--smoke` 才允许待审标签，输出不进入成绩比较。Trainer 打开 `eval_on_start`、每半 epoch 的 CE eval、TensorBoard；十语言与英语各 200 道 `dev_tune`、`dev_select` 固定题单已按 v8 导出，生成式 checkpoint 选择仍待完整运行。24GB 单卡无法直接容纳 4B FFT 的 AdamW 状态，正式入口要求四张 GPU 的 FSDP，运行前需测显存与吞吐。

## 已知阻塞

1. 十语言目标语言完整推理标签未完成，独立质量抽查尚未完成。当前禁止正式 SFT。8B 首轮 8 个英文小样本只有 4 条进入待审，SW 4 条里 2 条生成失败；第二轮共同小样本中 TE 0/4、BN 0/3 有闭合格式，SW 1/4 待审且仍有语言质量疑点。第二轮在质量门槛处主动停止，原始部分结果保留于 `labels_v2/` 与 `labels_v2_audit/`。不能把 `pending_review` 当成验收通过。
   标签生成器复用已有文件时现核对题目、gold、目标语言、当前 split、模型 revision 与提示词哈希；训练和标签审计也检查 split。用旧 v4 标签碰 v8 划分的真实样本，身份门槛按预期拒绝，避免把旧待审标签静默混入新题单。
2. Qwen3-8B 对低资源语言不合格；14B 已完整下载。旧 v3–v7 逐步暴露源 `Answer` 指向中间结果、证明题不能用最终式自动评分、问题直接写出答案，以及训练/开发集同题改写交叉等问题；这些旧题单均不可用于正式训练。当前 v8 保守排除测试近似题 108 ID、答案/评分/问题泄漏风险 824 ID、内部同答案高相似（数值答案忽略单位写法）29 ID，三类并集 935，再排除长度 7，剩余 2058 共同题，划为 1000/200/200/658。它仍只是筛查结果，不证明每条数学推理正确。v4 14B 初探针首题虽答对 33 却写错递推参数，SW 输出 640 token 后陷入重复；更严格提示后同题十语格式闭合，但 SW 有“kusolve system”等英语夹杂、生硬表述，自动 langdetect 仍误报通过。v5 英语 10 train+10 dev_tune 的 20 条短解答全被自动标为 `pending_review`、没有格式告警；独立检查至少发现 2 条实质推理错误（最小值只给可行例、不证下界；兔笼计数把自己算作邻居）。记录见 `logs/math_single_10lang/labels_14b_v5_english20_20261015/MANUAL_REVIEW.json`。所有旧样本均未接受；v8 需重新生成并独立审查。
3. SW 四卡 FSDP 20 步烟测已成功，GPU2–5 各约 11.6 GiB 常驻，20 步训练及 eval/save 总用时约 169 秒，TensorBoard 文件已生成。ZH 尝试 FSDP 自带激活检查点虽也成功，但四卡 nvidia-smi 显示约 24 GiB 占用、训练步速仍约 5.5 秒，因此已恢复 SW 的低显存配置。FSDP 当前将 BF16 参数上转 FP32，单份 checkpoint 权重实际为 17,645,742,616 字节（约 16.4 GiB）；按方案每语四份、十语至少约 657 GiB，而 `/data0` 烟测后剩余不足 250 GiB，尚未计终点副本、日志和后续 seed。现已在 `/data1/lixiaoyang_math_single_10lang_20261015` 建立本账号 700 私有目录（该盘剩余约 6.8 TiB），并从统一 checkpoint 根目录建立 `formal_runs` 软链接，原子写入探针已通过。正式 run 应放此链接下独立目录，并继续检查实际剩余空间。`save_only_model=True` 不含优化器状态，中断后必须从 Base 重跑。
4. 固定生成入口已实现且数值提取样例通过；PolyMath 官方 `math_equal` 已在本任务隔离依赖目录通过基础自检，并完成一题真实模型输出端到端验证，保存在 `/data0/users/lixiaoyang/logs/math_single_10lang/base_polymath_sw1_20261015/`。SW Base 两题 GPU 小样本也已完成，原始输出与评分在 `base_sw2_retry1_20261015/`：两题均错，思考区为英语候选、最终解释为 SW 候选。SW 20 步烟测权重可重载并完成一题开发集生成；Base 与烟测权重均耗尽 4096 token，未给出合格答案，每题约 139 秒，配对入口通过。ZH 20 步烟测权重也可重载；一题算出正确值 502，但没有关闭 `</think>`，按协议计错。全部仅为链路 smoke，不是正式分数。第一次 SW Base 尝试因 Transformers 4.57.6 不接受 `generate(generator=...)` 失败，失败输出保留在 `base_sw2_20261015_0247/`；已改为逐题设置 PyTorch seed。语言识别尚未校准；无 Base/单语正式分数。
   IFEval 英语入口复用本机固定 541 条唯一题及既有评分代码，一题 Base smoke 通过，319 token、11.6 秒，提示级严格未通过；见 `logs/math_single_10lang/ifeval_base_smoke_result_20261015/`，不能作正式成绩。随后 GPU2 上已启动全量 Base IFEval，输出在 `logs/math_single_10lang/ifeval_base_formal_20261015_v1/`，完成以 `result/DONE.json` 为准。v5 英语 dev_select Base seed17 因源题泄漏在 0 题后停止，v7 因额外同题改写在 1 题后停止，均保留原始日志，不作成绩。v8 seed17 原单卡已完成前 13/200 行并主动切换为互不重叠的三片，剩余 187 行在 GPU3/4/6 运行；题目 ID、gold、顺序、源文件哈希和逐题随机种子都已核验。分片与合并门槛见 `logs/math_single_10lang/dev_select_en_base_seed17_v8_parallel_20261015/SHARD_MANIFEST.json`，各片 `DONE.json` 齐全后才运行 `merge_eval_shards.py` 并报告完整分数。该任务仍只是一组 seed 的 Base 开发基线，不能单独选后续语言。
   对发布数据中每语 100 条译题的筛查，`langdetect` 将中文仅 49 条标为目标语言，31 条错判、20 条过短或无法判断；该工具只能输出 `candidate_*` 状态，不能用于 best 选择门槛。原始筛查见 `audit_v1/langdetect_question_screen.json`。另在 v8 的 200 道共享题上比较 GlotLID 与 langdetect，中文 GlotLID 150 pass/48 unknown/2 fail，langdetect 103/48/49；SW 分别为 163/6/31 与 187/6/7。GlotLID 对前述夹杂英语且生硬的 SW 标签仍给出 99.7% 的 SW 预测概率，不能据此验收流畅度或代码混用；记录在 `audit_v8_screened/glotlid_selected200.json`、`langdetect_selected200.json`，都只是弱阳性筛查，缺人工标注的生成输出校准。
5. 后续教师、A/B/C、三语言顺序和保持机制的启动条件均未满足，仅 `future_stages.json` 定义接口与评测矩阵字段，未启动后续训练。
6. 当前共同备用池仅 658 原题，不能直接支持三阶段各 1000 道互不重叠的新题；后续顺序阶段需先重新设计共同规模或扩展合格来源。

生成成本约束：14B 同一道题十语探针输出 2719 token，首英文到末目标生成约 184 秒；20 道英文短解答共输出 4934 token，首末约 253 秒。按每语至少 1200 道 train+dev_tune 粗略外推，单份两卡生成器需要约 66 小时才产生英文与十语初稿，尚未计加载、失败重试、独立修订、训练和测评；这个外推不保证未来速度，见 `logs/math_single_10lang/FEASIBILITY.json`。在标签质量门槛未过且本机当前空闲卡不足四张时，无法于本日中午前完成十语正式 SFT 与全量评测，不能为赶时间把待审标签改成已验收。

v8 新探针：14B 从英文原解答整理的 20 道短解答共输入 16,672、输出 5,787 token，18 条待审、1 条格式错误、1 条截断；独立检查至少 3 条数学推理或证明不合格，详见 `logs/math_single_10lang/labels_14b_v8_english20_20261015/MANUAL_REVIEW.json`。新增 `--english-strategy independent_thinking` 只从题面独立求解，按题目 ID 固定采样种子；1600-token 上限首题即截断，4096-token 上限已完成两题中，一题以 2693 生成 token 修正源解错误，另一题用满 4096 token 仍未闭合。原始输出与提前结束原因在 `logs/math_single_10lang/labels_14b_v8_independent4_4096_20261015/RESULT.json`。两种方式都未通过完整性与独立质量门槛，零条标签验收；不能据此估计全库错误率。代码保留精确题目 ID 探针入口，新增 `merge_eval_shards.py` 只在所有分片完成并核对配置、ID 和源 SHA 后合并。

额外源解筛查：v8 保留池的独立展示整数算式中，至少 4 个原题有 8 处明确算术矛盾；这只是极窄规则检出的下界，最终 gold 可能仍正确，详见 `data/math_single_10lang_20261015/audit_v8_screened/reference_arithmetic_exact_display.json`。对这些题必须修订或隔离源解，不能把表面 boxed 相符当作解题过程正确。

后续接口：`export_teacher_check_items.py` 在冻结三语言名单、划分哈希匹配、每语训练/开发标签均为 `accepted` 且英语短解答已审查时，才能导出共同训练侧 100—200 道教师检查题；输出显式区分 `student_visible` 与 `teacher_only`。`make_future_eval_matrix.py` 在同一冻结名单下生成 SFT/B/C、正反顺序、Base 和三阶段、全部入选语言及英语的 360 条计划评测行，含训练/生成 seed 与 token、教师评分和耗时字段。两者都是接口与门槛检查，当前无教师测评或后续训练结果。

## 官方 COPSD 代码核对（仅作后续参考）

已读取 `cisnlp/COPSD` commit `4f4410730e9b6775a30d20f4b6e141cba2e87590` 的 `multilingual_opsd_trainer.py`、`multilingual_data_collator.py`、`language_config.py` 和训练入口。其 `fixed_teacher` 要求 PEFT，并通过关闭 LoRA adapter 恢复原始 Base；这不能直接用于本项目 FFT。其 `use_thinking_machines_loss` 分支在采样 token 上计算策略梯度式损失，不能把该分支直接称作新方案要求的全词表 reverse KL。`generalized_jsd_loss` 是另一分支，支持完整词表或可选 top-k。后续一语言 B/C 需单独冻结 4B 教师并实现/验证全词表 reverse KL，不能直接复用名称相近的代码路径。

PolyMath 官方评分代码 commit `fbf4e41cae78687d6be7447dbea897357c06aaa7` 的仓库副本缺少 `eval/bundled/symeval/__init__.py` 导出，本地评分副本仅新增一行导出；所需 `word2number==1.1`、`antlr4-python3-runtime==4.11.0`、`pebble==5.1.0` 装在本任务 `scorer_deps/`，未改动原有 Python 环境。`math_equal("2","2")`、`math_equal("2","3")`、分数与小数样例均通过，正式评分仍需逐题验收。
