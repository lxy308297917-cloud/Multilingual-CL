# 十语言 Qwen3 数学 Single 交接与实时状态

更新时间：2026-10-15 04:17（Asia/Shanghai）。状态：**共享题单 v5 已冻结、SW/ZH 训练链路烟测已完成；标签验收、正式 SFT 与正式评测均未完成**。

## 代码与环境

- 主仓库远端 `main` 已包含当前主要接口，核对提交 `a6a8be6`；本工作树 `/data0/users/lixiaoyang/Multilingual-CL-math-single-20261015` 的后续修订仍待提交。原 `/data0/users/lixiaoyang/Multilingual-CL` 有未提交改动，保留原样。
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
$PY experiments/math_single_10lang/audit_gold_semantics.py --inventory "$ROOT/audit_v2/inventory.jsonl" --prior-splits "$ROOT/split_v4_goldclean/problem_ids.json" --output-dir "$ROOT/audit_v4_scoreable_gold_semantics"
$PY experiments/math_single_10lang/freeze_problem_split.py --inventory "$ROOT/audit_v2/inventory.jsonl" --audit "$ROOT/audit_v2/audit.json" --near-duplicates "$ROOT/audit_v2/near_duplicates.jsonl" --problem-exclusions "$ROOT/audit_v4_scoreable_gold_semantics/excluded_problem_ids.json" --tokenizer /data0/users/lixiaoyang/models/Qwen3-4B --max-prompt-tokens 1024 --output-dir "$ROOT/split_v5_scoreable"
$PY experiments/math_single_10lang/export_dev_eval_items.py --inventory "$ROOT/audit_v2/inventory.jsonl" --splits "$ROOT/split_v5_scoreable/problem_ids.json" --language sw --split dev_tune --output "$ROOT/eval_items_dev_v5/dev_tune_sw.jsonl"
```

标签小样本入口：

```bash
CUDA_VISIBLE_DEVICES=GPU-UUID $PY experiments/math_single_10lang/build_math_sft_data.py --inventory "$ROOT/audit_v2/inventory.jsonl" --splits "$ROOT/split_v5_scoreable/problem_ids.json" --model /data0/users/lixiaoyang/models/Qwen3-14B --generator-id Qwen/Qwen3-14B --generator-revision 40c069824f4251a91eefaf281ebe4c544efd3e18 --device-map auto --student-tokenizer /data0/users/lixiaoyang/models/Qwen3-4B --output-dir "$ROOT/labels_v5_probe" --languages sw zh --smoke-train 4 --smoke-dev 4
```

该入口只生成 `pending_review`，绝不自动验收。可加 `--english-only` 先做英文数学推理质量门槛；正式训练入口只接受 `accepted` 的全量同 ID 标签；`--smoke` 才允许待审标签，输出不进入成绩比较。Trainer 打开 `eval_on_start`、每半 epoch 的 CE eval、TensorBoard；十语言与英语各 200 道 `dev_tune`、`dev_select` 固定题单已按 v5 导出，生成式 checkpoint 选择仍待完整运行。24GB 单卡无法直接容纳 4B FFT 的 AdamW 状态，正式入口要求四张 GPU 的 FSDP，运行前需测显存与吞吐。

## 已知阻塞

1. 十语言目标语言完整推理标签未完成，独立质量抽查尚未完成。当前禁止正式 SFT。8B 首轮 8 个英文小样本只有 4 条进入待审，SW 4 条里 2 条生成失败；第二轮共同小样本中 TE 0/4、BN 0/3 有闭合格式，SW 1/4 待审且仍有语言质量疑点。第二轮在质量门槛处主动停止，原始部分结果保留于 `labels_v2/` 与 `labels_v2_audit/`。不能把 `pending_review` 当成验收通过。
2. Qwen3-8B 对低资源语言不合格；14B 已完整下载。旧 v3 14B 探针暴露源 `Answer` 与题目终值冲突，v4 先隔离 433 个复杂答案 ID。后来发现证明题的 `Answer` 往往就是待证明命题，当前最终答案自动评分不能判断证明是否成立；v5 把这类题也隔离，合计 785 个源 ID，正式有效共享池 2118 题，划为 1000/200/200/718，旧 v3/v4 标签及题单不用于正式训练。v4 14B 初探针首题英文最终答 33 虽然正确，但递推参数写错（`n1=1,a=3` 得 `n8=509`，题设 257）；同题 SW 输出 640 token 后陷入重复，记录于 `logs/math_single_10lang/labels_14b_v4_probe_20261015_0408/RESULT.json`。随后更严格的提取与翻译提示词使同一题英文数学推理和十语格式均可闭合，原始待审输出在 `labels_14b_prompt2_probe_v4/`，但 SW 有“kusolve system”等英语夹杂、生硬表述，自动 langdetect 仍误报通过，独立语言审查门槛未通过。新版提示词需在 v5 多题上重新检验，所有样本仍未接受。
3. SW 四卡 FSDP 20 步烟测已成功，GPU2–5 各约 11.6 GiB 常驻，20 步训练及 eval/save 总用时约 169 秒，TensorBoard 文件已生成。ZH 尝试 FSDP 自带激活检查点虽也成功，但四卡 nvidia-smi 显示约 24 GiB 占用、训练步速仍约 5.5 秒，因此已恢复 SW 的低显存配置。FSDP 当前将 BF16 参数上转 FP32，单份 checkpoint 权重实际为 17,645,742,616 字节（约 16.4 GiB）；按方案每语四份、十语至少约 657 GiB，而 `/data0` 烟测后剩余不足 250 GiB，尚未计终点副本、日志和后续 seed。现已在 `/data1/lixiaoyang_math_single_10lang_20261015` 建立本账号 700 私有目录（该盘剩余约 6.8 TiB），并从统一 checkpoint 根目录建立 `formal_runs` 软链接，原子写入探针已通过。正式 run 应放此链接下独立目录，并继续检查实际剩余空间。`save_only_model=True` 不含优化器状态，中断后必须从 Base 重跑。
4. 固定生成入口已实现且数值提取样例通过；PolyMath 官方 `math_equal` 已在本任务隔离依赖目录通过基础自检，并完成一题真实模型输出端到端验证，保存在 `/data0/users/lixiaoyang/logs/math_single_10lang/base_polymath_sw1_20261015/`。SW Base 两题 GPU 小样本也已完成，原始输出与评分在 `base_sw2_retry1_20261015/`：两题均错，思考区为英语候选、最终解释为 SW 候选。SW 20 步烟测权重可重载并完成一题开发集生成；Base 与烟测权重均耗尽 4096 token，未给出合格答案，每题约 139 秒，配对入口通过。ZH 20 步烟测权重也可重载；一题算出正确值 502，但没有关闭 `</think>`，按协议计错。全部仅为链路 smoke，不是正式分数。第一次 SW Base 尝试因 Transformers 4.57.6 不接受 `generate(generator=...)` 失败，失败输出保留在 `base_sw2_20261015_0247/`；已改为逐题设置 PyTorch seed。语言识别尚未校准；无 Base/单语正式分数。
   IFEval 英语入口复用本机固定 541 条唯一题及既有评分代码，一题 Base smoke 通过，319 token、11.6 秒，提示级严格未通过；见 `logs/math_single_10lang/ifeval_base_smoke_result_20261015/`，不能作正式成绩。
   对发布数据中每语 100 条译题的筛查，`langdetect` 将中文仅 49 条标为目标语言，31 条错判、20 条过短或无法判断；该工具只能输出 `candidate_*` 状态，不能用于 best 选择门槛。原始筛查见 `audit_v1/langdetect_question_screen.json`。
5. 后续教师、A/B/C、三语言顺序和保持机制的启动条件均未满足，仅 `future_stages.json` 定义接口与评测矩阵字段，未启动后续训练。
6. 当前共同备用池仅 718 原题，不能直接支持三阶段各 1000 道互不重叠的新题；后续顺序阶段需先重新设计共同规模或扩展合格来源。

## 官方 COPSD 代码核对（仅作后续参考）

已读取 `cisnlp/COPSD` commit `4f4410730e9b6775a30d20f4b6e141cba2e87590` 的 `multilingual_opsd_trainer.py`、`multilingual_data_collator.py`、`language_config.py` 和训练入口。其 `fixed_teacher` 要求 PEFT，并通过关闭 LoRA adapter 恢复原始 Base；这不能直接用于本项目 FFT。其 `use_thinking_machines_loss` 分支在采样 token 上计算策略梯度式损失，不能把该分支直接称作新方案要求的全词表 reverse KL。`generalized_jsd_loss` 是另一分支，支持完整词表或可选 top-k。后续一语言 B/C 需单独冻结 4B 教师并实现/验证全词表 reverse KL，不能直接复用名称相近的代码路径。

PolyMath 官方评分代码 commit `fbf4e41cae78687d6be7447dbea897357c06aaa7` 的仓库副本缺少 `eval/bundled/symeval/__init__.py` 导出，本地评分副本仅新增一行导出；所需 `word2number==1.1`、`antlr4-python3-runtime==4.11.0`、`pebble==5.1.0` 装在本任务 `scorer_deps/`，未改动原有 Python 环境。`math_equal("2","2")`、`math_equal("2","3")`、分数与小数样例均通过，正式评分仍需逐题验收。
