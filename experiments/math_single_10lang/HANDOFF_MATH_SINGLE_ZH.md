# 十语言 Qwen3 数学 Single 交接与实时状态

更新时间：2026-10-15。状态：**数据审计完成首轮；标签、SFT 与正式评测均未完成**。

## 代码与环境

- 主仓库远端 `main` 基点：`14bdac3`；本工作树 `/data0/users/lixiaoyang/Multilingual-CL-math-single-20261015`。原 `/data0/users/lixiaoyang/Multilingual-CL` 有未提交改动，保留原样。
- Python：`/data0/users/lixiaoyang/ssu-migration/envs/cl/bin/python`；torch 2.5.1+cu121、transformers 4.57.6、datasets 4.8.5、accelerate 1.7.0。当前显卡为 8× RTX 3090 24GB；启动前仍需重新核验 GPU 利用率、显存、进程归属和本账号占卡总量。
- 学生模型：`Qwen/Qwen3-4B`，revision `1cfa9a7208912126459214e8b04321603b3df60c`，本地 `/data0/users/lixiaoyang/models/Qwen3-4B`，下载进行中。
- 候选标签生成器：`Qwen/Qwen3-8B`，revision `b968826d9c46dd6066d109eabc6255188de91218`，本地 `/data0/users/lixiaoyang/models/Qwen3-8B`，下载进行中；尚未通过小样本质量验收。
- 训练 checkpoint 计划放 `/data0/users/lixiaoyang/models/checkpoint/math_single_10lang/` 的独立 run 子目录；日志放 `/data0/users/lixiaoyang/logs/math_single_10lang/`。尚未创建真实训练 run。

## 可复现命令

```bash
PY=/data0/users/lixiaoyang/ssu-migration/envs/cl/bin/python
ROOT=/data0/users/lixiaoyang/data/math_single_10lang_20261015
$PY experiments/math_single_10lang/export_benchmark_questions.py --mgsm "$ROOT/mgsm_rev2/MGSM-Rev2/mgsm_en.tsv" --polymath "$ROOT/polymath/en" --gsm8k-arrow /data0/users/lixiaoyang/data/sw-d2-eval-inputs/hf_cache/datasets/gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866/gsm8k-test.arrow --output "$ROOT/benchmark_questions_en.jsonl"
$PY experiments/math_single_10lang/audit_math_data.py --data-dir "$ROOT/copsd_train" --output-dir "$ROOT/audit_v1" --benchmark-questions "$ROOT/benchmark_questions_en.jsonl"
$PY experiments/math_single_10lang/near_duplicate_audit.py --inventory "$ROOT/audit_v1/inventory.jsonl" --benchmark "$ROOT/benchmark_questions_en.jsonl" --output "$ROOT/audit_v1/near_duplicates.jsonl"
$PY experiments/math_single_10lang/freeze_problem_split.py --inventory "$ROOT/audit_v1/inventory.jsonl" --audit "$ROOT/audit_v1/audit.json" --near-duplicates "$ROOT/audit_v1/near_duplicates.jsonl" --tokenizer /data0/users/lixiaoyang/models/Qwen3-4B --max-prompt-tokens 1024 --output-dir "$ROOT/split_v2"
```

标签小样本入口：

```bash
CUDA_VISIBLE_DEVICES=GPU-UUID $PY experiments/math_single_10lang/build_math_sft_data.py --inventory "$ROOT/audit_v1/inventory.jsonl" --splits "$ROOT/split_v2/problem_ids.json" --model /data0/users/lixiaoyang/models/Qwen3-8B --student-tokenizer /data0/users/lixiaoyang/models/Qwen3-4B --output-dir "$ROOT/labels_v1" --languages sw zh --limit 2
```

该入口只生成 `pending_review`，绝不自动验收。正式训练入口只接受 `accepted` 的全量同 ID 标签；`--smoke` 才允许待审标签，输出不进入成绩比较。Trainer 打开 `eval_on_start`、每半 epoch 的 CE eval、TensorBoard；开发集生成式 checkpoint 选择和固定正式评测尚待实现与验证。24GB 单卡无法直接容纳 4B FFT 的 AdamW 状态，正式入口要求四张 GPU 的 FSDP，运行前需测显存与吞吐。

## 已知阻塞

1. 十语言目标语言完整推理标签未完成，独立质量抽查尚未完成。当前禁止正式 SFT。
2. Qwen3-8B 是否足够可靠待小样本验证；不能把其自评视作独立审查。
3. 四卡 4B FFT 尚未 smoke，实际可用显存和速度未知；checkpoint 容量需按磁盘余量复核。代码目前选择 `save_only_model=True` 控制磁盘，这意味着中断后必须从 Base 重跑，不能声称支持优化器状态恢复。
4. 固定生成入口已实现且数值提取样例通过；PolyMath 官方 `math_equal` 已在本任务隔离依赖目录通过基础自检，尚待真实逐题端到端验证。语言识别尚未校准；无 Base/单语正式分数。
5. 后续教师、A/B/C、三语言顺序和保持机制的启动条件均未满足，仅 `future_stages.json` 定义接口与评测矩阵字段，未启动后续训练。
6. 当前共同备用池仅 1477 原题，不能直接支持三阶段各 1000 道互不重叠的新题；后续顺序阶段需先重新设计共同规模或扩展合格来源。

## 官方 COPSD 代码核对（仅作后续参考）

已读取 `cisnlp/COPSD` commit `4f4410730e9b6775a30d20f4b6e141cba2e87590` 的 `multilingual_opsd_trainer.py`、`multilingual_data_collator.py`、`language_config.py` 和训练入口。其 `fixed_teacher` 要求 PEFT，并通过关闭 LoRA adapter 恢复原始 Base；这不能直接用于本项目 FFT。其 `use_thinking_machines_loss` 分支在采样 token 上计算策略梯度式损失，不能把该分支直接称作新方案要求的全词表 reverse KL。`generalized_jsd_loss` 是另一分支，支持完整词表或可选 top-k。后续一语言 B/C 需单独冻结 4B 教师并实现/验证全词表 reverse KL，不能直接复用名称相近的代码路径。

PolyMath 官方评分代码 commit `fbf4e41cae78687d6be7447dbea897357c06aaa7` 的仓库副本缺少 `eval/bundled/symeval/__init__.py` 导出，本地评分副本仅新增一行导出；所需 `word2number==1.1`、`antlr4-python3-runtime==4.11.0`、`pebble==5.1.0` 装在本任务 `scorer_deps/`，未改动原有 Python 环境。`math_equal("2","2")`、`math_equal("2","3")`、分数与小数样例均通过，正式评分仍需逐题验收。
