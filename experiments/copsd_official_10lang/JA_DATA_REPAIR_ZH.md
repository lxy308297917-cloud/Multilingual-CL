# JA 官方数据异常与独立适配

## 已核验的异常

两次 JA 训练均在同一个采样位置或包含该位置的 rollout 组停滞。官方原始日语 parquet 中，该条记录包含70,738字符的重复“ーハン”前缀，后接154字符日语题干。原题70,831 token，官方 collator 截断至20,000 token，恰占满 vLLM 上下文。栈打印落在生成调度路径；尚未独立复现具体调度器缺陷，不把被采样到的单一函数直接判为死循环。

十语言各2900条训练题的 tokenizer 长度审计中，仅此条超过17,000 token。它不是缺少目标语言 SFT 标签；本次 OPD 仍在线生成学生轨迹。

## 最小数据适配

`repair_ja_repetition.py` 依据本机私有证据清单，重新核对官方 parquet 与适配数据中的原文完全一致，只删除该条已核验的字面重复前缀，保留末尾原有日语题干。没有重写或生成翻译、解答或答案。

- 原始官方数据、原适配数据、失败日志与目录保留。
- 新数据独立版本化，2900题、英文原题 ID、顺序、来源、英文参考解答与答案全部保持。
- 仅一条 `problem_ja` 字段发生变化；完整前后哈希与原题身份保留于私有 manifest。
- 修复后日语题干最长658 token；受影响题在实际官方 collator 中由20,000降至135个学生 prompt token。
- 本轮须标记为**项目数据适配实验**，不能宣称逐条未经修改的官方数据复现。不得与旧失败任务合并计算成功训练成本。

本机复现接口：

```bash
python repair_ja_repetition.py \
  --input ORIGINAL_ADAPTED_JA_JSON \
  --official-parquet ORIGINAL_JA_PARQUET \
  --evidence PRIVATE_SOURCE_AUDIT_JSON \
  --output NEW_JA_JSON --model LOCAL_QWEN3_1_7B
```

## 独立训练入口

`run_ja_repaired_single_with_dev.sh` 复用已完成且身份核验通过的 JA Base 开发评测，从原始 Base 新建学生 adapter，经 `run_ja_data_repair_17b.sh` 调用 `repaired_data_entry.py`。完整100更新后执行同协议开发评测，保留成本、checkpoint和TensorBoard。

原官方 trainer、LoRA、固定教师、AdamW、学习率、有效 batch、损失、2048回答预算及采样参数不变。`prompt_budget_guard.py` 在实际批量 prefetch 后检查学生 prompt 与完整回答预算之和不超过上下文；不截断或改写输入，违规时在调用 vLLM 生成前明确报错。只安装于新入口，不修改运行中其他实验。

CPU 集成使用实际 tokenizer、官方 collator 和批量 prefetch：原损坏输入被拒绝，修复题通过。这不能替代真实 GPU 运行验收；新任务是否已启动、通过原故障组、保存 checkpoint 或完成100更新，分别记录，不预先宣称成功。

## SW 后处理修复

SW 独立2更新 smoke 已实际完成，64次有限 loss、100更新学习率日程、Base不变、三次学生同步、完整恢复文件和正常 CUDA 清理均已核验。旧父流程在随后读取 `training_args.bin` 时导入 TRL/vLLM，因 UUID 映射失败退出，未进入20更新训练。

新 `run_prefix20_with_dev_cpucheck.sh` 仅令 checkpoint 验收进程隐藏 GPU、关闭其 UUID 兼容开关，实际 CPU 验收已通过；复用已完成 smoke，正式20更新仍从 Base 全新初始化。原脚本、错误日志和分配成本保留。用户已明确授权恢复该停止队列及取消停滞 JA 后重训；仍按本账号最多四卡执行，不自动重试。

证据见 `ja_repair_prepared_20261020.PUBLIC.json`、`sw_prefix2_cpucheck_repair_20261020.PUBLIC.json`。本文件不包含原题 ID、原始私有路径或权重身份。
