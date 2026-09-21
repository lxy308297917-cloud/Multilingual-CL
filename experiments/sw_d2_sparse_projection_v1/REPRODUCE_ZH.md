# SW D2 五方法复现包

本目录对应冻结协议 `sw_d2_sparse_projection_seed42_v1`。五个模型均独立从同一 `Qwen2.5-1.5B-Instruct` Base 初始化，使用同一 SW D2 样本顺序：FineWeb2 70%、Aya 10%、EN→SW 10%、SW→EN 10%；seed 42、最大长度 2048、有效 batch 2、实际输入 15,360,372 token。优化器状态为 FP32。历史 SW D2 FFT 只供跨协议参考；本包的 `fft_fp32_matched` 才是新方法的同协议密集对照。

| 运行脚本 | 训练方法 | 关键差异 |
|---|---|---|
| `run_element_protected.sh` | 英语保护动态稀疏 | Attention/MLP 各矩阵选 15% 元素，英语校准风险惩罚，20 步刷新掩码、200 步刷新风险 |
| `run_element_unprotected.sh` | 去保护对照 | 相同 15% 稀疏预算与刷新规则，移除英语风险项 |
| `run_edge_projected_rows.sh` | 首尾层投影 | 层 1–5、23–27；秩 32 英语输入子空间；投影强度 0.5；每矩阵选 50% 输出行 |
| `run_edge_rows_no_projection.sh` | 无投影对照 | 相同层、行预算和刷新规则，投影强度为 0 |
| `run_fft_fp32_matched.sh` | 同协议密集 FFT | 相同数据、输入预算和 FP32 Adam 动量，全部模型参数更新 |

## 另一台服务器的目录与输入

为严格复现，克隆仓库至 `/root/autodl-fs/ssu-project`，使本目录位于 `/root/autodl-fs/ssu-project/experiments/sw_d2_sparse_projection_v1`，主训练入口位于同级项目的 `ssu-main/scripts/train_cl.py`。冻结的 `TRAINING_RELEASE.json` 同时锁定配置的**真实路径**与 SHA-256；在另一种挂载布局下直接改 JSON 路径会改变协议身份。AutoDL 实例需核对 `/root/autodl-fs` 的 `realpath` 是否仍为 `/autodl-fs/data`。

以下大文件和环境不包含在 Git 仓库中，须从已授权的原始来源或原服务器复制到冻结路径，并按 `TRAINING_RELEASE.json` 及 `manifest.json` 核对 SHA-256：

- Base 模型：`/root/models/Qwen2.5-1.5B-Instruct`，特别是模型、tokenizer 与配置文件；
- SW D2 数据：`/root/autodl-tmp/d2_bilingual_v1/sw/{manifest.json,train.jsonl,dev.jsonl,indices.json}`；
- 英语校准池：`/root/autodl-tmp/sw_update_utility_v1/calibration_candidates.jsonl`；
- 冻结投影基：`/root/autodl-tmp/sw_d2_sparse_projection_v1/edge_bases_rank32.safetensors`；
- Python：`/root/miniconda3/envs/cl/bin/python`，其中 Python 3.10.20、torch 2.5.1+cu121、transformers 4.57.6、datasets 4.8.5、accelerate 1.7.0；此外需安装训练代码导入的 safetensors、tensorboard 等包。训练入口逐项校验核心版本。

`TRAINING_RELEASE.json` 中还锁定了代码、校准审计、投影基构建记录、数据文件等哈希。脚本的 `--check` 只检查基本路径和协议标识；**正式训练入口才会执行完整哈希、预算及数据身份校验**。如需在不同目录、不同软件版本或重新生成数据上运行，需另建带新 ID 的 release，不得把结果冒充为本冻结协议复现。Git 仓库不包含训练语料、校准样本、模型权重、最终 checkpoint 或全部下游评测数据。

## 运行

在数据就位后，每个脚本均可单独运行，默认先训练、再运行固定评测；`--gpu` 指明单卡编号。两卡可分别跑不同脚本，但不要让同一 GPU 同时执行两个任务。

```bash
cd /root/autodl-fs/ssu-project/experiments/sw_d2_sparse_projection_v1
bash run_element_protected.sh --check
bash run_element_protected.sh --gpu 0
bash run_element_unprotected.sh --gpu 1
bash run_edge_projected_rows.sh --gpu 0
bash run_edge_rows_no_projection.sh --gpu 1
bash run_fft_fp32_matched.sh --gpu 0
```

其余选项：`--train-only` 只训练、`--eval-only` 只评测已完成模型、`--resume` 从最近的正式 checkpoint 恢复训练。五个脚本分别读取 `configs/<方法名>.json`，最终仍经 `ssu-main/scripts/train_cl.py` 与 `evaluate_cl.py` 进入冻结实现；已完成的训练会校验权重哈希后跳过。不要在另一个服务器复用当前服务器的 `TRAINING_DONE.json` 而不复制相应权重。

固定评测除本包代码外还需要原项目的 SW 评测、GSM8K harness 0.4.8、PPL 诊断代码、数据、协议锁与各自的 Python 环境；`evaluate_method.py` 会逐项验证任务身份及 DONE 文件。缺失时评测会停止，不能把部分结果当作全套完成。完整评测环境须连同原项目的 `sw_fineweb2_aya_v1`、`ig_gsm8k_harness048_v1` 和评测数据一起迁移。本仓库只提供新方法调用入口与结果，不重分发这些评测数据。

## 结果文件与当前进度

`METHOD1_EXTERNAL_REVIEW_ZH.md` 给出了方案一与去保护对照、Base、历史 SW D2 FFT 的详细比较；`METHOD1_REVIEW_COMPARISON.csv` 是该表的原始机器可读版本。`REPORT_ZH.md`、`raw_scores.csv`、`reading_pairs.csv`、`training_costs.json`、`SUMMARY_STATUS.json` 构成本批次当前**全部已验证结果**。汇总脚本只纳入通过训练 DONE、评测 DONE、模型与任务哈希检查的单元，尚未完成的方法明确写在 `SUMMARY_STATUS.json` 的 `missing` 字段。PPL/NLL 仅作诊断，不参与下游能力综合。后续方法完成后，应重新运行 `summarize_results.py` 并更新这些文件。
