# 从公开来源重建统一评测数据

本页解决学校服务器无法访问 AutoDL 共享盘的问题。无需复制原服务器的评测文件，也无需下载模型权重来构建数据。公开源版本、原固定样本 ID、顺序及内容校验保存在 `configs/ig_baseline_data_v1.json`；脚本直接访问带完整提交 SHA 的公开下载地址，不使用 `main` 或可变分支。

## 构建

在仓库根目录执行（路径按学校服务器修改；Python 环境需已有 datasets、pyarrow、pandas）：

```bash
python ssu-main/scripts/prepare_baseline_eval_data.py --output-root /data/ig-baseline-eval
```

脚本只构建数据，不启动 GPU。网络中断后可重跑，已下载的公开文件继续使用。仅当所有 126 个文件均通过数量、逐行内容与顺序校验时，生成 `/data/ig-baseline-eval/DATA_VERIFIED.json`。有任何不匹配会报错，不能把失败状态当作完成。该清单由学校服务器本地生成，包含当地路径，不依赖原服务器的 `evaluation_inputs_inherited` 清单。

| 数据 | 重建方式与范围 |
|---|---|
| EN、IG 摘要 | 固定版本 XL-Sum 的 test + validation，按发布的有序 ID 各取原 500 条；EN/IG 原选样已核实为 Qwen tokenizer 长度稳定升序前 500 条。用 ID 重放避免 tokenizer 库变化影响选样。 |
| HA、KY 文本 | 同样重建原 500 条，仅供既有 PPL 诊断；不新增下游任务。 |
| EN、IG Belebele | 各 900 条完整固定测试数据。 |
| 双向翻译 | FLORES-200 公开镜像的 EN/IG 文本，dev 全 997 条供示例，devtest 按 pandas `sample(n=500, random_state=42)` 固定顺序。镜像与原本地双语文本匹配；不声称它等同于当前 FLORES+。 |
| MMLU | 固定版本的 57 学科完整 test 和 validation，保留学科划分和原顺序。 |
| G-MMLU IG | 完整 dev、test；数据文件条数与任务过滤后的实际评分条数区分记录，沿用任务代码。 |
| IFEval | 固定版本 541 条。 |
| GSM8K | 固定版本 1,319 条 test，转换为现有评测脚本读取的 Arrow。 |

哈希对任务消费的字段逐行计算，同时包含顺序。JSON 空格、Parquet 压缩和 Arrow 序列化格式可能不同，因此不要求重建文件与旧文件二进制一致。学校本地清单还会额外记录重建文件的二进制哈希，用于后续防止数据漂移。摘要校验也覆盖 PPL 使用的 title/text。

## 接入学校服务器的统一入口

在学校服务器已准备好训练/评测环境、Base 和待测 checkpoint 的前提下，生成独立本地配置：

```bash
python ssu-main/scripts/configure_baseline_eval.py \
  --data-root /data/ig-baseline-eval \
  --base-model /data/models/Qwen2.5-1.5B-Instruct \
  --training-python /opt/envs/cl/bin/python \
  --evaluation-python /opt/envs/cl_eval/bin/python \
  --output-root /data/ig-baseline-results \
  --config-out /data/ig-baseline.local.json \
  --gpus 0,1

export BASELINE_TRAINING_PYTHON=/opt/envs/cl/bin/python
export BASELINE_EVAL_CONFIG=/data/ig-baseline.local.json
bash ssu-main/scripts/baseline_eval.sh /data/models/Qwen2.5-1.5B-Instruct --check
bash ssu-main/scripts/baseline_eval.sh /data/models/Qwen2.5-1.5B-Instruct
bash ssu-main/scripts/baseline_eval.sh /data/models/FFT
```

配置完成后，换模型只改最后一个模型路径。控制器会先保证同协议 Base 已完成，已完成且身份匹配的任务可复用。`--check` 只做身份/配置检查，不是正式分数。标准入口仍固定 seed42、每任务一次；不自动增加种子或解码对照。

本地配置只适配路径、解释器和 GPU，保留 v3 的任务、样本限制、提示、生成参数、评分与 EN3/Target3 聚合。数据目录经 `evaluation_data_root` 传给任务进程；本地清单经 `evaluation_data_manifest` 进入协议锁。输出必须使用新的空目录，不能混进原服务器结果。评测环境仍需使用仓库 `lighteval_latest` 源码及兼容依赖；本工具不安装或升级环境。Base 权重、tokenizer 和环境一致性仍需核对，数据一致不等于跨硬件生成必然逐 token 一致。

这是数据重建与路径迁移工具的发布，不代表已经在学校服务器完成全套模型评测。正在原服务器运行的实验不受本次仓库更新影响。本工具也不构建训练语料。

## 发布前验证

2026-09-14：在隔离目录从公开固定 revision 下载全部源文件，126 个重建文件均通过原评测消费字段与顺序的 SHA-256 校验；另冻结每个下载源文件的二进制 SHA-256。数据身份单元测试通过，五类语言任务的新目录路由导入检查通过。未启动 GPU 推理，未修改原服务器运行中的实验。
