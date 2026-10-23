# 共用4B独立单语言入口

TH试验开始使用共用入口，减少每种语言重复复制训练脚本。每次只读取显式指定的一份本地冻结配置，不自动循环语言、启动额外seed或开启顺序训练。

训练入口是`single4b_compact_entry.py`、`single4b_compact_job.py`与`run_single4b_compact.sh`。配置明确目标语、同一原始Qwen3-4B revision、数据manifest及私有存储绑定。数据字段严格使用`problem_`加目标语代码，英文题目和参考仅进入教师上下文。模型、LoRA、AdamW、全词表beta0/分量裁剪0.05及2048预算沿用已记录4B组。

从训练开始应用已验证的`compact_response_logits.py`存储修复；它不改变loss算术。CPU、独立CUDA及TE实际Trainer越过原OOM样本的证据复用，不重复消耗GPU做相同smoke。TH本身从原始Base开始，没有继承TE adapter。

```bash
COPSD_SINGLE4B_COMPACT_PREPARE_ONLY=1 \
  bash run_single4b_compact.sh "$LOCAL_TRAIN_FREEZE"
```

此命令只进行CPU预检。真实训练需由操作者确认空闲且归属正确的两个设备，并遵守整个账号上限；脚本不会替操作者选择或抢占设备。

评测使用`evaluate_single4b_development.py`、`release_single4b_checkpoint20.py`、`collect_single4b_development.py`与`run_single4b_development.sh`。每份评测配置绑定自己的训练配置、固定200题、原始Base与终点checkpoint20，保持native16/n1/8192/逐题seed42。Base英语仅使用已验收的同协议缓存。共用收集器仍核对逐题内容、提示、采样、软件和模型身份，读取实际训练/评测成本并回读TensorBoard。

```bash
COPSD_SINGLE4B_EVAL_PREPARE_ONLY=1 \
  bash run_single4b_development.sh "$LOCAL_EVAL_FREEZE" Base
```

`single4b_candidate_series.py`只绑定已启动的训练及Base评测；训练成功并放行checkpoint20后，启动一次终点评测。等待资源或观察超时不等于模型失败，不进行自动重试或开机启动。

当前TH本地配置及队列包含机器绑定，未作为公开模板上传；共用入口依赖本项目已建立的4B资源锁、数据及工程证据，不是脱离既有项目的一键复现软件包。训练代码和数据在运行期间保持冻结；新增待运行评测配置独立准备。TE恢复继续使用自己的已冻结恢复入口，不迁移健康运行任务。
