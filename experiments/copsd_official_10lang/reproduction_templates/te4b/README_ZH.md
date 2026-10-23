# TE4B 源码与科学参数模板

这是完整本机运行配置的安全替代发布：只包含通用源码及科学设置，删除实际存储路径、主机空间快照、进程、设备分配、队列配置和环境快照。本目录模板**不等于本机原冻结文件**，不能用于宣称本机运行哈希一致。实际冻结文件及完整启动证据保留于本地。

## 方法与范围

从原始 Qwen3-4B 独立初始化，模型 revision 与相邻4B组一致。固定 COPSD 官方源码及原始Base教师，LoRA64/128，AdamW 5e-6，有效batch32，2048训练生成上限，beta0分量裁剪0.05；20更新使用100步调度。沿用已记录的服务器在线批量rollout、非破坏性LoRA权重导出及loss内存适配，不宣称严格按论文硬件复现。

评测固定200题、native LoRA／并发16／n1／8192／逐题seed42，先准备BaseTE，再评TE20的TE／英语；复用已验收Base英语缓存。当前公开内容是执行入口，不是效果结论或三语言选择记录。

## 入口

源码位于本目录上两层的实验目录：

- `prepare_remaining_4b_pool.py`：从官方数据核验并准备相同2700训练题及200开发题。
- `te4b_server20_entry.py`、`te4b_server20_job.py`、`run_te4b_server20.sh`：官方trainer上的独立训练入口，两个GPU分别训练及在线推理。
- `evaluate_te4b_followup200.py`、`release_te4b_checkpoint20.py`、`run_te4b_followup200.sh`、`collect_te4b_followup200.py`：固定开发评测、checkpoint放行及逐题验收。
- `te4b_candidate_series.py`：仅供已配置的本地队列集成，观察已有训练/Base评测，成功后启动一次训练模型评测。这里不发布主机队列配置。

上述源码使用调用者提供的项目根目录；不附实际机器地址、账号路径、设备UUID或本机配置。既有公共4B代码、资源锁和数据准备流程仍是依赖，不是一键脱离本项目复现的独立软件包。

## 在具备既有4B产物的工作目录生成配置

先按既有数据/资源流程准备官方模型与数据、共享4B资源锁以及已验收的Base英语输出，再准备本语言：

```bash
python prepare_remaining_4b_pool.py \
  --root "$COPSD_PROJECT_ROOT" --language te \
  --output-report te4b_pool2700_prepared_20261024.PUBLIC.json
```

使用该项目对应Python环境，不修改系统或共享Python。模板中的 `@...@` 是本地绑定占位符，不是编码后的本机信息。渲染器读取显式指定的本地文件，重新计算绑定，不加载模型、启动队列或占用GPU：

```bash
python reproduction_templates/te4b/render_local_configs.py \
  --root "$COPSD_PROJECT_ROOT" \
  --site-packages-relative "$LOCAL_SITE_PACKAGES_RELATIVE" \
  --checkpoint-storage "$CHECKPOINT_STORAGE" \
  --base-english-folder "$ACCEPTED_BASE_EN_FOLDER" \
  --output-dir "$NEW_LOCAL_CONFIG_DIR"
```

渲染器拒绝覆盖已有配置。生成的文件包含调用者本地路径和哈希，**不应作为公开模板上传**。在独立工作目录审核后，将两份配置放到训练/评测源码旁，并确保统一checkpoint入口的软链接实际指向自己选择的存储目录。确认资源归属、足够空间和本地设备分配后，按脚本执行各阶段；渲染配置本身不会开始实验。

本机运行中的冻结文件、代码、采样及权重不因模板发布而改变。不要将模板渲染结果与原运行身份、不同模型规模、不同生成预算或其他serving协议的结果混用。
