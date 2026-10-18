# COPSD 官方源码十语言实验状态

更新时间：2026-10-19 00:50（Asia/Shanghai）。本轨道执行用户新目标：十语言独立 COPSD，再以独立开发集选择三语言做两条相反顺序的持续 LoRA OPD。此前自写 FFT、SFT 标签批量合成与自写 OPD 均为旧协议，不计入本轨道。当前状态以最近日期的更新为准，较早段落保留为历史快照。

## 2026-10-19 当前进展

### 00:50 小时监督：SW checkpoint-80／TH checkpoint-35

- 同一长期只读会话核查SW PID1933369和TH launcher3570465实际进程身份，并从日志确认SW80/100、TH35/100。两项均继续，未取消、重启或更改冻结训练配置。00:47心跳时近期估计分别剩余约18小时28分、18小时7分，另计后续评测；不是完成时间保证。
- SW checkpoint-80已于00:24前保存：trainer state global_step=80，adapter、optimizer、scheduler和RNG文件非空。前80更新共2560题次、4,641,757生成token，其中1798条触及2048上限；总用时79小时6分39秒，分配成本79.11 GPU小时。哈希和验证范围见 `sw_17b_checkpoint80_progress_20261019.json`。
- TH checkpoint-35已于00:50前保存：trainer state global_step=35，adapter、optimizer、scheduler和四rank RNG文件非空。前35更新共1120题次、2,135,920生成token，其中853条触及2048上限；总用时9小时47分13秒，分配成本39.15 GPU小时。见 `th_17b_4gpu_checkpoint35_progress_20261019.json`。仅验证文件与trainer state，不重复全量optimizer加载或宣称已验证实际恢复；生成token不等于有效蒸馏token。
- 00:50三个开发集队列均为waiting，状态时间当前；没有新增数学效果分数或vLLM GPU测速结果。暂停测速的确认尚未到达，现有训练继续。长期只读会话保持运行，观察后续保存点和小时心跳。

## 2026-10-18 历史进展


### 23:48 小时监督与 checkpoint-30

- 新的长期只读观察会话按每分钟检查SW PID1933369和TH launcher3570465的实际进程身份，并读取日志确认SW79/100、TH31/100。最新保存为SW checkpoint-75与TH checkpoint-30；近期估计剩余分别约19小时11分、19小时27分，另计后续评测。两项均继续，没有取消、重启或改动冻结配置。
- TH checkpoint-30已于23:27前落盘，trainer state global_step=30，adapter、optimizer、scheduler与四rank RNG均非空。前30更新完成960条轨迹、1,813,696生成token，其中703条触及2048上限；总用时8小时24分33秒，分配成本33.64 GPU小时。权重SHA256和验证范围见 `th_17b_4gpu_checkpoint30_progress_20261018.json`；没有重复optimizer全量加载或声称实际恢复已通过。
- 23:48读取三个开发集队列仍为waiting，均有当前时间状态更新；尚无新增效果分数或vLLM GPU测速结果。暂停TH测速的确认仍未到达，现有训练继续。长期只读观察会话保持存活，下一保存点为SW checkpoint-80／TH checkpoint-35，不因原观察程序正常结束而重启训练。


### 22:44 小时监督与 checkpoint-25

- 同一只读监测会话直接检查SW PID1933369和TH launcher3570465进程身份、确认日志持续推进：SW78/100，TH27/100，最新保存分别为checkpoint-75与checkpoint-25。近期日志估计剩余约19小时36分、20小时50分，另计后续评测；未执行取消或重启。
- TH checkpoint-25于22:03前保存，adapter、optimizer、scheduler和四rank RNG均非空，trainer state global_step=25。前25更新共800条在线轨迹、1,500,656生成token，573条触及2048上限；累计7小时1分、分配成本约28.07 GPU小时。哈希和范围说明见 `th_17b_4gpu_checkpoint25_progress_20261018.json`。没有把生成token当作有效蒸馏token，未重复加载全部optimizer或声称已验证实际恢复。
- 22:44读取三个开发集队列仍为waiting，均有当前时间状态更新；没有新增开发集分数或GPU测速结果。暂停TH测速的用户确认仍未到达，现有训练继续；同一监测会话继续观察checkpoint-30。


### 21:40 小时监督

- 持续运行的同一只读监测会话直接复查SW PID1933369与TH launcher3570465的进程身份，并从实际训练日志确认SW77/100、TH23/100；两项训练均继续。最新保存仍为SW checkpoint-75、TH checkpoint-20。
- 进度条近期估计分别剩余约20小时41分、21小时41分，尚未计入后续正式评测。当前没有新的开发集分数或vLLM GPU测速结果；不把训练存活当作方法效果证据。
- 未收到暂停TH测速的确认，未执行取消或重启；原监测会话继续观察同一训练的checkpoint-25，不因观察窗口或等待而重启训练。


### 20:39 TH checkpoint-20 已保存

- 同一只读监测会话已观察TH checkpoint-20落盘，随后直接核查SW、TH launcher及四worker仍为本账号存活进程。SW75/100，TH20/100。TH权重、optimizer、scheduler、trainer state与四rank RNG均非空；记录及权重SHA256见 `th_17b_4gpu_checkpoint20_progress_20261018.json`。没有重复全量optimizer验证，也未宣称实际恢复测试已通过。
- TH前20更新完成640条在线轨迹、1,194,860生成token，452条触及2048上限；总用时5小时36分44秒，分配GPU成本约22.45小时。近期进度条估计剩余80更新约22小时17分，仍取决于轨迹长度，另计后续评测成本；有效蒸馏token未单独计量。
- checkpoint-20现已满足对应开发集队列的权重存在条件，但本账号五张卡仍在训练，其他卡仍有任务，未获得GPU的新评测仍无分数。暂停TH测速的用户确认尚未到达，现有SW与TH均继续，未执行取消或恢复操作。


### 19:50 SW checkpoint-75 已保存

- 原只读监测会话已确认SW checkpoint-75完整文件落盘；trainer state global_step=75，adapter、optimizer、scheduler和RNG文件均非空，adapter哈希已保存于 `sw_17b_checkpoint75_progress_20261018.json`。前75更新共2400题次、4,361,803生成token，1703条触及2048上限。没有把文件存在解释为实际恢复测试，也没有重复加载全部optimizer状态。
- 随后直接核查SW、TH launcher及四worker均存活：SW75/100，TH17/100；本账号仍为GPU0及2–5共五卡，其他三卡仍被占用。三个开发集评测队列均waiting，没有新增效果分数。
- 四卡恢复入口已通过CPU预检并提交；暂停TH开展vLLM实测的确认请求仍待答复。未取消、重启或更改当前训练，继续只读观察TH checkpoint-20。


### 19:15 TH checkpoint-15 已保存

- 同一只读监测会话确认 checkpoint-15 落盘，trainer state global_step=15；adapter、optimizer、scheduler 与四 rank RNG 文件均非空。随后直接核查 SW、TH launcher 和四个 worker 仍为本账号存活进程，没有重启、取消或改变冻结配置。
- TH 前15更新：480条在线轨迹、894,363生成token，341/480触及2048上限；训练用时4小时13分37秒，分配成本约16.91 GPU小时。记录与adapter SHA256见 `th_17b_4gpu_checkpoint15_progress_20261018.json`。这是保存与吞吐证据，不是效果分数或实际分布式恢复证明。
- SW 最近核实74/100，TH 15/100。SW checkpoint-65、TH Base及checkpoint-20开发集评测仍待五卡额度内空位；现有SW效果证据仍为同题开发集Base 2/100、checkpoint-20 11/100，尚无新分数。vLLM与修正padding的batch2 GPU测速尚未启动。


### 18:36 小时监督与 checkpoint-10

- 原监测会话直接核查训练 PID 存活并确认日志更新：SW 73/100，TH 12/100。最新保存分别为 checkpoint-70 与 checkpoint-10；两项均在继续，尚无新的开发集／正式测试分数。剩余时长仍约一天，取决于后续轨迹长度及评测成本。
- TH checkpoint-10 已在17:51前落盘，adapter权重及优化器／调度器／四rank RNG文件均在；trainer state global_step=10。前10更新完成320条轨迹、595,236生成token，226条触及2048上限，总用时约2小时48分06秒，分配GPU成本约11.21小时。权重哈希与记录在 `th_17b_4gpu_checkpoint10_progress_20261018.json`。首次checkpoint-5已做CPU完整状态验证，本次未重复加载全部优化器，也未把文件存在当作实际分布式恢复测试。
- 已完成的源码重建、完整官方依赖环境和固定输入测速工具保留就绪；当前等待现有训练和已排队开发集评测推进，未追加GPU实验、重启训练或修改冻结配置。原只读监测会话继续观察checkpoint-15。



### 17:34 小时监督

- 原只读监测会话直接检查 SW PID1933369／TH launcher3570465 仍存活，并从阶段日志核对更新继续增加：SW 72/100（最新保存 checkpoint-70），TH 9/100（最新保存 checkpoint-5）。日志的近期估计分别剩余约26小时／25小时，均未包含后续完整评测且会随输出长度变化。
- 三项已手动启动的开发集评测仍等待账号五卡授权内的空位；没有新增数学效果分数。继续观察同一任务的 checkpoint-10 保存，不因观察窗口、排队或等待而重启训练。



### 正式源码可完整重建

- 已从固定官方 commit 独立取得四个核心原文件，并分别生成 SW single／TH DDP 的完整补丁。临时目录应用后，train、trainer、collator、language_config 四文件逐一与实际运行 manifest 哈希一致，记录在 `CORE_SOURCE_REBUILD_LOCK_20261018.json`。新增完整补丁为 `patches/single_complete_from_4f441073.patch` 和 `patches/formal_ddp_complete_from_4f441073.patch`；它们各自直接用于干净官方 checkout，不与旧增量补丁叠加。
- SW 存活进程实际使用 `COPSD-local-adapter-20261015`，trainer 与官方原文件一致；TH trainer 仅加入教师前向绕过 DDP 的适配。完整四卡补丁还补齐了增量记录遗漏的加载缓存设置，运行中的文件未改。它在当前开启梯度检查点的正式配置下与原条件式相同，但源码哈希复现仍需包含。
- vLLM0.11 的 `external_launcher` 静态接口链保留了官方同步代码所访问的兼容属性，无需据此先行改写同步器；GPU 运行和实际同步仍待空卡验证。原只读 checkpoint-10 监测会话继续观察同一 SW／TH PID，不因本轮源码归档而重启训练。



### TH 四卡首个正式 checkpoint-5

- 只读监测实际 SW／TH PID 等待后，TH 已保存 checkpoint-5 并继续训练。前5次更新完成160条在线轨迹、297,405个生成token，112/160触及2048上限，总用时约1小时24分24秒；完整训练吞吐约58.73生成token/s，分配成本约5.63 GPU小时。按前5次均值粗估剩余95次更新约26.7小时，另计后续保存和评测，学生输出长度可能改变。有效蒸馏token尚未与生成token分开计量，不将二者混称。
- 首次正式 checkpoint 的 CPU 可读性检查通过：392个LoRA tensor均有限，196个LoRA B矩阵非零；392个optimizer参数状态的step均为5，scheduler last_epoch=5，四个rank的Python／NumPy／CPU／CUDA RNG状态均可读取。权重哈希、完整路径和计数见 `th_17b_4gpu_checkpoint5_validation_20261018.json`。这里只验证反序列化和状态一致性，尚未执行实际四卡恢复训练，不宣称完整resume已验证。
- 实际训练进程持续存活，SW与TH的开发集评测仍等待五卡授权内空位。checkpoint保存和非零梯度是工程进展；当前没有TH方法效果分数。



### 英语保持入口与官方依赖就绪

- 已补齐 `run_english_math_17b.sh`，为 Base／single／顺序阶段提供同协议英语 GSM8K 0-shot 保持评测，也支持 MGSM-Rev2 English 的同题阶段对齐补充。新增代码只适配数据与运行记录，复用冻结的现有生成／评分 evaluator，不修改 SW／TH 的排队代码。正式测试尚未启动，须先冻结开发集选择的 checkpoint。
- GSM8K 1319 个英文测试题和数值答案已逐项核对官方固定 commit `b0bb162abedc65e1fdd8e93ed090fd7598ee68bc` 的原始文件；与当前训练2900／开发100的规范化英文题面精确重叠均为0。与 MGSM-Rev2 English 250题共享228题，二者不作为独立证据。`prepare_english_gsm8k.py` 的临时重建与冻结输入字节一致，现有 evaluator 读取1319项及新脚本语法验证通过；详见 `english_math_data_20261018.json`。不复用旧 SW D2 5-shot 分数。
- 独立新环境已进一步安装官方 DeepSpeed0.18.2／bitsandbytes0.48.2，官方 `environment.yml` 的17项固定版本全部匹配；依赖检查及 GPU 不可见时官方训练入口 `--help` 均退出码0。记录在 `vllm011_training_environment_20261018.json` 和完整 freeze。安装禁用预编译 GPU 算子，运行时内核、显存、vLLM生成和权重同步仍待空卡验证。
- 本轮已通过实际 PID 确认 SW／TH 持续运行；SW 最新保存 checkpoint-70，TH 已完成至少2次正式更新。没有因新环境准备或英语数据准备而停止、重启训练或新增 GPU 占用。



### 本轮续跑新证据

- SW 新保存 checkpoint-70，checkpoint-65 的固定开发集评测继续等待；尚无第 65／70 步的完整数学分数。四卡 TH 已完成 2 次正式更新，前两次总用时 34 分 09 秒；grad norm 0.1071、学习率 4.95e-6，进程仍存活。100 更新粗估约 28.5 小时，仍是早期估计，另计保存和评测。
- 保存的同一 100 题 SW 输出已完成 CPU GlotLID 识别：思考区 top1 标准 SW 为 62→79、英语为 1→4；最终回答区存在 9→44，其中 top1 标准 SW 为 6→26，swc 方言标签为 3→8。语言识别未经数学输出校准，不代表流畅度，也不把相关方言自动算失败。原始逐题标签、模型哈希和说明见 `sw_dev100_language_diagnostics_20261018.json`。
- 独立 vLLM0.11／Torch2.8 环境安装已退出码 0 完成，`pip check` 通过；GPU 不可见时成功导入 torch、vllm 与 vllm._C。`vllm011_environment_20261018.json` 与 `vllm011_rollout_environment_freeze_20261018.txt` 保存实际配置。当前没有 GPU 内核或生成测速结果，后续空卡时使用已固定学生 prompt 的 benchmark；正式训练环境未修改。



### 四卡首个正式更新与加速工具准备

- TH 四卡完成第一个 batch32 正式更新：32 条在线轨迹、61,685 生成 token、1048.32 秒，完整更新的生成 token 吞吐约 58.84/s；仅以此步线性外推 100 更新约 29.12 小时，另计评测和保存。记录为 `th_17b_four_gpu_first_update_20261018.json`，不与不同语言的 SW 直接作同题速度比较。
- 已实际核对 SW PID1933369 与 TH launcher3570465／四个 GPU worker 存活；SW 使用 GPU0，TH 使用 GPU2–5，本账号合计五卡。正式 TH 继续运行，不接续调试权重。三项开发集评测继续等待空位。
- 分块 KL＋左填充候选已加入通过四卡 smoke 的 teacher-unwrapped／DDP 参数适配，另准备单卡完整长度 micro2 入口，均未启动。`benchmark_rollout.py` 已在 CPU 固化 4 道 SW 训练题的官方学生 prompt token 清单，后续在同一独立环境比较 Transformers 与 vLLM 的 batch1/2/4、实际生成 token、终止和设备遥测；当前没有 GPU 测速结果。
- 独立 vLLM0.11／Torch2.8 环境正在安装；首次依赖下载因网络读超时退出，延长超时后第二次安装有实际下载进展。安装日志和环境 manifest 保存于 `logs/copsd_official_10lang/`；正式训练环境不受影响，尚不能宣称 vLLM 在 driver535/glibc2.31 上可运行或已提速。



### 15:12 SW 效果复核与提速建议

- 复核 Base 与 checkpoint-20 的同题、同提示、同逐题 seed 和同评分器 100 题开发集：正确 2→11，正常终止 9→43，截断 91→57，可提取答案 11→56，平均 token 1925.65→1680.57。9 题新增正确、原有 2 题均保留。人工确认其中算术、极限和三角化简有正确数学变换；同时存在英语偏移、混乱语句及证明／多问类的答案匹配盲点。原始分数保留，不当正式测试或完整 SW 推理能力结论。详见 `sw_dev100_diagnostics_20261018.json` 与 `ACCELERATION_AND_METHOD_REVIEW_ZH.md`。
- 已对候选分块 KL 补做 CPU AdamW 三更新一致性比较：梯度、参数和动量相同，数值范围与 BF16 舍入差记录在 `fastkl_cpu_equivalence_20261018.json`。现有正式训练代码保持冻结，分块 KL＋正确左填充 batch2 与独立 vLLM 测速仍需 GPU 实测。
- SW 最新可用 checkpoint-65 的同一 100 题开发集评测 `copsd-sw-17b-dev100-ckpt65-v1` 已预检并手动启动排队，等待五卡授权内空位，当前没有第 65 步的效果分数。评测不修改现有 SW/TH 训练。用户建议文本作为参考材料，未将其中反向 KL 公式替换官方 beta=0 的实际损失方向。


### 14:52 用户改为 SW 单卡加四卡联合任务

- 用户要求取消刚开始的 TE、JA、RU，改为四卡联合 OPD；随后明确临时允许本周末占用本账号最多五卡，保留 SW GPU 0。三个队列分别用 `stop --cancel-job` 取消，仅操作本账号新建进程组；状态中退出码 -15 是用户取消，不是训练算法失败。各自日志和目录保留，没有 checkpoint。GPU 1/6/7 为其他用户任务，未操作；GPU 2/3/4/5 被选作四卡任务。
- GPU 2–5 的四卡 SW 工程短测 v1 在模型加载阶段因 TRL `device_map=auto` 与 DDP 冲突失败；隔离源码副本把多进程非量化模型的 `device_map` 设为 `None`。v2 完成 32 条、55,918 生成 token，但在首次反传报 DDP `Expected to mark a variable ready only once`，没有更新。v3/v4/v5 改变梯度累积和检查点配置，仍在反传失败；**v5 名称虽含 no-gc，实际启动参数遗漏关闭梯度检查点，日志仍显示 PyTorch checkpoint，不能作为无检查点反证**。所有失败原始日志分别在 `logs/copsd_official_10lang/sw_17b_fourgpu_smoke_v*.run.log`，无有效四卡训练结果。
- 14:51 手动启动 v6 一步短测：独立官方源码适配副本 `COPSD-ddp-teacher-unwrapped-20261018` 使无梯度固定教师前向绕过 DDP 包装，学生前向仍通过 DDP；同时脚本明确传 `--no_gradient_checkpointing`。这仅验证四卡反传和完整 2048-token 长度的显存，输出独立目录 `models/checkpoint/copsd_four_gpu_17b_smoke/sw_17b_4gpu_smoke_seed42_v6_teacher_unwrap`，队列 `copsd-sw-17b-fourgpu-smoke-v6-teacher-unwrap` 当前运行。若短测成功，需据实测吞吐再从 Base 启动独立四卡正式任务；不能把短测接为正式训练，也不能把 DDP/SDPA 后端称为官方 DeepSpeed/vLLM 完整复现。


- 14:53 v6 结束：关闭检查点后，完整词表 JSD 在 GPU 2 仅剩约 698 MiB 时申请 1.16 GiB，CUDA OOM；没有完成更新。上述“当前运行”为当时快照。
- 14:57–14:59 v7 使用相同隔离源码、非重入式激活检查点和 `ddp_find_unused_parameters=false` 完成四卡完整长度一步：四 rank 生成长度 1181/2048/2048/2048，整步 123.47 秒，loss 0.000343、grad norm 0.1346，退出码 0；`checkpoint-1` 含 adapter、optimizer、scheduler 和四 rank RNG 文件，日志及 TensorBoard 留存。它证明工程更新能完成，不是方法效果结论。短测不接续正式训练。
- 根据 v7 冻结的源码和参数，从原始 Qwen3-1.7B 配置了 TH 独立 single：四 GPU、每卡 micro 1、梯度累积 8、有效 batch 32、100 更新、2048 token 上限、LoRA r64、固定教师、每 5 更新保存，TensorBoard 开启。15:01 已手动启动队列 `copsd-th-17b-project2900-4gpu-v1`，脚本 `run_four_gpu_17b_project_v2.sh`，输出 `models/checkpoint/copsd_official_17b_single_4gpu_v1/th_17b_project2900_seed42_4gpu_20261018_v2`，日志 `logs/copsd_official_10lang/th_17b_project2900_4gpu_v2.run.log`；冻结脚本、源码、数据、模型哈希在同名前缀 `MANIFEST.json`。需核查取得 GPU 和首步更新，启动命令本身不代表训练成功。正式 eval 继续由独立开发集脚本在 checkpoint 后执行；官方入口 `eval_dataset=None`，没有把训练 loss 冒充 eval。


- 15:01:25 队列实际取得 GPU 2/3/4/5，四个训练进程开始 TH 在线生成；SW GPU 0 同时运行，本账号正好五卡，其他用户 GPU 1/6/7 未动。训练队列 `phase=running`，首个有效 batch32 更新仍未完成，正式总耗时需据首个完整更新估计。
- TH Base 与四卡 TH checkpoint-20 的同一 100 题开发集、同一单次回答协议评测队列已分别 `validate` 并手动启动，当前均为 `waiting`，不占 GPU。队列 `copsd-th-17b-dev100-base-4gpu-v2`、`copsd-th-17b-dev100-ckpt20-4gpu-v2` 遵守五卡总上限；仅当 SW 或 TH 释放卡后才运行。checkpoint-20 若未形成，适配评测脚本会拒绝启动并保留失败记录，不能将排队视为已获得效果结果。

- SW 1.7B single 仍在 GPU 0 运行，已保存至 checkpoint-65。14:03 日志中约 2201 条在线 rollout、4,005,418 个生成 token，纯生成累计 68.04 小时；最近 100 条平均 98.8 秒/题，约 71.5% 的全部回答触及 2048 token 上限。100 步目标与正式测试尚未完成。队列 `copsd-sw-17b-project2900-formal-v1` 为 `running`，对应 PID 1933369。
- 同一 100 题独立开发集和同一单次生成协议下，Base 2/100、正常终止 9/100；SW checkpoint-20 11/100、正常终止 43/100。两项队列于 10 月 16 日分别以退出码 0 完成，原始逐题输出、配置和 `DONE.json` 在 `logs/copsd_official_10lang/evaluation/sw_17b_dev100_{base,ckpt20}_v1/`。这是项目开发集早期改进信号，不能代替正式 MGSM-Rev2、PolyMath 或官方 AfriMGSM 成绩，也不能据此单独宣称十语效果。
- 隔离的 micro batch 2 分块 KL 单步探针于 10 月 16 日完成，生成两条各 2048 token 用时 138.61 秒、整步 144.77 秒，未 OOM。日志同时给出 decoder-only **右侧填充警告**；不同长度提示词的批量生成可能受填充影响，因此该步仅证明显存和计算链路能运行，29.6 token/s 不能直接当作可采用的正确提速结果。正式 SW 与新语言继续使用未改 trainer 的 micro 1；左侧填充与教师/学生位置及 mask 修正须在独立探针验证后才能作为新协议运行。
- 用户于 14:00 明确要求合计占用四张空闲 GPU 推进 OPD。启动前确认本账号仅 SW 占 GPU 0，GPU 1/6/7 是其他用户任务，GPU 2/3/4/5 空闲；没有操作他人进程。将 TE、JA、RU 三个先前停止且未训练的命名队列 `max_user_gpus` 从 3 改为默认上限 4，逐个通过 `validate` 并手动 `start`。14:03 确认 TE、JA、RU 分别在 GPU 2/3/4 以 PID 3537260/3537358/3537622 运行，连同 SW 共四张本账号 GPU；GPU 5 留空。三个语言从同一原始 Qwen3-1.7B revision 独立初始化、共享原题集合与 100 步预算，使用原单卡官方 trainer 适配路径，日志与 checkpoint 各自独立。队列不设开机自启、失败自动重试；当前尚无三语有效训练结果。

## 2026-10-16 实时进展

- SW 1.7B single 正式任务继续在 GPU 0 运行，队列 `copsd-sw-17b-project2900-formal-v1` 于 16:15 状态为 `running`，有对应 GPU 计算进程。最新已落盘 checkpoint-20（100 次优化更新目标的 20 次）；日志至 16:16 已完成 730 条在线 rollout，共生成 1,312,571 token，生成阶段累计 22.70 小时；547/730 条触及 2048 token 上限。最后 100 条均值约 107.5 秒/题、1,718.9 token/题。按当前吞吐仅粗估剩余约 3 天生成计算，另有反传、保存和评测；学生轨迹长度可能变化，完成时间不能保证。loss 变化和 4B 八题 smoke 均不是方法有效的证据，尚无 1.7B 开发集完整 Base/训练模型对照分数。
- 单卡 micro batch 2 短测 `copsd-sw-17b-batch2-probe-v1` 于 2026-10-15 20:02 失败，退出码 1。官方完整词表 KL 的 `F.kl_div` 需要再分配 2.32 GiB 时 GPU 仅余 1.30 GiB，CUDA OOM；该配置不能直接用于本机 24GB 单卡正式训练。失败日志保留，不重复运行，也未影响 micro batch 1 正式任务。
- 16:15 手动启动 SW 100 题开发集 Base 与 checkpoint-20 两条相同协议的评测队列；两者当前均等待空闲 GPU。它们使用独立开发题、单次回答和同一 2048-token 上限，不用正式测试集调配置。评测结果完成前不宣称数学提升。
- 用户要求扩展后，TE、TH、BN、ZH、JA、RU、DE、FR、ES 九个 single 队列已全部单独配置、CPU 预检并于 16:15 手动启动；每个队列单卡、相同 Qwen3-1.7B revision、同一官方训练代码和 2900 条对应题、micro 1 × 累积 32、100 次更新、独立日志与 checkpoint 目录。当时九个均处于 `waiting`，不占 GPU：GPU 0 为本账号 SW，其他七卡均有计算进程。九条新训练队列将本账号并发阈值显式设为 3 张，为评测保留第四张卡的账号容量；实际占用仍以 GPU 空闲和队列复查为准，不占或中断其他用户任务。不设开机启动、故障自动重试。
- MGSM-Rev2 十语单次回答正式测评入口 `run_mgsm_rev2_17b.sh` 已补齐，直接调用现有固定提示和评分器；十个本地 TSV 均以同一加载器验证为 250 条、题目与答案非空，shell 语法预检通过。正式测试不会先于开发集选择拿来调参，当前也没有为此额外占用 GPU。
- PolyMath 补充测试输入已从本地四档 `low/medium/high/top` Parquet 固化为每语言 500 题 JSON，十语逐题答案与英文对应项一致；源文件与输出 SHA256 在 `data/copsd_official_20261015/polymath_project_eval_v1/MANIFEST.json`。项目单次回答入口 `run_polymath_project_17b.sh` 已通过 CPU 加载和语法预检。它与官方默认单档、vLLM 多次采样结果分开报告，目前尚未启动 GPU 测评。
- 用户进一步指出现有十语训练墙钟耗时不可接受后，于 16:28–16:29 停止上述九个**尚未占卡**的等待队列，九个状态均为 `stopped`；配置保留，均未发生训练和产出 checkpoint。SW 正式训练与两条 SW 开发集评测等待队列不受影响。742 条 rollout 的实测生成约 23.07 小时、最近 100 条约 15.97 token/s；详细瓶颈、方案和三篇方法比较见 `ACCELERATION_AND_METHOD_REVIEW_ZH.md`。
- 另立 `COPSD-fast-kl-probe-20261016` 源码副本，在官方 `beta=0` 损失中分块计算并用 autograd 重算以降低临时显存；CPU 小张量 loss/梯度对照原代码通过，补丁 `patches/fast_kl_batch2_probe.patch` 可对原适配源码 `git apply --check`。完整 2048 上限的单卡 micro batch 2 单更新短测 `copsd-sw-17b-fastkl-batch2-probe-v1` 于 16:35 手动排队，目前等待 GPU。它仅是吞吐/显存实验，不修改正式 SW 进程，也不在 GPU 实测之前声称已经提速。

## 官方来源与实现边界

- 官方代码：`cisnlp/COPSD`，固定提交 `4f4410730e9b6775a30d20f4b6e141cba2e87590`，树 `10de1fd160b34901c57e80a56f0b082a4727df12`。服务器到 GitHub 的 `git clone` 和 codeload TLS 握手失败；通过 GitHub 连接器取得独立源码快照 `/data0/users/lixiaoyang/COPSD-official-20261015`。19 个文本源码的 Git blob SHA 与该提交递归树逐一相同；两个无关 `.DS_Store` 二进制文件未取回。它是经哈希验证的源码快照，不冒称成功的原生 Git checkout。
- 官方源码 `multilingual_opsd_train.py`、`multilingual_opsd_trainer.py`、`multilingual_data_collator.py`、`language_config.py`、`multilingual_scripts/run_all_opsd_4b_3000.sh` 及两套评测入口已核对。单卡适配副本 `/data0/users/lixiaoyang/COPSD-local-adapter-20261015` 仅改入口中的缓存路径、`torch.dtype` 配置序列化，以及把官方已有的 `resume_from_checkpoint` 参数传入 `trainer.train()` 以便恢复；差异在零上下文补丁 `patches/local_runtime.patch`，可在官方快照上用 `git apply --unidiff-zero --check` 核验。训练损失与 collator 保持官方源码。恢复功能尚待实际短程验证。
- 论文目标为教师分布到学生分布的逐 token 差异；官方脚本传入 `beta=0`，trainer 使用完整词表的 `F.kl_div(student_log_probs, teacher_log_probs, log_target=True)`。此前自写反向 KL/全参数 SGD、AdamW 入口并非官方方法，已暂停。旧单卡 AdamW 探索性 run 在用户指令时停止，仅执行 2 步，未产出 checkpoint，不报告方法效果。
- 官方发布脚本默认 4 GPU、Qwen3-4B、LoRA r64/alpha128、固定教师禁用 adapter、AdamW、LR 5e-6、有效 batch 32、100 步、2048 completion、温度 1.1/top-p .95/top-k 20、每 5 步保存。默认语言循环只有 BN/SW/TE/TH/ZH/ES/RU/JA，漏 DE/FR；本项目十语言列表包含两者。论文报告官方实验使用 8×A100 或 4×H200，当前服务器 24GB 单卡运行仅作设备适配与先导，不把它叫硬件等同复现。
- 官方依赖写在 `environment.yml`。独立 venv `/data0/users/lixiaoyang/envs/copsd-official-20261015` 以已有 `cl` 环境为 system site packages，单独安装 TRL 0.26.0、Accelerate 1.11.0、PEFT 0.17.1、WandB 0.22.3 和 math-verify 0.8.0；复用 Torch 2.5.1+cu121、Transformers 4.57.6，与官方 Torch 2.8.0/Transformers 4.57.1 不完全一致。现有 FlashAttention 2 因 GLIBC_2.32 不兼容无法导入，单卡启动使用 SDPA；vLLM 未安装，使用官方 trainer 的 Transformers 生成路径，均须单独报告吞吐和效果协议差异。WandB 仅离线，TensorBoard 开启。

## 数据和评测

- 训练数据：`yihongLiu/COPSD-PolyMath-TrainDataset` 固定 revision `519ea2be68fec00c8a033cb2040059ff595b06a0`，十个语言 Parquet 位于 `data/copsd_official_20261015/`；各 3000 条、同一 3000 道英文原题、原顺序一致、无英文题精确重复或必要字段空值，`source/problem/solution/Answer` 在十语逐题一致。发布数据字段 `problem_translated` 与官方 trainer 预期的 `problem_<lang>` 不一致；`prepare_polymath_official.py` 仅按原顺序映射字段并保存哈希。完整 3000 条版本用于官方抽样规则参考，`adapted_v2_ordered/project_train_2900_*.json` 与 100 题共享独立开发集是另列的项目适配实验。详细哈希、清单及英文题精确重叠检查在 `adapted_v2_ordered/DATA_AUDIT.json`。
- 对本机锁定的 MGSM-Rev2 250、PolyMath 500、GSM8K-test 1319 道英文测试题做精确题面交叉检查，均为 0。另将发布训练集 3000 英文题对本机合并的 2319 条 benchmark 英文题做字符 3–5 gram TF-IDF 近重复初筛，最高相似度 0.69857，>=0.7 的 0 条，记录 `adapted_v2_ordered/NEAR_OVERLAP_HEURISTIC.json`；该启发式仍不能排除语义等价或翻译重叠，不能称完全无污染。
- 官方复现：SW 的 AfriMGSM、PolyMath 使用官方提示、采样与 pass@12 单独汇报。项目主评测固定 MGSM-Rev2 十语；PolyMath 十语作补充，英语 GSM8K 作保持评测。MGSM-Rev2 原始仓库本地版本 `5f091edeae66ebc605ac4f98dca07a6c345f0060`，`mgsm_{sw,te,th,bn,zh,ja,ru,de,fr,es}.tsv` 各 250 题。官方 PolyMath 评测代码支持十语，但默认批量脚本仅八语；DE/FR 的扩展需单独记录。单次准确率与 pass@12 分开报告，正式测试不用于挑语言或调参。

## SW 短程运行

- 官方源码的 collator 已在 CPU 上以真实 4B tokenizer 验证：学生提示只含 SW 题，英文题/参考均未泄漏；教师包含两者；两侧均出现官方 SW thinking 前缀。
- `copsd-sw-4b-official-smoke1-v1` 16:26 启动后在第 0 步之前失败，原因是 TensorBoard 序列化 `torch.dtype`；队列失败退出码 1，日志 `logs/copsd_official_10lang/sw_4b_smoke1_v1.run.log`，没有更新或 checkpoint。适配副本改为等价的字符串 `bfloat16` 传给模型加载器，不修改算法。
- `copsd-sw-4b-official-smoke1-v2` 16:28 在 GPU 0 手动启动，16:30 **工程短程完成，退出码 0**。完整 2048 completion 上限、LoRA/fixed teacher/官方损失与参数；仅将更新数设为 1、micro/effective batch 设为 1、关闭 vLLM，属于设备先导，不能代替正式 100 步和有效 batch 32。学生一次在线生成 1176 token，93.50 秒（12.6 tok/s），整步 97.97 秒，train loss 0.007201、非零 grad norm 0.2708。日志 `logs/copsd_official_10lang/sw_4b_smoke1_v2.run.log`；checkpoint、optimizer.pt、scheduler.pt、rng_state.pth、trainer_state.json 和 TensorBoard 均位于 `models/checkpoint/copsd_official_sw_4b_smoke1_v2/`。独立 CPU 重载验证：同一短提示下冻结教师 `disable_adapter()` 与新鲜 Base 的全词表末位置 logits 最大绝对差为 0.0，更新后学生与教师最大差为 0.16015625。此证据支持 teacher adapter 禁用与学生权重实际改变；尚未检查 optimizer 恢复、正式数学评测或方法收益。运行时出现 gradient checkpointing 的 `inputs have requires_grad=False` 警告，LoRA 仍得到非零梯度，继续留意完整 run。
- 本机 GPU 为 8×RTX 3090 24GB；现有任务只能核实单张 GPU 0 空闲。将该短程速度粗略外推至官方 100×32 单题更新，约 87 小时，仅作容量规划，不能视为实际正式耗时，因为题目长度、模型尺寸、micro batch 和生成路径会改变吞吐。已获取 `Qwen/Qwen3-1.7B` 固定 revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` 的文件清单与 LFS SHA256，正在下载校验；十语统一 1.7B 轨道要在模型完整且短程吞吐核实后才启动。
- 16:43 手动启动 `copsd-sw-4b-dev8-base-v1` 单卡独立开发集 8 题 Base 探查，使用官方 PolyMath 语言提示字符串与 2048 回答预算，项目 Transformers 生成后端，逐题输出在 `logs/copsd_official_10lang/sw_4b_dev8_base_v1/`。该输出不是正式 MGSM-Rev2、PolyMath 或 AfriMGSM 分数。
- 上述 Base 8 题已完成：单次 0/8、正常终止 2/8、生成 15,573 token、583.2 秒、峰值 8,071.5 MiB，`DONE.json` 与原始逐题输出已保存。6/8 在 2048 token 上限截断且多数出现循环性 SW 文本。相同 8 题与采样协议的一步 checkpoint 队列 `copsd-sw-4b-dev8-smoke1-v1` 16:55 启动，17:09 完成：单次 1/8、正常终止 4/8、生成 12,059 token、861.6 秒、峰值 8,761.4 MiB。两组逐题原文、配置和 `DONE.json` 均保存；样本量、训练步数和评测预算不足以判断 COPSD 方法收益，尤其不能把 1/8 对 0/8 当正式提升。项目评测解析器随后改为与官方 `extract_boxed_answer` 一致，并在这两组共 16 个输出上验证提取结果一致。
- Qwen3-1.7B 固定 revision 的两个模型 shard 与 tokenizer LFS SHA256 均验证通过，`models/Qwen3-1.7B/MODEL_LOCK.json` 已生成。其他七张 GPU 为其他用户计算任务；`copsd-sw-17b-official-smoke1-v1` 于 17:14 在 GPU 0 启动、17:15 完成完整 2048 上限、LoRA/固定教师/官方损失的单步训练，退出码 0。一次在线生成 806 token、49.35 秒（16.3 tok/s），整步 51.71 秒，loss 0.009567、非零 grad norm 0.3859，checkpoint-1 与 optimizer/scheduler/RNG、TensorBoard 已保存。该单样本时长不能代表正式 32 条/更新的稳定吞吐；完整训练期间将继续核实显存峰值与样本成本。
- **SW 1.7B 正式 single 已于 17:17 启动**：命名队列 `copsd-sw-17b-project2900-formal-v1`，GPU 0，当前 `phase=running`。从原始 Qwen3-1.7B 开始，项目共享训练题 2900 条，独立开发题 100 条；官方 LoRA/fixed teacher/在线 rollout/KL，学习率 5e-6、有效 batch 32（micro 1 × 累积 32）、100 步、生成上限 2048、每 5 步保存、TensorBoard 开启。环境适配为单 GPU、SDPA 与 Transformers 生成，不称完全相同的官方 4B/vLLM 硬件复现。数据 SHA256 `59ea3c059d637cfb203c1abe9203764489dca06d66b3afaf624f8c4b917ad403`。完整冻结配置、源码/数据哈希、GPU UUID 和命令保存在 `logs/copsd_official_10lang/sw_17b_project2900_formal_v1.MANIFEST.json`，训练日志同名前缀 `.run.log`，checkpoint 在 `models/checkpoint/copsd_official_17b_single_v1/sw_17b_project2900_seed42_20261015_v1/`。目前只有启动和第一个 rollout 的证据，**尚无正式结果或方法效果判断**；开发集评测将在 checkpoint 产生后按固定协议独立运行。
- 17:30 核验该训练队列仍有真实计算进程，日志已完成 6 次在线生成、共 11,080 生成 token、生成耗时合计 679.1 秒；有效 batch 32，所以尚未到首次参数更新。GPU 0 前 3 分钟 86 次采样的显存使用量为 14,257–15,619 MiB，利用率均值 26.0%；这是采样观察范围，不是完整运行峰值。若许多回答都生成至 2048 上限，实际总耗时会远高于单条 806 token 的 smoke 外推，不能用小步测试声称已完成正式训练。
- 已用同一固定提示和随机种子准备 100 题 SW 开发集 Base 对照；命名队列 `copsd-sw-17b-dev100-base-v1` 手动启动，17:30 为等待状态，不占 GPU。保存逐题输出与 2048 token 上限、终止、答案提取及成本；checkpoint 出现后才启动对应模型的同协议评测。
- 用户要求优先实测加速后，17:33 停止**尚未占卡**的 `copsd-sw-17b-dev100-base-v1` 等待队列，保留配置和数据，正式训练不受影响。`copsd-sw-17b-batch2-probe-v1` 已手动排队：同一 1.7B/数据/2048 上限/官方损失，micro batch 2、一更新工程短测；若显存足够，再比较题目/秒、有效 token/秒及峰值。正式配置若切换，应设累积 16 保持有效 batch 32，重新从原始 Base 开始、另存 run ID，不把现在 micro 1 的权重混入新轨道。其他七卡均为他人任务，不能直接占用；目前无可立即运行短测的第二张卡。多卡数据并行和 vLLM 仅在资源与兼容性实证满足时考虑，不以等待四卡空闲作为当前 SW 正式训练前提。
- 17:35 早期 9 条正式在线轨迹中 7 条达到 2048 token 上限，均值生成约 107.3 秒/题，单按当前均值 × 3200 题曝光外推约 95.4 小时**生成时间**，尚未加上反向传播、保存和评测。这是早期估计，后续学生长度会变；不能用单步 smoke 的 51.7 秒推算 100 步。若 batch2 实测明显提速且显存安全，再决定是否从 Base 重启新的正式 run；当前 run 继续产出，不凭假定收益中断。
- 官方 AfriMGSM 数据集 `masakhane/afrimgsm` 锁定 revision `8e4268d2b94941f18f63f694cf48e4ae26fbec65`，SW test 250 题，文件哈希在 `data/copsd_official_20261015/afrimgsm_8e4268d2b949/MANIFEST.json`。与 MGSM-Rev2 SW test 有 207/250 完全相同的 SW 题面，四处相同题的答案文本仅差千位逗号；它们按两套协议单独报告，但不能当两份独立测试证据。SW 训练/开发题与 AfriMGSM SW test 的题面精确重叠均为 0，详见 `adapted_v2_ordered/BENCHMARK_OVERLAP_AFRIMGSM_SW.json`。官方 African 评测提示在一题上与本项目 Transformers 后端逐字相同；官方脚本的 pass@12、1024 token 入口 `run_sw_afrimgsm_17b.sh` 已备，vLLM 后端差异继续单列。
- 官方入口训练时 `eval_dataset=None`；直接打开 Trainer 的 `eval_strategy` 会调用没有数据集的评测。当前使用独立开发集评测脚本在训练外评估并保存逐题结果，TensorBoard 在训练中开启；正式训练需固定逐 checkpoint 评测调度并记录为本项目适配，不能把它叫官方内置 eval。

## 后续启动条件

先核对上述短程的显存峰值、每步耗时、有效 token、教师冻结及 checkpoint 可加载；若 4B 单卡不可行，固定 Qwen3-1.7B 的模型 revision，十语全部从同一规模重新开始。正式项目实验使用共享 2900/100 题、相同曝光预算与独立评测；官方 3000 题版本若运行，报告为单独的复现轨道。Base、single、顺序所有结果按协议分别保存，不合并不同模型规模或旧自写方法。
