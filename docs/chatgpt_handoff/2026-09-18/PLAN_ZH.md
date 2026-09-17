# SW 原生阅读数据配方：用户授权 v4

用户最新指令因译制阅读质量不足，改用开源原生阅读数据完成实验。本轨道替代v3译制阅读方案；旧翻译、审核证据与历史模型全部保留。原目标中的禁止TyDiQA限制由此最新明确指令覆盖，目标仍是比较三组数据配方的SW阅读提升及保持，不缩减为仅数据准备。

C采用官方TyDiQA-GoldP secondary_task/train Swahili，revision da78f23f9119363459acbaf46bf89426ff26c259。原问题、上下文和首个原序有效答案不翻译不改写；严格字符偏移检查。官方validation不并入，按语言+文章title及内容连接组留出5%开发集。固定评测和few-shot排除、跨来源检测、完整模板与assistant答案+结束符mask验证全部保留。

采用用户此前给出的原生阅读规则：三组统一2048；C名义Aya55%/阅读10%/双向翻译各10%/摘要15%。阅读三遍容量不足时，在冻结前把差额转给Aya。实际阅读唯一训练输入478384 token，分配1435152 token，差额100848转给Aya。完整样本边界造成的最终误差见C/manifest.json，禁止运行中调整比例或使用第四遍。B纯同一Aya池，A纯FineWeb2，D只作历史参考。

每组独立从同一Qwen2.5-1.5B-Instruct Base初始化，FFT，seed42，15.36M非padding输入预算；有效batch2，AdamW lr5e-5、cosine、warmup5%、weight decay0.01、clip1.0。固定初始/20/40/60/80%/最终开发集，只用最终预算checkpoint。按C→B→A分配空闲GPU，每个模型单卡，可独立并行。≥50GiB空间、来源哈希、数据验收和TRAINING_RELEASE.json通过前禁止训练。

下游沿用固定SW/EN阅读、摘要、双向翻译、IFEval、harness0.4.8五样例GSM8K，以及单独PPL诊断。C−B、B−A比较、2pp与配对文章bootstrap/Holm、3pp/3chrF++保持线、EN3/Target3、最终中文报告/CSV均保留。TyDiQA短答案任务与Belebele四选一形式不同；Aya本身已有CoQA，C是额外阅读监督，不能把C−B全部归因于TyDiQA。C摘要仍是同分布监督泛化。

官方SW原始2755条，初筛2080条；偏移不符478、评测重叠162、已确认内容缺陷10、同源重复23、矛盾答案组2已排除。组拆分和长度过滤后的最终规模、真实监督量、最大重复数、样本与mask在NATIVE_READING_ACCEPTANCE_ZH.md/JSON/CSV及NATIVE_READING_FORMATTED_EXAMPLES.jsonl交付。自动验收不等于全量人工内容审核。
