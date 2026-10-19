# 顺序 OPD 原题池接口

2026-10-19。本项目扩展，非 COPSD 官方划分；未启动顺序训练、未选择三种语言，不改变当前十语言 single 的数据或预算。

## 实际产物

从已冻结的2900道共同训练题构建三个互不重叠的原题池，大小为967／967／966。核验十语言的原题顺序、英文问题、英文参考、答案和来源完全对应；各池与100题独立开发集无交集。另核验英文问题经过NFKC与空白规范化后，三池仍无交集。

按来源、英文问题字符长度四分位、英文参考字符长度四分位分层，固定seed20261019，各分层的三池样本数最多差1。发布训练数据的来源全部是olympiads，且没有难度标签；**长度只是分层代理，不能声称实际数学难度已匹配，也不代表token预算匹配**。

接口已实际导出SW pool0的967条记录，并逐条确认全部字段与原输入一致。这仅验证导出功能，不表示SW已入选顺序实验。没有合成目标语言SFT解答。

统计见 `sequential_pools_20261019.PUBLIC.json`。原题ID、原始路径和输入文件指纹仅保存在本机私有manifest。

## 后续绑定规则

开发集选择完成后，固定L1→pool0、L2→pool1、L3→pool2。

| 顺序 | 阶段1 | 阶段2 | 阶段3 |
|---|---|---|---|
| 正向 | L1／pool0 | L2／pool1 | L3／pool2 |
| 反向 | L3／pool2 | L2／pool1 | L1／pool0 |

反向时语言与原题池一起反转，不能按阶段位置重新给同一语言分配另一池。这样，同一语言在两个顺序中的题目一致。

- 补充三个从Base出发的single对照，各自使用相同语言／池与阶段预算；现有2900题single不能冒充小池匹配对照。
- 每池只有966／967道独立题；若仍使用100更新、名义有效batch32，就会重复题目。记录实际曝光和epoch尾部不足batch的情况，不把3200题次称作3200道独立题。
- 顺序阶段继续同一个adapter，教师保持原始Base；默认各阶段重置optimizer／scheduler。此处只准备数据接口，模型接续入口仍需单独验证。
- 每阶段测全部三语言和英语，使用相同生成和评分协议；本接口不读取测试分数或选择语言。

## 使用

```bash
python prepare_sequential_pools.py build \
  --input-dir "$COPSD_TRAIN_DIR" \
  --output-dir "$COPSD_SEQUENCE_POOLS" --seed 20261019

python prepare_sequential_pools.py materialize \
  --manifest "$COPSD_SEQUENCE_POOLS/POOLS.PRIVATE.json" \
  --language sw --pool 0 --output "$COPSD_SEQUENCE_POOLS/sw_pool0.json"
```

输出位置必须不存在，防止覆盖旧清单。materialize检查原数据指纹并保留原字段；只按既定ID取子集，不重新翻译或生成标签。顺序训练启动门槛仍是完成single开发集筛选与工程检查。
