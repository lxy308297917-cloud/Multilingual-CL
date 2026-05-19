#!/usr/bin/env bash
set -e

# 五种语言
LANGS=(
  "amh_Ethi"
  "hau_Latn"
  "ibo_Latn"
  "npi_Deva"
  "kir_Cyrl"
)

# 使用的NPU（按需改）
NPUS=(0 1 2 3 4)

SCRIPT_PATH="training/scripts/single_3b.sh"

echo "🚀 开始5语言并行训练（3B）"

for i in "${!LANGS[@]}"; do
  LANG=${LANGS[$i]}
  NPU=${NPUS[$i]}

  echo "👉 启动 ${LANG} 在 NPU ${NPU}"

  (
    export ASCEND_RT_VISIBLE_DEVICES=${NPU}

    bash ${SCRIPT_PATH} ${LANG} \
      > logs_${LANG}.log 2>&1
  ) &

done

wait

echo "🎉 所有语言训练完成"