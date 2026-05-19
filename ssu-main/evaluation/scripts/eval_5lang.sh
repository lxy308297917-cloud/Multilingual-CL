#!/usr/bin/env bash
set -e

echo "======================================"
echo "开始 5 语言 FineWeb2 PPL 测试"
echo "======================================"

MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"

DATA_ROOT="/d/fineweb2_cpt"   
BATCH_SIZE=8

# 五种语言（与你的数据目录一致）
LANGS=(
  swh_Latn
  amh_Ethi
  tel_Telu
  kir_Cyrl
  ibo_Latn
)

python ../src/eval_ppl_fineweb2.py \
  --model_name "${MODEL_NAME}" \
  --data_root "${DATA_ROOT}" \
  --langs "${LANGS[@]}" \
  --batch_size ${BATCH_SIZE}

echo "======================================"
echo "5 语言测试完成"
echo "======================================"



# python evaluation/src/eval_ppl_fineweb2.py \
#   --model_name "D:/ckpt_fft_seq/ibo_Latn" \
#   --data_root "D:/fineweb2_cpt" \
#   --langs "swh_Latn,ibo_Latn" \
#   --batch_size 8 \
#   --max_test_blocks 5000



python evaluation/src/eval_ppl_fineweb2.py \
  --model_name "D:/ckpt_lwf_seq/amh_Ethi" \
  --data_root "D:/fineweb2_cpt" \
  --langs swh_Latn ibo_Latn kir_Cyrl tel_Telu \
  --batch_size 8 \
  --max_test_blocks 1000