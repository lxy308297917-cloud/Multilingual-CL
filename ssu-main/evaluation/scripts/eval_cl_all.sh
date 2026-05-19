#!/usr/bin/env bash
set -e

echo "======================================"
echo " FineWeb2 Continual Learning PPL Eval "
echo " Replay + LwF (Lower-Triangular)     "
echo "======================================"

# ======================
# 路径配置（按你的实际情况）
# ======================
DATA_ROOT="D:/fineweb2_cpt"

REPLAY_ROOT="E:/ckpt_replay_seq"
LWF_ROOT="D:/ckpt_lwf_seq"

BATCH_SIZE=8

# ======================
# 语言顺序（与你训练顺序一致）
# ======================
LANGS=(
  swh_Latn
  ibo_Latn
  kir_Cyrl
  tel_Telu
  amh_Ethi
)

cd evaluation/src

# =========================================================
# Replay Evaluation
# =========================================================
echo ""
echo "========== Replay Evaluation =========="

# Task2: ibo_Latn
# python eval_ppl_fineweb2.py \
#   --model_name "${REPLAY_ROOT}/ibo_Latn" \
#   --data_root "${DATA_ROOT}" \
#   --langs swh_Latn ibo_Latn \
#   --batch_size ${BATCH_SIZE}

# Task3: kir_Cyrl
python eval_ppl_fineweb2.py \
  --model_name "${REPLAY_ROOT}/kir_Cyrl" \
  --data_root "${DATA_ROOT}" \
  --langs swh_Latn ibo_Latn kir_Cyrl \
  --batch_size ${BATCH_SIZE}

# Task4: tel_Telu
python eval_ppl_fineweb2.py \
  --model_name "${REPLAY_ROOT}/tel_Telu" \
  --data_root "${DATA_ROOT}" \
  --langs swh_Latn ibo_Latn kir_Cyrl tel_Telu \
  --batch_size ${BATCH_SIZE}

# Task5: amh_Ethi
python eval_ppl_fineweb2.py \
  --model_name "${REPLAY_ROOT}/amh_Ethi" \
  --data_root "${DATA_ROOT}" \
  --langs swh_Latn ibo_Latn kir_Cyrl tel_Telu amh_Ethi \
  --batch_size ${BATCH_SIZE}

# =========================================================
# LwF Evaluation
# =========================================================
echo ""
echo "========== LwF Evaluation =========="

# Task2: ibo_Latn
python eval_ppl_fineweb2.py \
  --model_name "${LWF_ROOT}/ibo_Latn" \
  --data_root "${DATA_ROOT}" \
  --langs swh_Latn ibo_Latn \
  --batch_size ${BATCH_SIZE}

# Task3: kir_Cyrl
python eval_ppl_fineweb2.py \
  --model_name "${LWF_ROOT}/kir_Cyrl" \
  --data_root "${DATA_ROOT}" \
  --langs swh_Latn ibo_Latn kir_Cyrl \
  --batch_size ${BATCH_SIZE}

# Task4: tel_Telu
python eval_ppl_fineweb2.py \
  --model_name "${LWF_ROOT}/tel_Telu" \
  --data_root "${DATA_ROOT}" \
  --langs swh_Latn ibo_Latn kir_Cyrl tel_Telu \
  --batch_size ${BATCH_SIZE}

# Task5: amh_Ethi
python eval_ppl_fineweb2.py \
  --model_name "${LWF_ROOT}/amh_Ethi" \
  --data_root "${DATA_ROOT}" \
  --langs swh_Latn ibo_Latn kir_Cyrl tel_Telu amh_Ethi \
  --batch_size ${BATCH_SIZE}

echo ""
echo "======================================"
echo " All Replay & LwF Evaluation Finished "
echo "======================================"
