#!/usr/bin/env bash
set -e

echo "========================================"
echo "🚀 顺序 CL：Task4 + Task5"
echo "Task4: npi_Deva"
echo "Task5: amh_Ethi"
echo "========================================"

# 项目根目录下运行
# 确保你已经在 ssu-main 目录

# ===== Task4: Nepali =====
echo "▶️ 开始 Task4: npi_Deva"
bash training/scripts/cl.sh \
  npi_Deva \
  E:/ckpt_fft_ssu_seq/kir_Cyrl

echo "✅ Task4 完成"

# ===== Task5: Amharic =====
echo "▶️ 开始 Task5: amh_Ethi"
bash training/scripts/cl.sh \
  amh_Ethi \
  E:/ckpt_fft_ssu_seq/npi_Deva

echo "🎉 Task4 + Task5 全部完成"
