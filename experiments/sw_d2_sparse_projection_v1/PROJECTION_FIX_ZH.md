# 投影优化器启动修复

此前方案二训练在第0步前因 Accelerate 将状态副本中的 basis 移到 CUDA、而身份校验使用 CPU 张量比较而失败。将已保存 basis 转回 CPU/FP32 后按数值比较；投影、梯度、更新和预算逻辑未改。

旧代码 SHA-256：8402049bb1f03e6f6171b0773748ecd18315dd1b15308e9096911591d97a9963
新代码 SHA-256：71375bc33ecce71d703f507ec465f733481444fe5ad03d1077b4ce17237352b2
原冻结放行清单保留在 TRAINING_RELEASE_PRE_PROJECTION_FIX.json；当前放行清单已记录新哈希。失败记录保留，重启只在清理精确 FAILED 标记后执行。
