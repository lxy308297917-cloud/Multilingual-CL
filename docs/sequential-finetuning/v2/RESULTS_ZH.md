# Benchmark V2 新实验结果

每条顺序轨迹先评 Base，再逐阶段评同语题组。AP 为终点评分等权均值，以百分数报告。其他指标以百分点报告，方括号是 5,000 次同题配对区间。阶段矩阵的格数包含 Base。历史确认的共享格也计入覆盖。 [实验设置和完整指标定义](PROTOCOL_ZH.md)

## 三个基准的官方测试 Baseline

MASSIVE 使用原顺序。MasakhaNER 和 MTOP 使用冻结的新五语课程。均采用 5e−5，Joint 只报告最终五语平均，1e−5 对照和早期轨迹在后面的表格与引用中列出。

| 基准 | 编号 | 条件 | 完成度 | AP | BWT | FWT | 习得 | 负 BWT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MASSIVE | A1 | LoRA Sequential | 完整矩阵 | 61.05 [+59.84, +62.25] | +0.05 [-0.76, +0.85] | +28.80 [+27.52, +30.04] | +24.41 [+23.48, +25.36] | 1.35 [+0.81, +2.00] |
| MASSIVE | A2 | LoRA Replay | 完整矩阵 | 61.81 [+60.58, +63.00] | +0.99 [+0.17, +1.84] | +29.75 [+28.47, +31.00] | +23.65 [+22.73, +24.61] | 0.93 [+0.47, +1.60] |
| MASSIVE | A3 | LoRA Joint | 五语终点 | 63.85 | — | — | — | — |
| MASSIVE | B1 | Full-FT Sequential | 完整矩阵 | 28.41 [+27.20, +29.67] | -23.72 [-25.04, -22.35] | -6.60 [-7.62, -5.57] | +39.10 [+37.82, +40.37] | 23.72 [+22.35, +25.04] |
| MASSIVE | B2 | Full-FT Replay | 完整矩阵 | 44.57 [+43.30, +45.81] | -2.80 [-3.86, -1.77] | -5.24 [-6.30, -4.16] | +37.44 [+36.22, +38.64] | 3.74 [+2.86, +4.63] |
| MASSIVE | B3 | Full-FT Joint | 五语终点 | 49.86 | — | — | — | — |
| MasakhaNER 2.0 | A1 | LoRA Sequential | 完整矩阵 | 67.37 [+66.56, +68.21] | -1.04 [-1.85, -0.20] | +51.45 [+50.36, +52.54] | +27.05 [+26.20, +27.93] | 1.71 [+1.20, +2.35] |
| MasakhaNER 2.0 | A2 | LoRA Replay | 完整矩阵 | 67.87 [+67.07, +68.70] | +0.15 [-0.68, +0.93] | +51.38 [+50.26, +52.46] | +26.65 [+25.76, +27.52] | 0.67 [+0.30, +1.21] |
| MasakhaNER 2.0 | A3 | LoRA Joint | 五语终点 | 71.41 | — | — | — | — |
| MasakhaNER 2.0 | B1 | Full-FT Sequential | 完整矩阵 | 57.89 [+57.01, +58.75] | -11.22 [-12.30, -10.14] | +41.85 [+40.80, +42.90] | +33.38 [+32.39, +34.36] | 11.22 [+10.14, +12.30] |
| MasakhaNER 2.0 | B2 | Full-FT Replay | 完整矩阵 | 61.22 [+60.38, +62.10] | +1.29 [+0.18, +2.40] | +39.09 [+38.07, +40.11] | +28.92 [+27.83, +30.02] | 2.99 [+2.21, +3.89] |
| MasakhaNER 2.0 | B3 | Full-FT Joint | 五语终点 | 67.65 | — | — | — | — |
| MTOP | A1 | LoRA Sequential | 完整矩阵 | 59.71 [+58.72, +60.69] | +1.66 [+0.84, +2.47] | +37.47 [+36.55, +38.36] | +23.85 [+23.11, +24.58] | 2.81 [+2.19, +3.48] |
| MTOP | A2 | LoRA Replay | 完整矩阵 | 61.39 [+60.39, +62.36] | +3.96 [+3.14, +4.76] | +37.39 [+36.52, +38.26] | +23.75 [+23.06, +24.47] | 1.17 [+0.78, +1.69] |
| MTOP | A3 | LoRA Joint | 五语终点 | 68.63 | — | — | — | — |
| MTOP | B1 | Full-FT Sequential | 完整矩阵 | 50.57 [+49.63, +51.47] | -4.32 [-5.24, -3.41] | +31.03 [+30.15, +31.91] | +24.64 [+23.88, +25.42] | 4.94 [+4.17, +5.69] |
| MTOP | B2 | Full-FT Replay | 完整矩阵 | 56.07 [+55.07, +57.01] | +2.10 [+1.26, +2.95] | +34.59 [+33.71, +35.45] | +22.15 [+21.39, +22.91] | 1.32 [+0.74, +2.00] |
| MTOP | B3 | Full-FT Joint | 五语终点 | 55.25 | — | — | — | — |

## 矩阵覆盖

| 基准与轨迹 | 划分 | 已评分格 | 热力图 |
| --- | --- | ---: | --- |
| MASSIVE Full-FT Replay · 原序 | 官方开发 | 30/30 | [三个阶段矩阵](figures/massive-MASSIVE-Full-FT-Replay--原序-dev.png) |
| MASSIVE Full-FT Replay · 原序 | 官方测试 | 30/30 | [三个阶段矩阵](figures/massive-MASSIVE-Full-FT-Replay--原序-test.png) |
| MASSIVE Full-FT Replay · 换序 | 官方开发 | 30/30 | [三个阶段矩阵](figures/massive-MASSIVE-Full-FT-Replay--换序-dev.png) |
| MASSIVE Full-FT Replay · 换序 | 官方测试 | 30/30 | [三个阶段矩阵](figures/massive-MASSIVE-Full-FT-Replay--换序-test.png) |
| MasakhaNER 2.0 LoRA Sequential · 新课程 | 官方开发 | 30/30 | [三个阶段矩阵](figures/ner-MasakhaNER-2.0-LoRA-Sequential--新课程-dev.png) |
| MasakhaNER 2.0 LoRA Replay · 新课程 | 官方开发 | 30/30 | [三个阶段矩阵](figures/ner-MasakhaNER-2.0-LoRA-Replay--新课程-dev.png) |
| MasakhaNER 2.0 Full-FT Sequential · 新课程 | 官方开发 | 30/30 | [三个阶段矩阵](figures/ner-MasakhaNER-2.0-Full-FT-Sequential--新课程-dev.png) |
| MasakhaNER 2.0 Full-FT Replay · 新课程 | 官方开发 | 30/30 | [三个阶段矩阵](figures/ner-MasakhaNER-2.0-Full-FT-Replay--新课程-dev.png) |
| MasakhaNER 2.0 LoRA Sequential · 新课程 | 官方测试 | 30/30 | [三个阶段矩阵](figures/ner-MasakhaNER-2.0-LoRA-Sequential--新课程-test.png) |
| MasakhaNER 2.0 LoRA Replay · 新课程 | 官方测试 | 30/30 | [三个阶段矩阵](figures/ner-MasakhaNER-2.0-LoRA-Replay--新课程-test.png) |
| MasakhaNER 2.0 Full-FT Sequential · 新课程 | 官方测试 | 30/30 | [三个阶段矩阵](figures/ner-MasakhaNER-2.0-Full-FT-Sequential--新课程-test.png) |
| MasakhaNER 2.0 Full-FT Replay · 新课程 | 官方测试 | 30/30 | [三个阶段矩阵](figures/ner-MasakhaNER-2.0-Full-FT-Replay--新课程-test.png) |
| MTOP LoRA Sequential · 新课程 | 官方开发 | 30/30 | [三个阶段矩阵](figures/mtop-MTOP-LoRA-Sequential--新课程-dev.png) |
| MTOP LoRA Replay · 新课程 | 官方开发 | 30/30 | [三个阶段矩阵](figures/mtop-MTOP-LoRA-Replay--新课程-dev.png) |
| MTOP Full-FT Sequential · 新课程 | 官方开发 | 30/30 | [三个阶段矩阵](figures/mtop-MTOP-Full-FT-Sequential--新课程-dev.png) |
| MTOP Full-FT Replay · 新课程 | 官方开发 | 30/30 | [三个阶段矩阵](figures/mtop-MTOP-Full-FT-Replay--新课程-dev.png) |
| MTOP LoRA Sequential · 新课程 | 官方测试 | 30/30 | [三个阶段矩阵](figures/mtop-MTOP-LoRA-Sequential--新课程-test.png) |
| MTOP LoRA Replay · 新课程 | 官方测试 | 30/30 | [三个阶段矩阵](figures/mtop-MTOP-LoRA-Replay--新课程-test.png) |
| MTOP Full-FT Sequential · 新课程 | 官方测试 | 30/30 | [三个阶段矩阵](figures/mtop-MTOP-Full-FT-Sequential--新课程-test.png) |
| MTOP Full-FT Replay · 新课程 | 官方测试 | 30/30 | [三个阶段矩阵](figures/mtop-MTOP-Full-FT-Replay--新课程-test.png) |

## 完整轨迹的指标

| 轨迹 | 划分 | AP | BWT | FWT | 习得 | 峰值遗忘 | 负 BWT | 明确习得 | 明确后续回落 |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| MASSIVE Full-FT Replay · 原序 | 官方开发 | 45.09 [+43.57, +46.61] | -1.82 [-3.06, -0.57] | -6.07 [-7.38, -4.73] | +36.66 [+35.16, +38.18] | 3.03 [+2.16, +4.18] | 3.03 [+2.14, +4.11] | 5/5 | 3/4 |
| MASSIVE Full-FT Replay · 原序 | 官方测试 | 44.57 [+43.30, +45.81] | -2.80 [-3.86, -1.77] | -5.24 [-6.30, -4.16] | +37.44 [+36.22, +38.64] | 3.74 [+2.89, +4.78] | 3.74 [+2.86, +4.63] | 5/5 | 3/4 |
| MASSIVE Full-FT Replay · 换序 | 官方开发 | 41.95 [+40.40, +43.53] | -4.72 [-6.07, -3.35] | -6.02 [-7.33, -4.67] | +35.80 [+34.29, +37.31] | 5.76 [+4.60, +7.20] | 5.04 [+4.03, +6.24] | 5/5 | 2/4 |
| MASSIVE Full-FT Replay · 换序 | 官方测试 | 42.15 [+40.88, +43.46] | -4.78 [-5.89, -3.64] | -5.05 [-6.11, -3.98] | +36.45 [+35.20, +37.66] | 5.98 [+4.85, +7.16] | 4.78 [+3.76, +5.90] | 5/5 | 3/4 |
| MasakhaNER 2.0 LoRA Sequential · 新课程 | 官方开发 | 64.65 [+62.64, +66.73] | -3.02 [-4.93, -1.24] | +47.02 [+44.67, +49.38] | +29.45 [+27.47, +31.54] | 3.74 [+2.32, +5.56] | 3.35 [+1.91, +5.17] | 5/5 | 2/4 |
| MasakhaNER 2.0 LoRA Replay · 新课程 | 官方开发 | 66.10 [+64.09, +68.03] | -1.47 [-3.83, +0.79] | +46.31 [+43.82, +48.69] | +30.23 [+28.29, +32.27] | 3.68 [+1.97, +5.95] | 2.41 [+1.00, +4.47] | 5/5 | 2/4 |
| MasakhaNER 2.0 Full-FT Sequential · 新课程 | 官方开发 | 58.38 [+56.29, +60.41] | -9.58 [-11.81, -7.34] | +36.43 [+34.13, +38.70] | +36.90 [+34.87, +39.08] | 10.63 [+8.60, +12.89] | 9.58 [+7.34, +11.81] | 5/5 | 4/4 |
| MasakhaNER 2.0 Full-FT Replay · 新课程 | 官方开发 | 61.01 [+58.92, +63.03] | -0.16 [-2.59, +2.32] | +31.93 [+29.85, +34.09] | +35.59 [+33.38, +37.66] | 4.37 [+3.14, +6.59] | 3.80 [+2.30, +5.56] | 5/5 | 3/4 |
| MasakhaNER 2.0 LoRA Sequential · 新课程 | 官方测试 | 67.37 [+66.56, +68.21] | -1.04 [-1.85, -0.20] | +51.45 [+50.36, +52.54] | +27.05 [+26.20, +27.93] | 2.28 [+1.83, +2.93] | 1.71 [+1.20, +2.35] | 5/5 | 2/4 |
| MasakhaNER 2.0 LoRA Replay · 新课程 | 官方测试 | 67.87 [+67.07, +68.70] | +0.15 [-0.68, +0.93] | +51.38 [+50.26, +52.46] | +26.65 [+25.76, +27.52] | 1.86 [+1.39, +2.59] | 0.67 [+0.30, +1.21] | 5/5 | 1/4 |
| MasakhaNER 2.0 Full-FT Sequential · 新课程 | 官方测试 | 57.89 [+57.01, +58.75] | -11.22 [-12.30, -10.14] | +41.85 [+40.80, +42.90] | +33.38 [+32.39, +34.36] | 11.65 [+10.63, +12.70] | 11.22 [+10.14, +12.30] | 5/5 | 4/4 |
| MasakhaNER 2.0 Full-FT Replay · 新课程 | 官方测试 | 61.22 [+60.38, +62.10] | +1.29 [+0.18, +2.40] | +39.09 [+38.07, +40.11] | +28.92 [+27.83, +30.02] | 3.58 [+2.78, +4.59] | 2.99 [+2.21, +3.89] | 4/5 | 3/4 |
| MTOP LoRA Sequential · 新课程 | 官方开发 | 57.24 [+55.92, +58.58] | +1.03 [-0.09, +2.15] | +35.81 [+34.66, +36.96] | +23.27 [+22.34, +24.19] | 6.15 [+5.09, +7.24] | 3.51 [+2.64, +4.37] | 5/5 | 3/4 |
| MTOP LoRA Replay · 新课程 | 官方开发 | 59.20 [+57.93, +60.54] | +3.53 [+2.45, +4.60] | +35.67 [+34.49, +36.80] | +23.35 [+22.40, +24.29] | 3.54 [+2.63, +4.68] | 1.35 [+0.80, +2.17] | 5/5 | 1/4 |
| MTOP Full-FT Sequential · 新课程 | 官方开发 | 48.62 [+47.33, +49.95] | -3.95 [-5.21, -2.73] | +30.19 [+29.00, +31.36] | +23.13 [+22.11, +24.19] | 5.39 [+4.51, +6.71] | 5.39 [+4.40, +6.38] | 5/5 | 3/4 |
| MTOP Full-FT Replay · 新课程 | 官方开发 | 54.03 [+52.72, +55.41] | +2.51 [+1.41, +3.69] | +32.79 [+31.63, +33.97] | +21.29 [+20.28, +22.34] | 1.34 [+0.69, +2.24] | 1.34 [+0.68, +2.23] | 5/5 | 1/4 |
| MTOP LoRA Sequential · 新课程 | 官方测试 | 59.71 [+58.72, +60.69] | +1.66 [+0.84, +2.47] | +37.47 [+36.55, +38.36] | +23.85 [+23.11, +24.58] | 6.00 [+5.27, +6.75] | 2.81 [+2.19, +3.48] | 5/5 | 3/4 |
| MTOP LoRA Replay · 新课程 | 官方测试 | 61.39 [+60.39, +62.36] | +3.96 [+3.14, +4.76] | +37.39 [+36.52, +38.26] | +23.75 [+23.06, +24.47] | 4.22 [+3.42, +4.98] | 1.17 [+0.78, +1.69] | 5/5 | 1/4 |
| MTOP Full-FT Sequential · 新课程 | 官方测试 | 50.57 [+49.63, +51.47] | -4.32 [-5.24, -3.41] | +31.03 [+30.15, +31.91] | +24.64 [+23.88, +25.42] | 5.09 [+4.35, +6.19] | 4.94 [+4.17, +5.69] | 5/5 | 4/4 |
| MTOP Full-FT Replay · 新课程 | 官方测试 | 56.07 [+55.07, +57.01] | +2.10 [+1.26, +2.95] | +34.59 [+33.71, +35.45] | +22.15 [+21.39, +22.91] | 1.32 [+0.74, +2.00] | 1.32 [+0.74, +2.00] | 5/5 | 3/4 |

## Replay 相对同参数更新的 Sequential

每个格是 Replay 减去同顺序、同题的 Sequential。负 BWT 的降低与训练当期的习得需要同时解读。

| Replay 比较 | 划分 | AP 差 | BWT 差 | 习得差 | 负 BWT 差 |
| --- | --- | --- | --- | --- | --- |
| MASSIVE Full-FT Replay · 原序 | 官方开发 | +16.74 [+15.37, +18.11] | +22.01 [+20.10, +23.88] | -2.22 [-3.30, -1.14] | -20.81 [-22.55, -18.85] |
| MASSIVE Full-FT Replay · 原序 | 官方测试 | +16.16 [+15.01, +17.31] | +20.91 [+19.37, +22.42] | -1.66 [-2.56, -0.76] | -19.98 [-21.49, -18.39] |
| MASSIVE Full-FT Replay · 换序 | 官方开发 | +15.18 [+13.88, +16.49] | +21.15 [+19.44, +22.82] | -3.18 [-4.21, -2.13] | -20.83 [-22.32, -19.08] |
| MASSIVE Full-FT Replay · 换序 | 官方测试 | +14.88 [+13.86, +15.91] | +21.00 [+19.63, +22.45] | -3.25 [-4.16, -2.40] | -21.00 [-22.36, -19.59] |
| MasakhaNER 2.0 LoRA Replay · 新课程 | 官方开发 | +1.45 [-0.08, +3.03] | +1.55 [-0.45, +3.64] | +0.77 [-0.57, +2.18] | -0.95 [-2.65, +0.91] |
| MasakhaNER 2.0 Full-FT Replay · 新课程 | 官方开发 | +2.63 [+0.73, +4.50] | +9.42 [+6.32, +12.57] | -1.31 [-3.86, +1.15] | -5.78 [-8.27, -3.03] |
| MasakhaNER 2.0 LoRA Replay · 新课程 | 官方测试 | +0.50 [-0.02, +1.01] | +1.20 [+0.46, +1.94] | -0.40 [-1.00, +0.21] | -1.04 [-1.58, -0.46] |
| MasakhaNER 2.0 Full-FT Replay · 新课程 | 官方测试 | +3.33 [+2.43, +4.20] | +12.51 [+11.21, +13.83] | -4.47 [-5.67, -3.34] | -8.24 [-9.40, -7.03] |
| MTOP LoRA Replay · 新课程 | 官方开发 | +1.96 [+1.33, +2.57] | +2.50 [+1.64, +3.35] | +0.08 [-0.58, +0.75] | -2.16 [-2.79, -1.31] |
| MTOP Full-FT Replay · 新课程 | 官方开发 | +5.42 [+4.35, +6.51] | +6.47 [+5.10, +7.90] | -1.83 [-2.81, -0.88] | -4.05 [-5.15, -2.83] |
| MTOP LoRA Replay · 新课程 | 官方测试 | +1.69 [+1.20, +2.16] | +2.30 [+1.63, +2.94] | -0.10 [-0.59, +0.41] | -1.65 [-2.18, -1.06] |
| MTOP Full-FT Replay · 新课程 | 官方测试 | +5.49 [+4.66, +6.32] | +6.42 [+5.32, +7.52] | -2.49 [-3.23, -1.74] | -3.62 [-4.51, -2.65] |

## 已评分控制组终点

表中逐语言分成绩使用同一套固定题。Single 由每个语言各自的模型组成。Joint 为同一个混合训练模型；其 AP 不用于算阶段指标。

| 基准 | 划分 | 条件 | 已评分 | 宏平均 |
| --- | --- | --- | ---: | ---: |
| MASSIVE | 官方开发 | Single LoRA | 5/5 | 54.81 |
| MASSIVE | 官方开发 | Joint LoRA | 5/5 | 64.51 |
| MASSIVE | 官方开发 | Single Full-SFT | 5/5 | 44.02 |
| MASSIVE | 官方开发 | Joint Full-SFT | 5/5 | 49.74 |
| MASSIVE | 官方开发 | Full-FT Joint 1e−5 | 5/5 | 65.57 |
| MASSIVE | 官方开发 | Full-FT Single 1e−5 | 5/5 | 59.32 |
| MASSIVE | 官方测试 | Single LoRA | 5/5 | 53.81 |
| MASSIVE | 官方测试 | Joint LoRA | 5/5 | 63.85 |
| MASSIVE | 官方测试 | Single Full-SFT | 5/5 | 44.26 |
| MASSIVE | 官方测试 | Joint Full-SFT | 5/5 | 49.86 |
| MASSIVE | 官方测试 | Full-FT Joint 1e−5 | 5/5 | 63.87 |
| MASSIVE | 官方测试 | Full-FT Single 1e−5 | 5/5 | 58.81 |
| MasakhaNER 2.0 | 官方开发 | LoRA Joint | 5/5 | 69.99 |
| MasakhaNER 2.0 | 官方开发 | Full-FT Joint | 5/5 | 65.27 |
| MasakhaNER 2.0 | 官方开发 | Full-FT Single | 5/5 | 57.41 |
| MasakhaNER 2.0 | 官方开发 | LoRA Single | 5/5 | 53.88 |
| MasakhaNER 2.0 | 官方测试 | LoRA Joint | 5/5 | 71.41 |
| MasakhaNER 2.0 | 官方测试 | Full-FT Joint | 5/5 | 67.65 |
| MasakhaNER 2.0 | 官方测试 | Full-FT Single | 5/5 | 57.82 |
| MasakhaNER 2.0 | 官方测试 | LoRA Single | 5/5 | 56.25 |
| MTOP | 官方开发 | LoRA Joint | 5/5 | 66.36 |
| MTOP | 官方开发 | Full-FT Joint | 5/5 | 53.34 |
| MTOP | 官方开发 | Full-FT Single | 5/5 | 46.50 |
| MTOP | 官方开发 | LoRA Single | 5/5 | 38.05 |
| MTOP | 官方测试 | LoRA Joint | 5/5 | 68.63 |
| MTOP | 官方测试 | Full-FT Joint | 5/5 | 55.25 |
| MTOP | 官方测试 | Full-FT Single | 5/5 | 48.68 |
| MTOP | 官方测试 | LoRA Single | 5/5 | 39.77 |

## 控制组与顺序模型的配对终点差

终点将同一语言的同题分数先配对，再对全部语言等权汇总。正数表示条件比参照得分高。

| 基准 | 划分 | 条件 | 参照 | AP 差与 95% 区间 |
| --- | --- | --- | --- | --- |
| MASSIVE | 官方开发 | Full-FT Joint 1e−5 | MASSIVE Full-SFT · 原顺序 · 1e−5 | +2.47 [+1.51, +3.44] |
| MASSIVE | 官方开发 | Full-FT Single 1e−5 | MASSIVE Full-SFT · 原顺序 · 1e−5 | -3.78 [-4.70, -2.81] |
| MASSIVE | 官方开发 | Full-FT Joint 1e−5 | Joint Full-SFT | +15.82 [+14.69, +17.01] |
| MASSIVE | 官方开发 | Full-FT Single 1e−5 | Single Full-SFT | +15.30 [+14.13, +16.43] |
| MASSIVE | 官方开发 | Joint Full-SFT | Joint LoRA | -14.77 [-16.07, -13.53] |
| MASSIVE | 官方开发 | Single Full-SFT | Single LoRA | -10.79 [-12.02, -9.51] |
| MASSIVE | 官方开发 | Single LoRA | MASSIVE SeqLoRA · 原顺序 | -6.34 [-7.33, -5.32] |
| MASSIVE | 官方开发 | Single Full-SFT | MASSIVE Full-SFT · 原顺序 | +15.67 [+14.21, +17.18] |
| MASSIVE | 官方开发 | Joint LoRA | MASSIVE SeqLoRA · 原顺序 | +3.36 [+2.52, +4.20] |
| MASSIVE | 官方开发 | Joint Full-SFT | MASSIVE Full-SFT · 原顺序 | +21.39 [+19.88, +22.87] |
| MASSIVE | 官方测试 | Full-FT Joint 1e−5 | MASSIVE Full-SFT · 原顺序 · 1e−5 | +1.21 [+0.38, +2.09] |
| MASSIVE | 官方测试 | Full-FT Single 1e−5 | MASSIVE Full-SFT · 原顺序 · 1e−5 | -3.84 [-4.69, -3.02] |
| MASSIVE | 官方测试 | Full-FT Joint 1e−5 | Joint Full-SFT | +14.01 [+13.10, +14.94] |
| MASSIVE | 官方测试 | Full-FT Single 1e−5 | Single Full-SFT | +14.55 [+13.61, +15.50] |
| MASSIVE | 官方测试 | Joint Full-SFT | Joint LoRA | -13.99 [-15.01, -12.97] |
| MASSIVE | 官方测试 | Single Full-SFT | Single LoRA | -9.55 [-10.56, -8.51] |
| MASSIVE | 官方测试 | Single LoRA | MASSIVE SeqLoRA · 原顺序 | -7.24 [-8.08, -6.41] |
| MASSIVE | 官方测试 | Single Full-SFT | MASSIVE Full-SFT · 原顺序 | +15.84 [+14.59, +17.11] |
| MASSIVE | 官方测试 | Joint LoRA | MASSIVE SeqLoRA · 原顺序 | +2.80 [+2.08, +3.51] |
| MASSIVE | 官方测试 | Joint Full-SFT | MASSIVE Full-SFT · 原顺序 | +21.44 [+20.17, +22.72] |
| MasakhaNER 2.0 | 官方开发 | MasakhaNER 2.0 LoRA Replay · 新课程 | MasakhaNER 2.0 LoRA Sequential · 新课程 | +1.45 [-0.08, +3.03] |
| MasakhaNER 2.0 | 官方开发 | MasakhaNER 2.0 Full-FT Sequential · 新课程 | MasakhaNER 2.0 LoRA Sequential · 新课程 | -6.27 [-8.26, -4.20] |
| MasakhaNER 2.0 | 官方开发 | MasakhaNER 2.0 Full-FT Replay · 新课程 | MasakhaNER 2.0 LoRA Sequential · 新课程 | -3.65 [-5.71, -1.65] |
| MasakhaNER 2.0 | 官方开发 | LoRA Joint | MasakhaNER 2.0 LoRA Sequential · 新课程 | +5.34 [+3.88, +6.84] |
| MasakhaNER 2.0 | 官方开发 | Full-FT Joint | MasakhaNER 2.0 LoRA Sequential · 新课程 | +0.61 [-1.44, +2.73] |
| MasakhaNER 2.0 | 官方开发 | Full-FT Single | MasakhaNER 2.0 LoRA Sequential · 新课程 | -7.24 [-9.50, -4.80] |
| MasakhaNER 2.0 | 官方开发 | LoRA Single | MasakhaNER 2.0 LoRA Sequential · 新课程 | -10.77 [-12.75, -8.73] |
| MasakhaNER 2.0 | 官方开发 | Full-FT Joint | MasakhaNER 2.0 Full-FT Sequential · 新课程 | +6.88 [+4.93, +8.92] |
| MasakhaNER 2.0 | 官方开发 | Full-FT Single | MasakhaNER 2.0 Full-FT Sequential · 新课程 | -0.97 [-3.13, +1.21] |
| MasakhaNER 2.0 | 官方开发 | Full-FT Joint | LoRA Joint | -4.73 [-6.67, -2.78] |
| MasakhaNER 2.0 | 官方开发 | Full-FT Single | LoRA Single | +3.53 [+1.29, +5.87] |
| MasakhaNER 2.0 | 官方测试 | MasakhaNER 2.0 LoRA Replay · 新课程 | MasakhaNER 2.0 LoRA Sequential · 新课程 | +0.50 [-0.02, +1.01] |
| MasakhaNER 2.0 | 官方测试 | MasakhaNER 2.0 Full-FT Sequential · 新课程 | MasakhaNER 2.0 LoRA Sequential · 新课程 | -9.49 [-10.40, -8.57] |
| MasakhaNER 2.0 | 官方测试 | MasakhaNER 2.0 Full-FT Replay · 新课程 | MasakhaNER 2.0 LoRA Sequential · 新课程 | -6.16 [-7.06, -5.25] |
| MasakhaNER 2.0 | 官方测试 | LoRA Joint | MasakhaNER 2.0 LoRA Sequential · 新课程 | +4.03 [+3.37, +4.69] |
| MasakhaNER 2.0 | 官方测试 | Full-FT Joint | MasakhaNER 2.0 LoRA Sequential · 新课程 | +0.28 [-0.61, +1.15] |
| MasakhaNER 2.0 | 官方测试 | Full-FT Single | MasakhaNER 2.0 LoRA Sequential · 新课程 | -9.55 [-10.52, -8.56] |
| MasakhaNER 2.0 | 官方测试 | LoRA Single | MasakhaNER 2.0 LoRA Sequential · 新课程 | -11.12 [-11.97, -10.28] |
| MasakhaNER 2.0 | 官方测试 | Full-FT Joint | MasakhaNER 2.0 Full-FT Sequential · 新课程 | +9.77 [+8.86, +10.67] |
| MasakhaNER 2.0 | 官方测试 | Full-FT Single | MasakhaNER 2.0 Full-FT Sequential · 新课程 | -0.07 [-1.07, +0.97] |
| MasakhaNER 2.0 | 官方测试 | Full-FT Joint | LoRA Joint | -3.75 [-4.56, -2.95] |
| MasakhaNER 2.0 | 官方测试 | Full-FT Single | LoRA Single | +1.57 [+0.55, +2.60] |
| MTOP | 官方开发 | MTOP LoRA Replay · 新课程 | MTOP LoRA Sequential · 新课程 | +1.96 [+1.33, +2.57] |
| MTOP | 官方开发 | MTOP Full-FT Sequential · 新课程 | MTOP LoRA Sequential · 新课程 | -8.62 [-9.80, -7.48] |
| MTOP | 官方开发 | MTOP Full-FT Replay · 新课程 | MTOP LoRA Sequential · 新课程 | -3.20 [-4.33, -2.06] |
| MTOP | 官方开发 | LoRA Joint | MTOP LoRA Sequential · 新课程 | +9.13 [+8.19, +10.08] |
| MTOP | 官方开发 | Full-FT Joint | MTOP LoRA Sequential · 新课程 | -3.90 [-5.09, -2.76] |
| MTOP | 官方开发 | Full-FT Single | MTOP LoRA Sequential · 新课程 | -10.74 [-11.85, -9.59] |
| MTOP | 官方开发 | LoRA Single | MTOP LoRA Sequential · 新课程 | -19.19 [-20.40, -18.02] |
| MTOP | 官方开发 | Full-FT Joint | MTOP Full-FT Sequential · 新课程 | +4.72 [+3.46, +5.97] |
| MTOP | 官方开发 | Full-FT Single | MTOP Full-FT Sequential · 新课程 | -2.11 [-3.35, -0.91] |
| MTOP | 官方开发 | Full-FT Joint | LoRA Joint | -13.03 [-14.17, -11.92] |
| MTOP | 官方开发 | Full-FT Single | LoRA Single | +8.45 [+7.36, +9.56] |
| MTOP | 官方测试 | MTOP LoRA Replay · 新课程 | MTOP LoRA Sequential · 新课程 | +1.69 [+1.20, +2.16] |
| MTOP | 官方测试 | MTOP Full-FT Sequential · 新课程 | MTOP LoRA Sequential · 新课程 | -9.14 [-10.01, -8.28] |
| MTOP | 官方测试 | MTOP Full-FT Replay · 新课程 | MTOP LoRA Sequential · 新课程 | -3.64 [-4.46, -2.84] |
| MTOP | 官方测试 | LoRA Joint | MTOP LoRA Sequential · 新课程 | +8.92 [+8.25, +9.57] |
| MTOP | 官方测试 | Full-FT Joint | MTOP LoRA Sequential · 新课程 | -4.46 [-5.35, -3.59] |
| MTOP | 官方测试 | Full-FT Single | MTOP LoRA Sequential · 新课程 | -11.03 [-11.87, -10.22] |
| MTOP | 官方测试 | LoRA Single | MTOP LoRA Sequential · 新课程 | -19.94 [-20.80, -19.09] |
| MTOP | 官方测试 | Full-FT Joint | MTOP Full-FT Sequential · 新课程 | +4.68 [+3.74, +5.61] |
| MTOP | 官方测试 | Full-FT Single | MTOP Full-FT Sequential · 新课程 | -1.89 [-2.76, -1.01] |
| MTOP | 官方测试 | Full-FT Joint | LoRA Joint | -13.38 [-14.19, -12.56] |
| MTOP | 官方测试 | Full-FT Single | LoRA Single | +8.91 [+8.08, +9.74] |

逐阶段变化表计算同一语言相邻两次评估的百分点差，逐格给出 95% 配对区间。未训练语言与旧语言的变化按训练位置标记。旧语言的阶段变化也可能出现在 Replay 混合训练之后。

[阶段完整评分 CSV](V2_STAGE_SCORES.csv) · [逐阶段差值与区间 CSV](V2_STAGE_CHANGES.csv) · [逐语控制组 CSV](V2_CONTROL_SCORES.csv) · [指标与区间 CSV](V2_CL_METRICS.csv) · [逐语统计明细](V2_CL_DETAILS.json) · [已核验历史基线](CL_HISTORICAL_ZH.md)
