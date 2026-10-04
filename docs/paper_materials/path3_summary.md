# 路径 3 消融矩阵汇总

- 来源：`C:\Users\Administrator\Documents\Codex\2026-08-20\https\outputs\hmsan-bsa\outputs\path3_matrix`
- 行数：4（完成 4）
- 参照行：`full_3ep`
- 说明：所有指标取自 **val**（测试集在矩阵期间不使用，见协议锁定）。

## 0 协议审计

- [warn] 11 manifest row(s) have no result yet: ['A10_dense', 'A11_no_gates', 'A1_no_g1', 'A2_no_g2', 'A3_fixed_k', 'A4_no_gated_fusion', 'A5_standard_ffn', 'A8_no_interpage_gate', 'both_unfrozen', 'frozen_both', 'image_unfrozen']
- [warn] single seed per row: the table carries no error bars.  Differences below the epoch-to-epoch swing of one run are not interpretable; report this in the limitations section or add seeds 43/44 later.

## 1 表 2 · 架构消融（val，3 epoch，budget-matched）

| 配置 | 消融内容 | 可训练参数 | val acc (%) | val macro F1 (%) | Δacc (pp) | ΔF1 (pp) | wall (h) | 峰值显存 (MB) |
|---|---|---|---|---|---|---|---|---|
| `full_3ep` | budget-matched reference for every ablation | 104,780,611 | 96.94 | 84.16 | +0.00 | +0.00 | 5.31 | 9210 |
| `A6_no_page_memory` | CORE: Infini page memory | 104,682,051 | 96.53 | 87.74 | -0.42 | +3.58 | 5.37 | 9195 |
| `A7_no_boundary_gate` | CORE: boundary-aware reset | 104,763,458 | 95.74 | 87.39 | -1.20 | +3.24 | 7.40 | 9203 |
| `A9_local_window` | CORE: GSA -> local window | 104,714,043 | 97.42 | 87.47 | +0.48 | +3.31 | 10.90 | 9150 |

## 2 表 4 · 冻结 2×2 参数效率

| 格 | 含义 | 配置 | 可训练参数 | val acc (%) | val macro F1 (%) | wall (h) |
|---|---|---|---|---|---|---|

## 3 逐类 F1（val，最后一轮）

| class | full_3ep | A6_no_page_memory | A7_no_boundary_gate | A9_local_window |
|---|---|---|---|---|
| 0 封面页 (n=279) | 1.000 | 1.000 | 1.000 | 1.000 |
| 1 目录 (n=1359) | 0.996 | 1.000 | 1.000 | 1.000 |
| 2 商务偏差表 (n=354) | 0.999 | 1.000 | 0.999 | 0.999 |
| 3 投标保证金 (n=1571) | 0.976 | 0.966 | 0.975 | 0.982 |
| 4 关系说明 (n=396) | 0.990 | 0.997 | 0.992 | 0.997 |
| 5 基本情况表 (n=2444) | 0.992 | 0.993 | 0.992 | 0.993 |
| 6 营业执照 (n=284) | 0.758 | 0.724 | 0.747 | 0.757 |
| 7 税务证明 (n=103) | 0.709 | 0.905 | 0.891 | 0.645 |
| 8 资格证明文件 (n=10642) | 0.988 | 0.971 | 0.992 | 0.986 |
| 9 财务状况 (n=6186) | 0.979 | 0.972 | 0.945 | 0.986 |
| 10 财务凭证单 (n=70) | 0.778 | 0.819 | 0.774 | 0.793 |
| 11 资质业绩凭证单 (n=13) | 0.143 | 0.133 | 0.250 | 0.143 |
| 12 评分支撑材料 (n=33686) | 0.948 | 0.976 | 0.933 | 0.984 |
| 13 名称变更 (n=235) | 0.884 | 0.773 | 0.906 | 0.858 |
| 14 一致性承诺函 (n=160) | 0.997 | 0.979 | 0.991 | 0.973 |
| 15 十不准 (n=299) | 1.000 | 1.000 | 1.000 | 1.000 |
| 16 公章授权书 (n=119) | 0.970 | 0.979 | 0.970 | 0.983 |
| 17 其他 (n=3255) | 0.662 | 0.844 | 0.607 | 0.875 |
| 18 法定代表人授权委托书 (n=627) | 0.698 | 0.646 | 0.676 | 0.665 |

## 4 GPU 遥测（monitor.csv 窗口内峰值）

| 配置 | 峰值显存 (MB) | 峰值温度 (℃) | 峰值功耗 (W) | 采样数 | 实测 wall (h) | 预估 (h) | 尝试次数 |
|---|---|---|---|---|---|---|---|
| `full_3ep` | 10223 | 66 | 116 | 635 | 5.31 | 7.61 | 1 |
| `A6_no_page_memory` | 10071 | 65 | 108 | 642 | 5.37 | 7.61 | 1 |
| `A7_no_boundary_gate` | 10902 | 67 | 118 | 885 | 7.40 | 7.61 | 1 |
| `A9_local_window` | 9992 | 66 | 111 | 1302 | 10.90 | 7.61 | 2 |

## 5 使用说明

- 本文件由 `scripts\paper\summarize_comparison.py` 生成，请勿手改。
- 表中 `Δ` 一律相对参照行 `full_3ep`，单位 pp。
- 提交前需在论文中声明：测试集未参与任何配置或检查点选择。
