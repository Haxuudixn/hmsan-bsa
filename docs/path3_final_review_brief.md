# 路径 3 收尾整体评估清单（用户 2026-09-24 指示：跑完剩余行后整体看结果再决定后续工作）

> 用途：9 行全部跑完后，按本清单做整体评估并把结论交给用户拍板。
> 本文件是"评估口径与已知坑"的固化记录，不是训练脚本的一部分。

## 0 触发条件
`outputs\path3_matrix\matrix.log` 出现最后一行 `done A10_dense ...`，且 `Get-CimInstance Win32_Process` 里不再有 `python train.exe`/`train.py`。

## 1 必做汇总
```
C:\hmsan\.venv\Scripts\python.exe C:\Users\Administrator\Documents\Codex\2026-08-20\https\outputs\hmsan-bsa\scripts\paper\summarize_comparison.py
```
产物：`docs\paper_materials\path3_summary.json / path3_summary.md / path3_tables.tex`（只读产物，可随时重跑）。

## 2 必须一起报的四个"坑"（否则会被误导）

### 2.1 口径工件（最重要，曾导致错误结论）
`full_3ep` 的 macro F1 逐轮上升（ep1 84.16 → ep2 85.67 → ep3 86.67），但它的 val acc 最高轮是 **ep1**。
"取 acc 最高轮再配该轮 F1" 这个口径等于给参照行记了它最低的 F1、给消融行记了接近最高的 F1。
实测：A6/A7/A9 相对参照行的 ΔF1 在该口径下是 +3.58/+3.24/+3.31 pp，换成**末轮口径**塌到 +1.09/+0.92/+0.80 pp —— 口径解释了 70–75% 的表观差距。
=> 收尾评估必须同时给三种口径：best-val-acc（配对）、末轮、F1 最大轮，并明确指出排序会翻转。

### 2.2 噪声地板大于效应量
已完成行的**行内轮间波动**：acc 中位 5.82 pp（3.21–7.34），macro F1 中位 3.59 pp（1.63–6.00）。
行间 acc 极差仅 3.99 pp。单 seed、无误差棒（M5 多种子已推迟）。
=> 只把超过噪声地板的差异写成结论，其余显式标注 "within noise"。

### 2.3 结构任务没有指标（主张"组件必要性"的最大漏洞）
`train.py` 同时优化三个头：分类 + boundary(O/B/I/E) + section，`alpha_boundary=0.5`、`beta_section=0.5`，即**损失里一半权重投给两个从不评估的任务**。
`train.py` 的验证函数只返回 19 类的 acc/macro F1/逐类指标；`scripts\eval_test.py` 只输出 accuracy/macro_f1/per_class/confusion_matrix。
全仓库没有任何地方计算边界或分段指标（`boundary_logits`/`section_logits` 只在模型输出、损失和 shape 断言里出现）。
=> A6 跨页记忆 / A7 边界感知重置 / A8 跨页门控都是为结构任务设计的模块，现在只用块分类指标打分。
=> 收尾报告里建议**补算 boundary/section 指标**（checkpoint 现成，不需重训）；该脚本尚未实现。

### 2.4 协议违规：checkpoint 选择用了测试集
`scripts\eval_test.py` 的选择准则是 `"criterion": "macro_f1 on the test split"`（`best = max(models, key=macro_f1)`），
`docs\paper_materials\results.md` 的交付模型 last.pt 就是这么选出来的；
但 `path3_summary.md` 与 results.md 都声明"测试集未参与任何配置或检查点选择"。**两者矛盾，投稿前必须处理**（改声明，或改用 val 选出的 epoch5：test F1 0.8717）。

## 3 已知的其它事实（写报告时别弄错）
- 交付模型 headline（test acc 0.9167 / macro F1 0.8854）= `delivered_last8` 的 **dirty_test** 分数；在无泄漏子集 `test_clean`（29 段/32,374 块）上 F1 掉到 0.8275，而 3 epoch 的 A7/last 是 0.8542。
- A′ 无泄漏重打分（`outputs\path3_diag\leak_diag_clean.json`）覆盖 4 行 × best/last + `delivered_last8` 共 9 个 checkpoint；**A11_no_gates 与 frozen_both 尚未重打分**（缺 2 个）。
- val 与 test 类别分布不同分布（例：财务状况 train/val/test = 39,207/6,186/12,652；名称变更 2,182/235/498），val 上选出的排序不能迁移到 test。
- 长尾主导指标：支持 <100 的类平均 F1 仅 32–47%，跨行极差 14.8 pp；class 10/11 在 val 只有 70/13 块。
- 切段上限 3,000 块 / 150 页（55 份超块数、123 份超页数），"跨页记忆"实际只在段内生效。
- 训练侧：batch_size=1 无梯度累积 → 3 epoch 仅 909 次 optimizer.step()；`ReduceLROnPlateau(patience=3)` 在 3 轮内永不触发；`class_weight_mode=none`，报告指标 macro F1 并非训练目标。
- 论文 `docs\paper_materials\ablation_params.json` 协议是 text_freeze=True + image_freeze=True（full=2.51 M 可训练），与 path3 的"文本解冻"（full=104.78 M）不是同一族，数字不可混排。

## 4 收尾要请用户拍板的三件事
1. **是否补算边界/分段指标**（零成本、可能直接支撑"组件必要性"主张）
2. **是否补误差棒 / 加轮数**：建议只给 3 行补 seed 43/44（≈+15 h）；若仍不可判定，再把 A6/A7/A9/A11 提到 ≥5–8 轮（每行 +6~9 h）
3. **报告口径冻结**：建议主表用末轮 + macro F1 +（支持度 ≥100 分层），并处理 2.4 的协议违规

## 5 硬性约束
- 收尾评估全部只读；**不要自动启动**补 seed、加轮数、重打分等新计算，一律先等用户决定。
- 不要修改 `run_comparison.ps1` / `run_train_final.ps1` / `train.py`。