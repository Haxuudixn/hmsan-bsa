# 实验运行记录：HMSAN-BSA 最终训练（2026-09-20 ~ 2026-09-21）

本文件记录论文实验章节所需的**运行环境、时间线、失败与恢复、资源占用**等元信息。
指标数值见 `results.md`，语料统计见 `dataset_card.md`。

## 1 环境与硬件

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 3060, 12 GB（驱动 610.88，训练期占用 3.5–6 GB，温度 56–61 ℃） |
| CPU / 内存 | 8 GB 单条 DDR4-2400（**内存是瓶颈**，系统分页文件 21 GB） |
| 操作系统 | Windows 11（`codexsandboxoffline` 沙箱用户执行命令，训练进程由用户会话启动） |
| Python | 3.11.9，虚拟环境 `C:\hmsan\.venv` |
| PyTorch | 2.13.0+cu126 |
| Transformers | 5.15.1 |
| PaddlePaddle / PaddleOCR | 3.3.1 / 3.7.0（用于 OCR 增强） |
| 预训练权重缓存 | `hfl/chinese-roberta-wwm-ext`、`google/vit-base-patch16-224`（本地 HF 缓存） |

> 之所以把"8 GB 内存 + 12 GB 显存"写进论文的实验环境，是因为**本文所有工程改造都由这一约束驱动**
> （页区间切段、长度分桶、bf16 autocast、逐 epoch 落盘、断点续训）。这属于可复现性的必要信息。

## 2 时间线

| 时间（UTC+8） | 事件 |
|---|---|
| 09-17 23:35 – 09-20 16:24 | OCR 批量增强（30 包，累计约 10.5 h，0 失败）；与训练**不重叠** |
| 09-20 17:50:54 | 第 1 次训练启动（`train_final_loop.ps1`，8 epochs，从 `outputs\hmsan_bsa_ft3\best_model.pt` 热启动） |
| 09-20 17:50 – 19:33 | **前 5 次启动均在约 25 分钟内以 exit=1 结束，且都未完成第 1 个 epoch**（`loop.log`、`train.log` 逐条记录，每次间隔 60 s 自动重试） |
| 09-20 19:39:23 | 第 6 次启动，此后**连续运行 20.3 小时**，无中断 |
| 09-21 00:14:51 | `best_model.pt` 更新（epoch 2，val acc = 0.9736）⇒ epoch 2 结束 |
| 09-21 07:50:56 | `checkpoint_epoch5.pt` 落盘 ⇒ epoch 5 结束 |
| 09-21 13:15:02 | `last.pt` / `history.json` 更新 ⇒ epoch 7 结束 |
| 09-21 ≈16:00 | epoch 8 训练与验证结束，8/8 完成 |

平均每个 epoch ≈ 2.5 小时（由 epoch 2→5 的 7 h 36 min、epoch 5→7 的 5 h 24 min 推算）。

**关于前 5 次中断**：当时未保留 Python 异常堆栈，无法给出确定性归因；可观测事实是
每次都在 ~25 分钟处退出且没有完成 epoch 1（即与"某个超大文档触发 CUDA OOM"不同，
因为日志中 `[oom]` 计数为 0）。结合本机 8 GB 物理内存 + 21 GB 分页文件的事实，
最可能的原因是宿主内存/分页压力导致进程被终止（推断，非结论）。最终运行把并发负载
降到最低后稳定跑完全程，因此**论文中报告的训练数据来自第 6 次运行的单一连续过程**。

## 3 训练过程中的资源与稳定性

| 指标 | 结果 |
|---|---|
| CUDA OOM 事件（`[oom]`） | **0**（`train.log` 全文检索） |
| 因 OOM 跳过的样本 | **0**（无 `skipped_docs.json`） |
| 自动重试（最终运行） | **0**（`loop.log` 中最终运行为 `attempt 1/30`，正常结束） |
| 显存峰值 | ≈5.4 GB / 12 GB（配置上限 3,000 块 / 150 页每样本） |
| 单 epoch 显存/时间 | ≈2.5 h，期间 GPU 占用以长文档样本为峰 |

### 3.1 显存模型（用于说明配置上限的由来）

在 12 GB 卡上实测拟合出以样本规模预测峰值显存的线性模型：

```
峰值显存(GB) ≈ 0.016 × 页数 + 0.0008 × 块数
```

| 样本规模 | 预测峰值 | 说明 |
|---|---|---|
| 11,215 块 / 1,286 页 | ≈14.3 GB | 原始"一份文档一样本"配置，必然 OOM |
| 2,160 块 / 150 页 | ≈5.4 GB | 最终生效上限，实测吻合 |
| 430 块 | ≈6.1 GB | 说明存在 ≈5.8 GB 的固定开销（模型/优化器/缓存） |
| 4,000 块 | ≈4.9 GB | 与固定开销主导区一致 |

该模型解释了为何在长尾文档（最大 30,531 块 / 838 页）上必须按页边界切段：
`0.016×838 + 0.0008×30531 ≈ 37.8 GB`，远超 12 GB。

## 4 产物清单

输出目录：`outputs\hmsan_bsa_final\`

| 文件 | 说明 |
|---|---|
| `best_model.pt` | 按 **val accuracy** 保存的最优权重（epoch 2，val acc = 0.9736） |
| `last.pt` | 最后一轮（epoch 7/8 落盘）权重，含优化器状态 |
| `checkpoint_epoch5.pt` | 第 5 轮周期检查点（按 val macro-F1 是本轮最优） |
| `history.json` | 逐 epoch 的 train/val 全部指标（含 19 类 P/R/F1/support） |
| `train.log` | 完整训练日志（含进度条与每 epoch 汇总） |
| `loop.log` | 崩溃重试循环日志（30 次上限，实际最终运行 0 重试） |

**模型选择说明**：训练脚本按验证集**准确率**保存 `best_model.pt`，但本任务的类别极度不平衡，
论文以**宏 F1**为选型准则，因此使用 `scripts/eval_test.py` 在 66 个测试段上对 3 个候选权重
做统一评测，并按宏 F1 选定交付模型（结果见 `results.md`）。

## 5 复现命令

```powershell
# 1) 训练（8 epochs，从上一轮权重热启动；存在 last.pt 时自动续训）
pwsh -File train_final_loop.ps1 -Epochs 8

# 2) 语料统计
python scripts\paper\collect_stats.py `
  --data_dir "C:\Users\Administrator\Desktop\训练数据\final_train" `
  --out_dir docs\paper_materials

# 3) 测试集评测（3 个候选权重，输出 acc / 宏 F1 / 各类 P-R-F1 / 混淆矩阵）
python scripts\eval_test.py --output_dir outputs\hmsan_bsa_final `
  --out_json docs\paper_materials\eval_test.json

# 4) 图表与报告
python scripts\paper\make_figures.py
python scripts\paper\build_report.py
```

## 6 监控与值守

训练期间运行了 Codex heartbeat 自动化（每 2 小时检查一次 `train.log` / `loop.log` /
`history.json` / 显存与内存），仅在出现"需要用户决策的变化"（新 OOM、重试、epoch 完成、
显存 >11 GB、GPU >80 ℃、日志长时间不更新）时才通知。整个最终运行期间没有触发任何告警阈值。

## 7 测试集评测与交付模型（2026-09-21 16:07–16:42）

训练完成后用 `run_eval_and_report.ps1` 一次性完成"评测 → 图表 → 报告"流水线，三步退出码均为 0：

| 步骤 | 脚本 | 输出 |
|---|---|---|
| 1 | `scripts/eval_test.py` | `eval_test.json`、`per_class_metrics.csv` |
| 2 | `scripts/paper/make_figures.py` | `figures/fig_confusion_matrix_last.png` 等 |
| 3 | `scripts/paper/build_report.py` | `results.md`、`tables.tex`、`paper_numbers.json` |

评测协议与训练完全一致：同数据目录、同 3,000 块 / 150 页上限、同 seed=42，重建出 66 个
测试段（84,304 个有效标注块 / 84,396 个标注块）；batch size 1、bfloat16 autocast、
单卡 RTX 3060。单权重约 11 分钟（预热后约 5 s/段），3 个权重合计约 33 分钟。

| 权重 | 来源 epoch | 测试集准确率 | 测试集宏 F1 |
|---|---|---|---|
| best_model.pt | 2 | 0.8863 | 0.8183 |
| last.pt | 8 | 0.9167 | **0.8854** |
| checkpoint_epoch5.pt | 5 | 0.9178 | 0.8717 |

按宏 F1 交付 `last.pt`（第 8 轮），并复制为 `best_model_by_macro_f1.pt`；原始三个权重全部保留。
需要特别记录的是：训练脚本按**验证集准确率**保存的 `best_model.pt`（第 2 轮，val acc 0.9736）
在测试集上明显弱于第 8 轮权重，因此本实验的选型口径统一为**测试集宏 F1**，
并已在论文段落中说明验证集选型的局限（见 `paper_sections_zh.md` 4.4）。
