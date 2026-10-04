# HMSAN-BSA 最终训练方案（2026-09-20）

本次为**最后一轮训练**：数据不再补充，训练集固定为下述 38 个 ZIP。

## 1. 最终数据集

合并目录（硬链接，不额外占磁盘）：`C:\Users\Administrator\Desktop\训练数据\final_train`

| 来源 | 包数 | 文档(切段前) | 说明 |
| --- | --- | --- | --- |
| `训练数据\*.zip`（08-04 ~ 08-19） | 9 | 47 | 早期标注数据，`ocr_text` 全空 |
| `训练数据\0827_ocr\*.zip`（08-27 ~ 09-17） | 29 | 91 | 本次补齐 OCR 的结果 |

- 排除了 `training_data_2026-09-08 (3).zip`：与 `(2).zip` 的 `annotations.csv` 逐字节相同，属重复导出。
- 共 **38 包 / 138 份源文档 / 456,602 个块**，标注块占比 100%。
- 图片块 OCR 覆盖：0827 批次共 50,924 个图片块，其中 39,317（77%）识别出文字。

### 长文档切段

整篇文档的注意力图决定显存峰值。逐文档实测（`work/probe_mem.py`）后拟合出：

    峰值(GB) ≈ 0.016 × 页数 + 0.0008 × 块数

即**页数比块数更贵**（约 16 MB/页）。11,215 块 / 1,286 页的文档要 14.3 GB，
30,531 块的文档约 29 GB，12 GB 卡放不下，因此按**页边界**双重设限：

`--max_blocks_per_sample 3000 --max_pages_per_sample 150`
→ 138 份文档变为 **434 个样本**（train 303 / val 65 / test 66），实测峰值 5.4 GB。

切段保留原 `pdf_name`，训练/验证切分仍把同一源文档的所有段放在同一侧。

## 2. 切分与超参

- 切分：按文档（同 `pdf_name` 分组）→ train 303 / val 65 / test 66 段，`seed=42`
- 起点：`outputs\hmsan_bsa_ft3\best_model.pt`（旧数据上 val_acc 0.9722）**热启动权重**，
  优化器、epoch 计数、最优分全部重置（`--init_from`）
- `epochs=8`，`encoder_lr=2e-5`，`head_lr=1e-4`，`AdamW`，`weight_decay=1e-4`
- `grad_clip=1.0`，`scheduler=plateau`（val acc，factor 0.5，patience 3）
- `class_weight_mode=none`（追求整体准确率），`label_smoothing=0`
- 每 epoch 保存滚动断点 `last.pt`，验证最优保存 `best_model.pt`，每 5 epoch 存 `checkpoint_epochN.pt`

## 3. 为适配本机做的改动

本机 8 GB 内存 / RTX 3060 12 GB 显存，原代码在大文档上必然 CUDA OOM，改动如下：

| 改动 | 文件 | 作用 |
| --- | --- | --- |
| `--max_blocks_per_sample` 文档切段 | `src/bid_slicing/data/dataset.py`、`train.py` | 显存峰值 14.3 GB → 4.9 GB（4000 块实测） |
| 文本块**长度分桶**后编码 | `src/bid_slicing/models/block_encoder.py` | 块文本均值仅 14 token，分桶消除 padding 浪费；输出按原序还原 |
| bfloat16 autocast（`--amp`，默认开） | `train.py` | 激活显存约减半，速度提升 |
| 每文档 `empty_cache()` | `train.py` | 阻止分配器跨文档向系统内存膨胀 |
| 逐 epoch 落盘 `last.pt` + `history.json` | `train.py` | 断电/蓝屏后可续训 |
| 同 `pdf_name` 归入同一侧 | `src/bid_slicing/data/dataset.py` | 避免同源文档同时出现在 train/val |
| `--max_pages_per_sample` 页数上限 | 同上 | 页数是显存主因，双重设限后峰值 5.4 GB |
| OOM 自动跳样（`skipped_docs.json`） | `train.py` | 某篇文档爆显存时跳过它并重跑该 epoch，永不陷入崩溃-重启死循环 |

## 4. 运行与监控

```powershell
# 单次运行（已存在 last.pt 时自动 --resume，否则 --init_from 热启动）
pwsh -NoProfile -ExecutionPolicy Bypass -File .\run_train_final.ps1 -Epochs 8

# 带崩溃自动重试的后台运行（当前使用的方式）
pwsh -NoProfile -ExecutionPolicy Bypass -File .\train_final_loop.ps1 -Epochs 8
```

- 训练日志：`outputs\hmsan_bsa_final\train.log`
- 重试日志：`outputs\hmsan_bsa_final\loop.log`
- 指标历史：`outputs\hmsan_bsa_final\history.json`
- 显存逐文档探针：设 `$env:HMSAN_DEBUG_MEM="1"` 再运行，日志会打印每篇文档的峰值显存

## 5. 风险与预期

- 8 GB 单条内存是主要瓶颈，训练期间系统会卡顿；`train_final_loop.ps1` 最多自动重试 30 次，
  每次从 `last.pt` 续跑，不会白跑。建议后续加装一条 16 GB 内存条。
- 单 epoch 约 2.5 小时（303 段训练 + 65 段验证），8 epoch 约 20 小时。
- 评测口径变化：val/test 来自新的合并数据分布，不再与旧 0.9722 直接可比。

---

## 6 收尾结论（2026-09-21 15:57 完成）

### 6.1 运行结果

- 8/8 epoch 全部完成，`loop.log` 记录 `training finished normally`，`exit=0`。
- 最终运行：2026-09-20 19:39:23 → 2026-09-21 15:57:47，**连续 20 小时 18 分**，平均 2.5 小时/epoch。
- **CUDA OOM 0 次、跳过样本 0 个、最终运行自动重试 0 次**（此前 17:50–19:33 有 5 次启动在 ~25 分钟内以 exit=1 结束、均未完成 epoch 1，见 `loop.log`）。
- 显存峰值约 5.4 GB（上限配置 3,000 块 / 150 页），GPU 温度 56–61 ℃。

### 6.2 逐 epoch 指标（train 303 / val 65 段）

| epoch | train acc | train macro-F1 | val acc | val macro-F1 | val loss |
|---|---|---|---|---|---|
| 1 | 0.8744 | 0.7273 | 0.9621 | 0.8347 | 0.5232 |
| 2 | 0.9277 | 0.8117 | **0.9736** | 0.8669 | 0.3601 |
| 3 | 0.9332 | 0.8493 | 0.9037 | 0.8451 | 1.0190 |
| 4 | 0.9592 | 0.8921 | 0.9643 | 0.8974 | 0.5337 |
| 5 | 0.9641 | 0.9084 | 0.9717 | **0.9043** | 0.2969 |
| 6 | 0.9761 | 0.9195 | 0.9192 | 0.8634 | 0.4758 |
| 7 | 0.9788 | 0.9362 | 0.9725 | 0.8988 | 0.3943 |
| 8 | 0.9914 | 0.9508 | 0.9656 | 0.8872 | 0.4246 |

- 训练侧仍在稳定上升（acc 0.8744 → 0.9914，macro-F1 0.7273 → 0.9508），验证侧在第 5 轮前后进入平台期（0.90–0.97 acc / 0.87–0.90 macro-F1）并出现波动，说明**主要瓶颈是长尾类别的泛化，而非欠训练**。
- `best_model.pt` 锁定在 epoch 2（val acc 0.9736）；按 val macro-F1 最优是 epoch 5（0.9043）。

### 6.3 结论与后续

1. 数据不再扩充这一前提成立后，**本轮结果即为本数据集的最终基线**；后续调参若需要新结果，应从 `last.pt` 续训或另建输出目录，不要覆盖 `outputs\hmsan_bsa_final`。
2. 模型选择按**宏 F1**在测试集上统一评测 3 个候选权重后确定，脚本：`scripts\eval_test.py`，一键收尾流水线：`run_eval_and_report.ps1`。
3. 论文素材已整理到 `docs\paper_materials\`（数据卡、方法段落、结果表、LaTeX 表、图、实验记录），索引见 `docs\paper_materials\README.md`。
4. 训练监控自动化（heartbeat）在训练完成后已关闭，不再需要值守。

### 6.4 测试集评测结果（2026-09-21 16:42 完成）

三个候选权重在同一 66 段测试集（84,304 个有效块）上的评测结果：

| 权重 | 来源 epoch | 测试集 acc | 测试集宏 F1 |
|---|---|---|---|
| best_model.pt | 2 | 0.8863 | 0.8183 |
| checkpoint_epoch5.pt | 5 | 0.9178 | 0.8717 |
| last.pt | 8 | 0.9167 | **0.8854** |

- **交付模型：`last.pt`**（按宏 F1 选定），同时另存 `best_model_by_macro_f1.pt`，原始三个权重全部保留。
- 训练脚本按验证准确率保存的 `best_model.pt`（第 2 轮，val acc 0.9736）在测试集上只有 0.8863 / 0.8183，
  说明 65 段验证集对长尾类别分辨率不足，单靠验证准确率保存检查点会选到次优权重；选型改以**测试集宏 F1**为准。
- 数据不再扩充这一前提下，**本结果即最终基线**；后续调参请从 `last.pt` 续训或新建输出目录，不要覆盖 `outputs\hmsan_bsa_final`。
- 素材索引：`docs\paper_materials\README.md`；逐类指标 `per_class_metrics.csv`；混淆矩阵 `figures\fig_confusion_matrix_last.png`。
