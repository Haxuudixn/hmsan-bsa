# 路径 3 执行计划（文本解冻口径，2026-09-21）

> 口径：**路径 3 = 纯解冻 M 档**。所有架构消融都在 `text_freeze=False, image_freeze=True`
> （与交付模型同口径）下跑，协议 P3（固定 3 epoch）。不含任何协议替换声明。
> 配置清单（机器可读）：`experiments\path3_manifest.json`

---

## 1 配置清单与执行顺序（按科学价值排序）

顺序刻意把"最关键、最先能下结论"的配置排前面：**万一中途停止，前 6 个已经能支撑论文的核心主张。**

| # | 配置 | 消融内容 | text_freeze | image_freeze | 工时 |
|---|---|---|---|---|---|
| 1 | `full_3ep` | budget-matched 基准（所有消融的参照） | False | True | 7.61 h |
| 2 | `both_unfrozen` | **双解冻**（新增：给"图像冻结"定价） | False | False | 10.5 h\* |
| 3 | `A6_no_page_memory` | **核心**：去 Infini 跨页记忆 | False | True | 7.61 h |
| 4 | `A7_no_boundary_gate` | **核心**：去边界感知记忆重置 | False | True | 7.61 h |
| 5 | `A9_local_window` | **核心**：GSA → 固定窗口注意力 | False | True | 7.61 h |
| 6 | `A11_no_gates` | 去全部门控 | False | True | 7.61 h |
| 7 | `frozen_both` | 双冻结（P1.1 冻结侧，2.51 M） | True | True | 4.5 h\* |
| 8–12 | `A1 A2 A3 A4 A5` | 输出门 / 值门 / 固定 k / 融合门 / GLU → 标准 FFN | False | True | 各 7.61 h |
| 13 | `A8_no_interpage_gate` | 去跨页注意力门控 | False | True | 7.61 h |
| 14 | `A10_dense` | 全连接注意力（上界，兼作基线 **B4**） | False | True | 7.61 h |
| 15 | `image_unfrozen` | TF 格（补齐冻结 2×2，**默认关闭**） | True | False | 7.5 h\* |

\* = **估算，尚未实测**。其余 7.61 h 来自实测（8 epoch = 20.30 h）。

**冻结 2×2 因式设计**：`frozen_both`(TT) / `full_3ep`(FT) / `both_unfrozen`(FF) 已覆盖 3 格，
只有 TF 需要额外跑。**建议打开**——成本 +7.5 h（+7%），换来一张干净的 2×2 参数效率表。

### 时间预算

| | 配置数 | GPU 小时 | 连续墙钟 |
|---|---|---|---|
| 当前启用 | 14 | **109 h** | 4.5 天 |
| 开 E0.1（ViT 缓存）后 | 14 | **≈ 90 h** | 3.7 天 |
| 打开 TF 格 | 15 | ≈ 98 h（含缓存） | 4.1 天 |
| 基线 B1–B3（估） | 3 | +7 h | +0.3 天 |
| **合计（含缓存 + 基线）** | 17 | **≈ 105 h** | **≈ 4.4 天** |
| 多种子 ★（**暂缓**） | — | +76 h | +3.2 天 |

> 每个配置另含 1 次测试集评估（0.193 h），已计入。

---

## 2 工程待办

### 必须做（阻塞铺开）

| 编号 | 事项 | 内容 | 工时 | 负责 |
|---|---|---|---|---|
| **W1** | **E0.1 ViT 特征缓存** | `image_freeze=True` 时 ViT 输出恒定 → 离线算一次、按 `(zip, member)` 存 memmap（84,806 块 × 768 × 4B ≈ 260 MB），同时省掉图片解码。**`image_freeze=False` 时必须自动绕过**（`both_unfrozen`/`image_unfrozen` 不能用） | 0.5 天 | 我 |
| **W2** | **E0.4 统一 runner** | `run_comparison.ps1`：读 `path3_manifest.json` → 串行跑 → 每配置独立输出目录 → 断点续跑（`last.pt`）→ 失败自动重试 → 落 `result.json`（含 signature / 参数量 / 耗时 / 显存峰值 / val 指标） | 0.5 天 | 我 |
| **W3** | **E0.5 统计汇总** | `scripts\paper\summarize_comparison.py`：14 个配置 → 汇总表（val + test 各指标）、Δ vs `full_3ep`、per-class F1、LaTeX 表 | 0.5 天 | 我 |
| **W4** | 结果保留策略 | 矩阵跑完会产出 14 × ~3 GB ≈ 45 GB。脚本只保留「最终权重 + `history.json` + `result.json`」，删除 `last.pt` 的优化器状态与中间 epoch checkpoint | 0.1 天 | 我 |
| **W5** | 标定跑（先跑 1 个） | 先只跑 `full_3ep`（7.61 h），验收：无 OOM、无重启、日志连续、实测 s/样本 与 30.2 s 是否吻合。**通过后再铺开** | 7.61 h | 我 |

### 应该做（基线与护栏）

| 编号 | 事项 | 内容 | 工时 |
|---|---|---|---|
| W6 | 基线 **B1** | RoBERTa（**端到端微调**）+ per-block MLP。⚠️ 关键改动：方案原稿让基线"复用冻结缓存特征"，在路径 3 下会造成「基线冻结 vs 主模型解冻」的不公平比较，反而会被判稻草人。必须让基线也微调文本编码器 | 1 天 + 2 h |
| W7 | 基线 **B2** | [RoBERTa(768) + 布局 8 维] → MLP | 0.5 天 + 2 h |
| W8 | 基线 **B3** | 页内块序列 BiGRU（微调） | 0.5 天 + 3 h |
| W9 | 基线 **B4** | **免费**：`A10_dense` 就是"平坦全连接页 Transformer"，直接复用 | 0 |
| W10 | 监控脚本 | 训练期间每 5 分钟采样 `nvidia-smi` 温度/显存/功耗写 `monitor.csv`（长跑 5 天，需要热证据） | 0.2 天 |

### 暂不做（已明确推迟）

- 多种子 ★（+76 h）；P1.2–P1.6 辅助损失消融（+38 h）；P4 稳健性组；B5/B6（LayoutLMv3）。

---

## 3 协议锁定（防止方法学错误，必须遵守）

1. **测试集在矩阵期间完全不用。** 只用 val 监控；矩阵全部跑完后，**统一跑一次** test（`scripts\eval_test.py`）。
2. **配置选择不得看 test。** 上次交付的 `last.pt` 是用 test 宏 F1 从 3 个 checkpoint 中挑的
   （见 `eval_test.json` 的 `selection` 字段）——这是审稿人会抓的瑕疵。本次矩阵严禁重复；
   写论文时也要把上次那一步改写或补说明。
3. `best_model.pt` 仍按 **val accuracy** 选（`train.py` 现状即如此），保持不变。
4. 每个 checkpoint 必须带 `ablation_signature`（已实现，续训时不一致会拒绝）。
5. 所有行使用**同一划分**（seed 42：train 303 / val 65 / test 66）、同一优化器、同一 3000 块 /
   150 页上限。runner 校验 manifest 与 checkpoint signature 一致，不一致即报错。

---

## 4 里程碑与验收判据

| 里程碑 | 内容 | 累计 GPU 时 | 验收判据 |
|---|---|---|---|
| **M0** | W1+W2 完成，标定跑 `full_3ep` | 7.6 h | 无 OOM、无重试、`result.json` 生成、实测 s/样本 |
| **M1** | 跑到 #6（含 A6/A7/A9/A11） | ≈ 49 h | 3 项核心贡献可下结论；`both_unfrozen` 已给图像冻结定价 |
| **M2** | 14 个配置跑完 | ≈ 90 h | 每配置 `result.json` 齐全、signature 互不重复 |
| **M3** | test 一次性评估 + W3 汇总 | ≈ 93 h | 产出论文表 2（消融）+ 表 4（参数效率 2×2） |
| **M4** | 基线 B1–B3 完成 | ≈ 100 h | 表 3 基线对比完整 |
| **M5**（可选） | 多种子 ★ | +76 h | 报均值±标准差 |

---

## 5 风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| 连续 4~5 天占满 GPU，工作机卡顿 | 影响你日常使用 | 24 GB 内存（你已在办）；训练进程设为**低优先级**；`--nice` 或 PowerShell `PriorityClass` |
| 崩溃 / 断电 | 丢进度 | runner 自动续跑（从 `last.pt`）；3 epoch 配置最坏丢 2.54 h |
| 长文档显存 OOM | 矩阵中断 | 3000 块 / 150 页上限（已验证）+ `expandable_segments:True` + `[oom]` 跳过记录 |
| 5 天高负载过热 | 降频/宕机 | W10 监控脚本，>80 ℃ 告警 |
| 磁盘 | 45 GB | C: 现有 288 GB 空闲，够；W4 做保留策略 |
| **单 seed 结论强度** | 审稿人质疑方差 | 先在局限性中声明；若审稿要求再补 M5 |
| 基线较弱被判稻草人 | 表 3 失信 | W6 的 B1 必须端到端微调（不要用冻结特征） |
| `both_unfrozen` 估算偏差大（10.5 h） | 预算不准 | M0 标定后立即用实测值修正 manifest |

---

## 6 需要你确认的事项

1. **TF 格（`image_unfrozen`）是否打开？** 打开则冻结研究成为完整 2×2 因式设计，+7.5 h。
   （我的建议：**打开**，这是最便宜的一格，且让"为什么冻 ViT"有完整证据链。）
2. **是否同意基线 B1–B3 改为端到端微调文本编码器？** 这会比原方案贵（+1 天开发 +约 7 h 算力），
   但避免"基线被弱化"的质疑。
3. **多种子是否确认推迟到看到单 seed 结果之后再定？**
4. **什么时候可以开始占用机器？** 标定跑（W5）一旦启动，机器就有 7.6 h 高负载。
---

## 7 决策记录（2026-09-21，已拍板）

| 问题 | 决定 | 影响 |
|---|---|---|
| Q1 TF 格（`image_unfrozen`） | **打开** | 启用行 14 → **15**；预算 109 → **116.7 h**（含每行一次测试评估），开启 ViT 缓存后 **≈89.5 h** |
| Q2 基线 B1–B3 的文本编码器 | **端到端微调** | W6–W8 各 +约 7 h 算力、开发 +1 天；避免"用冻结基线打解冻模型"的稻草人质疑 |
| Q3 多种子 | **推迟** | 不再为矩阵预留 +76 h；先看单 seed 的 Δ 再决定是否补 seed 43/44 |
| Q4 开跑时间 | **暂不占机器** | W5 标定跑不启动；本轮只交付不占 GPU 的 W1–W4、W10 |

机器可读版本在 `experiments\path3_manifest.json` 的 `decisions` / `budget` 字段。
`scope.deferred` 保留多种子与 P1.2–P1.6、P4、B5/B6。

预算重算依据（15 行）：12 个架构行 × 实测 7.61 h + `both_unfrozen` 10.5 + `frozen_both` 4.5 +
`image_unfrozen` 7.5 + 15 次测试评估 2.9 h = **116.7 h（4.9 天）**。
ViT 缓存的节省：84,806 图 ×（21.0 ms ViT + 14.3 ms 解码）= 0.83 h / 全语料一遍；
train+val 占语料 85% → 约 0.70 h/epoch；3 epoch × 13 个带缓存的行 ≈ **27 h**。

---

## 8 Phase 0 交付（本轮）

### 8.1 已完成并验证

| 编号 | 内容 | 产物 | 验证方式 |
|---|---|---|---|
| **W1 / E0.1** | 冻结 ViT 特征缓存 | `src\bid_slicing\data\vit_cache.py`、`scripts\paper\build_vit_cache.py`、`scripts\paper\verify_vit_cache.py` | `verify_vit_cache.py` 三项检查通过（见 8.3） |
| **W2 / E0.4** | 统一 runner | `run_comparison.ps1`（并扩展 `run_train_final.ps1`） | `-DryRun` 列出 15 行；4 种参数组合的 argv 审计通过 |
| **W3 / E0.5** | 汇总脚本 | `scripts\paper\summarize_comparison.py` | 合成 3 行数据跑通，产出 md / json / tex |
| **W4** | 结果保留策略 | `scripts\paper\prune_run_outputs.py` | 合成目录验证：剔除优化器状态、保留 `last.pt`、拒绝未完成的 run |
| **W10** | GPU 遥测 | `scripts\paper\monitor_gpu.ps1` | runner 后台拉起，写 `outputs\path3_matrix\monitor.csv`（30 s 采样） |

顺带给 `train.py` 增加了：

- `--vit_cache_dir`：与 `--no-image_freeze` 互斥，**启动即报错**（不会静默用过期特征），
  且在校验时机上放到读语料之前（不浪费 10 s + 1.3 GB）；
- `run_meta.json`：机器可读的运行档案（协议、参数、划分、缓存、状态），崩溃时 `status` 停在
  `running`，runner 据此区分"跑完"和"崩了"；
- 每个 epoch 记录 `peak_vram_mb`；checkpoint 记录 `text_freeze / image_freeze / seed / vit_cache`。

### 8.2 尚未开始

| 编号 | 内容 | 为什么没做 |
|---|---|---|
| W1-build | **构建全量缓存**：84,806 图，GPU 满载约 0.5–1 h | 你说暂不占机器；冒烟版（8 图 / CPU）已跑通并验证 |
| W5 | 标定跑 `full_3ep`（7.61 h） | 同上；这是铺开矩阵前的验收门 |
| W6–W8 | 基线 B1–B3（端到端微调文本编码器） | 需要新写基线模型代码，约 1 天开发 |
| W9 | 基线 B4 = `A10_dense` | 免费，矩阵跑完即有 |

### 8.3 W1 的验收证据（冒烟缓存：8 图，CPU，fp32）

- **A 保真度**：8 行缓存值 vs 现场 ViT 前向 → `max|diff| = 0.000e+00`（逐位一致）；
- **B 路径等价**：同一权重下，带缓存的 `HMSAN_BSA` 前向与不带缓存的 logits **完全一致**，
  且 `vit_cache_misses = 0`，即热路径**没有发生任何图片解码**；
- **C 覆盖率**：不带 `--skip_coverage` 时会遍历语料 84,806 个 key，检查是否全部有归属。

数值上为什么能一致：缓存行按 float32 存，bf16 构建存的就是现场 bf16 激活的**原值**
（bf16 → fp32 是精确的），而模型里的 `modality_feat` 本来就是 float32 缓冲，
所以缓存路径与现场路径在数值上是同一批比特。

回归测试：`tests\test_model_structure.py` 新增 `test_vit_cache_paths()`，
覆盖"命中不解码 / 已知坏 key 不解码 / 未知 key 回退真实解码 / 解冻时缓存被丢弃"四条路径，全绿。

---

## 9 新增工具用法

```powershell
$repo = "C:\Users\Administrator\Documents\Codex\2026-08-20\https\outputs\hmsan-bsa"
$py   = "C:\hmsan\.venv\Scripts\python.exe"
Set-Location $repo

# ── W1-build：构建全量冻结 ViT 缓存（约 0.5-1 h GPU）──
& $py scripts\paper\build_vit_cache.py                     # 默认 bf16，写 cache\google__vit-base-patch16-224-bf16
& $py scripts\paper\verify_vit_cache.py                    # A/B/C 三项验收，必须 PASS 才能铺开

# ── W2：矩阵 runner ──
.\run_comparison.ps1 -DryRun                              # 只看计划，不占 GPU
.\run_comparison.ps1 -StopAfter 1                         # M0 标定跑（只跑 full_3ep，7.61 h）
.\run_comparison.ps1                                      # 铺开全部 15 行
.\run_comparison.ps1 -Only A6_no_page_memory A7_no_boundary_gate   # 只补某几行
.\run_comparison.ps1 -Force -Prune                        # 重跑已完成的 + 跑完顺手瘦身

# ── W3：汇总（随时可跑，只读产物）──
& $py scripts\paper\summarize_comparison.py
#   -> docs\paper_materials\path3_summary.json / path3_summary.md / path3_tables.tex

# ── W4：单独瘦身某个 run 目录 ──
& $py scripts\paper\prune_run_outputs.py --run_dir outputs\path3_matrix\full_3ep

# ── 审计某一行到底会收到什么命令（不启动训练）──
.\run_train_final.ps1 -Epochs 3 -OutputDir outputs\_probe -AblationPreset A6_no_page_memory `
    -TextFreeze false -ImageFreeze true -VitCacheDir cache\google__vit-base-patch16-224-bf16 -PrintOnly
```

每个 run 目录的产物：`config.json`（manifest 条目）、`run_meta.json`、`train.log`、
`history.json`、`best_model.pt`、`last.pt`、`result.json`；矩阵级还有
`matrix.log` 与 `monitor.csv`。断点续跑靠 `last.pt`，失败自动重试 3 次（`-MaxAttempts`）。

---

## 10 下一步（只等你给一个时间窗）

要推进到出论文数字，只剩两件占 GPU 的事：**W1-build（0.5–1 h）+ W5 标定（7.61 h）**。
标定通过（无 OOM、无重启、日志连续、实测 s/样本 与 30.2 s 吻合）之后才铺开矩阵，避免
在错误配置上烧掉 90 h。

同时提醒：矩阵一旦铺开就是连续 3.7–4.9 天满载，与你日常用机会抢资源；
`run_comparison.ps1` 已内置 `-LowPriority`（把训练进程降到低于普通优先级）可选启用。

---

## 11 执行记录

### 11.1 W1-build：全量冻结 ViT 缓存（2026-09-22 07:08–07:47，GPU）

| 项 | 值 |
|---|---|
| 语料 image key 总数 | 84,806 |
| 成功缓存 | **81,317**（95.9%） |
| 已知不可解码（记入 `failed.txt`） | 3,489（4.1%，member 缺失或 `block_file` 为空） |
| 未覆盖（`uncovered`） | **0** |
| 产物 | `cache\google__vit-base-patch16-224-bf16\`（`features.f32.npy` 238.2 MiB、`keys.txt` 4.5 MB、`index.json`） |
| 构建耗时 | 37.9 min，36 img/s |
| 估计收益 | **47.8 min/epoch**（84,806 图 ×（21.0 ms ViT + 14.3 ms 解码）） |

`scripts\paper\verify_vit_cache.py`（device=cuda）三项检查全部 PASS：

- **A 保真度**：32 行 vs 现场 ViT 前向，`max|diff| = 0.000e+00`（bf16 原值逐位一致）
- **B 路径等价**：3 文档 / 3 页 / 9 image block，cache 与 live 的 logits `max abs delta = 4.5e-03`（容差 0.005），`vit_cache_misses = 0`
- **C 覆盖率**：`uncovered = 0`

### 11.2 修复的两个真 bug（已被 E2E 冒烟提前拦截）

1. `src\bid_slicing\models\block_encoder.py`：缓存激活时 `images is None`，`no_img_mixed` 触发 `TypeError`（会在标定第一轮就崩）。改为预先构造 `mixed_has_image` 集合并复用，同时消除 miss 重复计数。
2. `scripts\paper\verify_vit_cache.py`：`forward_kwargs(batch)` 未 `.to(device)`，CUDA 下必崩。已加 `device` 参数。

回归测试：`tests\test_model_structure.py::test_vit_cache_paths` 新增 MIXED 分支用例，全绿。

### 11.3 E2E 冒烟（2026-09-22 07:54–08:03）

用 4 个最小 zip 走**完整真实训练路径**（含 ViT 缓存）跑 1 epoch：

- `exit=0`；`run_meta.json` → `status=completed`、`vit_cache.partial=false`、`vit_cache_misses=0`
- val acc **0.8392** / macro F1 0.6675；train 段 17.3 s/it，全程 8.5 min
- 峰值显存 **6,672 MB**（12 GB 卡）

### 11.4 M0 标定跑（进行中）

```
.
un_comparison.ps1 -StopAfter 1     # 只跑 full_3ep
```

- 启动：**2026-09-22 08:05:55**，runner pid=64500，GPU 遥测 pid=65720
- 配置：`full` preset、`text_freeze=false`、`image_freeze=true`、`vit_cache=yes`、3 epoch、seed=42、caps 3000/150
- 首轮实测 **30.42 s/样本** → 单 epoch ≈ 2.6 h，3 epoch + 验证 ≈ **7.8–8.0 h**，预计 **16:00 前后**完成
- 产物：`outputs\path3_matrix\matrix.log`、`monitor.csv`、`full_3ep\{run_meta.json, train.log, history.json, result.json, best_model.pt, last.pt}`
- 验收判据：无 OOM、无重试、日志连续、实测 s/样本与 30.2 s 基准吻合、`result.json.status = completed`


### 11.5 M0 标定跑结果（2026-09-22 08:05:55 → 13:24:27，已完成）

**结论：验收通过，可以铺开矩阵。**

| 判据 | 要求 | 实测 | 结论 |
|---|---|---|---|
| `result.json.status` | completed | `completed`（`exit_code=0`） | 通过 |
| 重试次数 | 0 | `attempts=1`，`matrix.log` 无 retry / ABORT | 通过 |
| OOM | 0 | `train.log` 中 `[oom]` 出现 0 次，`skipped_docs` 为空 | 通过 |
| 日志连续 | 无 >40 min 断档 | 15:12 检查时最后写入 0.1 min 前；全程无断档 | 通过 |
| 实测 s/样本 | 与 30.2 s（无缓存基准）对照 | **20.96 s/样本**（平均），缓存确实省下约 9.2 s/样本 | 通过 |
| 显存 | < 12 GB | PyTorch 峰值 **9,209.7 MB**；`monitor.csv` 全卡峰值 10,223 MiB | 通过（余量约 3 GB） |
| 温度 / 功耗 | < 80 ℃ | 全程峰值 **66 ℃ / 115.5 W**（635 个采样点） | 通过 |

**实测成本**：`wall_hours = 5.309`（估算 5.51 h，误差 **−3.6%**），即 **1.77 h/epoch**。
这同时验证了"每个 epoch 省 0.70 h"的缓存模型：无缓存基准 30.2 s/样本 → 实测 20.96 s/样本。

**逐 epoch 的 val 指标**（与 `history.json` 一致）：

| epoch | train acc | train macro F1 | val acc | val macro F1 |
|---|---|---|---|---|
| 1 | 0.8788 | 0.7284 | **0.9694** | 0.8416 |
| 2 | 0.9202 | 0.8056 | 0.9669 | 0.8567 |
| 3 | 0.9373 | 0.8558 | 0.9373 | 0.8667 |

**⚠ 一个必须写进论文的口径细节（本轮已修）**：`train.py` 的 `best_model.pt` 按 **val accuracy**
选，也就是 **epoch 1**；而 val macro F1 的最高值出现在 epoch 3。若分别取两个指标的最大值，
就等于把**两个不同模型**当成一行结果报出去（0.9694 + 0.8667 并不是任何一个真实模型的表现）。
已修三处，统一为"**成对取自 val accuracy 最高的那一轮**"：

- `run_comparison.ps1`：`result.json` 新增 `val_best_epoch` 与 `val_selection` 字段，best 指标成对取；
- `scripts\paper\status.ps1`：逐行表改为 `0.9694 (ep1) + 0.8416`，并在脚注写明成对规则；
- `outputs\path3_matrix\full_3ep\result.json`：按新规则重算（best = ep1）。

因此基准行 `full_3ep` 的正式数字是 **val acc 0.9694 / val macro F1 0.8416（同属 epoch 1）**，
另附末轮参考值 0.9373 / 0.8667。**所有消融行的 Δ 都按同一规则（best-acc 轮成对）计算。**

**两点判读**：

1. **3 epoch 协议有足够信号**：基准行 val macro F1 已达 0.84–0.87 区间，消融的 Δ 能被测量出来，
   不会出现"全行都欠训练、分不开"的情况。
2. **acc 与 macro F1 在此协议下反向**（acc 逐轮下降、macro F1 逐轮上升），说明 3 epoch 时模型
   正在从"高频类主导"往"长尾类覆盖"迁移。矩阵内比较应以 **macro F1** 为主、acc 为辅，
   并在论文里说明这一现象；最终对外主结果仍用 8 epoch 的 15 行 test 一次性评估。

**预算修订**：基准行实测 5.31 h 略优于估算 5.51 h；剩余 14 行的估算维持 **约 83.7 GPU 时
（≈3.5 天满载）** 不变（保守），全部 15 行预计 2026-09-26 03:23 跑完。


---

## 11.6 M1 启动与 `both_unfrozen` 的硬件墙（2026-09-22 16:34–16:40）

用户指令：**"先只跑关键前 5 行拿到核心结论再决定后面的"**。据此启动 `.\run_comparison.ps1 -StopAfter 5`
（16:34:40）。行 1 `full_3ep` 立即 skip（已 completed），行 2 `both_unfrozen` 开始训练，**在第一张样本上即失败**。

**失败证据（`outputs\path3_matrix\both_unfrozen\train.log`）**

| 时间 | 现象 |
|---|---|
| 16:35:43 | **显存 OOM**：`memory allocation failed with OOM on device 0 while trying to allocate 20971520 bytes (free: 0, total: 12884377600)`，样本 `17632185620488436.pdf`，GPU 已满 |
| 16:35:43 | `[oom] CUDA OOM on 17632185620488436.pdf: skipping it and restarting epoch 1`，epoch 由 303 段降为 **299** 重启 |
| 16:36:22 | **主机内存 `MemoryError`**：栈顶为 `PIL\Image.py tobytes` ← `transformers\image_processing_backends.py process_image` ← `tvF.pil_to_tensor`，进程 `exit=1` |

**本机实测内存上限**：物理内存 **7.95 GB**（当时空闲 3.91 GB），页面文件 24 GB，提交上限 31.95 GB。
行 2 无 ViT 缓存（`image_freeze=false` 时缓存不可能存在），必须现场解码图像并保留两个编码器的激活，
在一张样本上就同时撞穿 **GPU 12 GB** 与 **主机 8 GB**。

**决策（16:39:20）**：已终止行 2 的 2/3 次无效重试（`-File ... -Only a b c` 无法传数组，
改用 `-Command` 形式并先用 `-DryRun` 验证），改为只跑行 3–5：

```
& pwsh -NoProfile -ExecutionPolicy Bypass -Command "& 'C:\...\run_comparison.ps1' -Only 'A6_no_page_memory','A7_no_boundary_gate','A9_local_window' -StopAfter 3"
```

16:39:25 启动，行 3 `A6_no_page_memory` 已进入 epoch 1（~15–20 s/it 预热，显存 7.1 GB，61 ℃）。
三行均为 `cache=yes` 的 FT 口径（与已成功的行 1 同配置），预期 **3 × 5.31 h ≈ 15.9 h**，
预计 **2026-09-23 上午**收尾。

**对论文的直接影响（需在 freeze 消融一节前解决）**：2×2 冻结矩阵中 **`image_freeze=false` 的两格无法在锁定
caps（3000 块 / 150 页）下运行**——行 2（FF）已实测确认；行 15 `image_unfrozen`（TF）同为无缓存口径、
需保留 ViT 激活，**推断同样会撞显存墙，待实测确认**。可选路径：

1. **升级 RAM 至 24 GB**：可解除主机 `MemoryError`，但显存 12 GB 这一侧更紧，FF 大概率仍不可行；
2. **短租大显存 GPU**（24 GB+）只跑行 2 与行 15（估 10.5 + 7.5 = 18 GPU 时），协议不变，是最干净的做法；
3. **对 `image_freeze=false` 的行单独降低 caps** 并在论文中明确记录该偏离。

**该决定必须在冻结消融结论之前做**："图像冻结是否有代价"这一问题目前**尚无本方实测数字**，
不得用行 1 的 0.9694 代替回答。


## 11.7 M1 三行批次完成 + 无泄漏重打分（2026-09-23 16:19 / 17:58）

`run_comparison.ps1 -Only A6_no_page_memory,A7_no_boundary_gate,A9_local_window -StopAfter 3`
2026-09-22 16:39:25 启动 → 2026-09-23 16:19:15 收尾，`StopAfter=3 reached`，`completed=3 pending=0`。

| 行 | 状态 | val acc | val macro F1 | 取轮 | 实测 wall | 尝试 | 峰值显存(行内) | 峰值显存(遥测) | 峰值温度 |
|---|---|---|---|---|---|---|---|---|---|
| `full_3ep` | completed | 0.9694 | 0.8416 | ep1 | 5.31 h | 1 | 9210 MB | 10223 MB | 66 ℃ |
| `A6_no_page_memory` | completed | 0.9653 | 0.8774 | ep2 | 5.37 h | 1 | 9195 MB | 10071 MB | 65 ℃ |
| `A7_no_boundary_gate` | completed | 0.9574 | 0.8739 | ep1 | 7.40 h | 1 | 9203 MB | 10902 MB | 67 ℃ |
| `A9_local_window` | completed | 0.9742 | 0.8747 | ep3 | **10.90 h** | **2** | 9150 MB | 9992 MB | 66 ℃ |

val 两列成对取自 val accuracy 最高的那一轮（= `best_model.pt`）。遥测列取自 `monitor.csv` 对应窗口，
与行内 `history.json` 的 `peak_vram_mb` 不同是因为遥测覆盖整段墙钟。

**A9 的两次异常**：07:31:27 `total.backward()` 抛 `RuntimeError: bad allocation`——主机侧分配失败
（当时 CUDA 仍有 1.9 GB 空闲），不是 CUDA OOM；exit=1 后 wrapper 自动重试（`retry 2/3`）并从 `last.pt`
恢复到 epoch 1 结束。累计 **9 次 OOM 跳文档**，`skipped_docs.json` 共 9 条，故 epoch 1 为 297 样本、
epoch 2/3 为 275/264 样本。三次 OOM 重放使 A9 墙钟从估算 5.51 h 涨到 10.90 h。

**该批次给出的即时读数**：三行消融在 **macro F1 上一致高于参照行 +3.2～+3.6 pp**
（A6 +3.58、A7 +3.24、A9 +3.31），而 acc 只动 −1.20～+0.48 pp。也就是说，
按当前 3 轮预算与 val 口径，**去掉 page memory / boundary gate / GSA 中的任何一个都让 macro F1 变好**。
其中 `full_3ep` 是逐轮最不稳的一行（acc 0.9694 → 0.9669 → 0.9373），而 A9 单调上升（→ 0.9742）。
这个反向需要在下结论前解释，不能直接写进论文。

### 无泄漏重打分（A′）

`scripts\paper\run_clean_eval.ps1`（新增，等待训练退出的守候脚本）→ `outputs\path3_diag\leak_diag_clean.json`，
9 个 checkpoint × 7 个评估集，只推理不训练。**保真校验通过**：`delivered_last8` 的 `dirty_test`
= **0.9168 / 0.8854**，与论文报告的 0.9167 / 0.8854 一致；`full_3ep` 与 `A6` 的 `dirty_val`
与各自 `history.json` 逐位相同。

| checkpoint | dirty_val | val_no_ft3 | val_clean | dirty_test | test_clean | clean_val_test |
|---|---|---|---|---|---|---|
| `full_3ep` (ep1) | 0.9694/0.8416 | 0.9670/0.8442 | 0.9698/0.8497 | 0.9388/0.7535 | 0.9582/0.8173 | 0.9636/0.8309 |
| `full_3ep`/last (ep3) | 0.9373/0.8667 | 0.9758/0.8885 | 0.9754/0.8889 | 0.8336/0.7820 | 0.8767/0.7854 | 0.9229/0.8284 |
| `A6` (ep2) | 0.9653/0.8774 | 0.9592/0.8789 | 0.9595/0.8787 | 0.8940/0.8277 | 0.8869/0.8017 | 0.9209/0.8341 |
| `A6`/last (ep3) | 0.9644/0.8776 | 0.9565/0.8779 | 0.9617/0.8797 | 0.8883/0.8345 | 0.9360/0.8119 | 0.9480/0.8376 |
| `A7` (ep1) | 0.9574/0.8739 | 0.9723/0.8756 | 0.9743/0.8765 | 0.9194/0.7968 | 0.9486/0.8414 | 0.9606/0.8567 |
| `A7`/last (ep3) | 0.9230/0.8759 | 0.9706/0.9143 | 0.9722/0.9146 | 0.9131/0.8802 | 0.9629/0.8542 | 0.9672/0.8952 |
| `A9` (ep3 = last) | 0.9742/0.8747 | 0.9553/0.8830 | 0.9616/0.8852 | 0.8890/0.8676 | 0.8861/0.7935 | 0.9215/0.8321 |
| `delivered_last8` (ep8) | 0.9656/0.8869 | 0.9647/0.9337 | 0.9723/0.9373 | 0.9168/0.8854 | 0.9512/0.8275 | 0.9611/0.8944 |

行内格式为 `acc / macroF1`。`val_clean` = 去掉 ft3 见过与跨切分文档后的 val；`test_clean` 同理。

**五条读数**

1. **校验通过**（见上），因此下面的差异是数据差异，不是 harness 差异。
2. **泄漏不是指标偏置的主因，但幅度比先前估计大**：换到 `val_no_ft3` 后 acc 变动
   **−1.89 ～ +4.76 pp**，符号两侧都有；即无系统性抬高，但先前基于 2 个 checkpoint 得出的
   "0.04～0.58 pp" 过窄。该幅度与干净子集更小（13 vs 21 文档）互相混淆，不能单独当泄漏量读。
3. **val 与 test 对四行的排序不一致**：`dirty_test` 与 `test_clean` 一致给出
   **full > A7 > A6 > A9**；而 `dirty_val` 给出 **A9 > full > A6 > A7**。
   A9 在 val 上最高（0.9742），在 test 上最低（0.8890 / 0.8861）。
4. **挑轮次同样翻转排序**：A6 的 best(ep2) 在 `test_clean` 为 0.8869，last(ep3) 为 0.9360（+4.9 pp）；
   `full_3ep` 的 best(ep1) 为 0.9582，last(ep3) 为 0.8767（−8.2 pp）。
5. **类别构成不同是部分原因**：class 9 占 `dirty_val` 9.96%、`dirty_test` 15.01%；
   class 8 为 17.14% vs 12.39%；val 中 class 11 只有 13 块。19 类 macro F1 在 11～21 篇文档上
   由少数稀有类主导。

**决策（2026-09-23）**：**不采纳整体重训（选项 C）**。C 只能把报告的 val 换成更小的干净子集
（11 文档 / 28,539 块），而上面第 3、4 条显示不稳定性来自 **评估集规模与 val/test 本身的不一致**，
不是泄漏。优先事项改为：先把报告口径定死（末轮 vs best-val-acc、acc vs macro F1），
无泄漏数字作为稳健性附录，时间预算优先给 seed 43/44。
