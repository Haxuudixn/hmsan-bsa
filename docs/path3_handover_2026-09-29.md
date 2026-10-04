# 路径 3 交接（2026-09-29 停机前）

## 0. 一句话状态

6 行消融矩阵 + B1/B2/B3 基线**全部跑完**。B3 早前"macro F1 高出 `full` 6.2 pp"
经同分母复评**已作废**。Stage 5 干净复评在跑到第 1/24 个 checkpoint 时被停机截断，
**需要重跑**（约 80 min，纯推理，可随时重启）。

## 1. 已完成

| 任务 | 状态 | 结束时间 |
|---|---|---|
| 阶段 1 冒烟 `stage1_smoke_full` | ✅ | 09-27 20:20 |
| 阶段 2 pilot `stage2_pilot_full_seed42`（8 ep） | ✅ | 09-28 07:21 |
| `A11_no_gates` | ✅ | 09-28 17:13 |
| `A10_dense`（= B4） | ✅ | 09-29 05:03 |
| `A2_no_g2` / `A6_no_page_memory` / `A7_no_boundary_gate` | ✅ | 09-28 |
| `delivered` | ✅ | 09-29 07:35 |
| `B1_roberta_mlp`（4.22 h） | ✅ | 09-28 22:26 |
| `B2_roberta_layout_mlp`（3.89 h） | ✅ | 09-29 02:20 |
| `B3_page_bigru`（6.10 h） | ✅ | 09-29 08:26 |
| Stage 5 干净复评 | ⚠️ **被截断，需重跑** | — |

## 2. 结果表（块级；best = val acc 最高的那个 epoch，last = 第 8 epoch）

| 行 | best ep | acc(best) | mF1(best) | last ep | acc(last) | mF1(last) | 验集块数 |
|---|---|---|---|---|---|---|---|
| `full` (delivered) | 6 | 0.9626 | 0.8362 | 8 | 0.9476 | 0.8546 | 64,788 |
| `full` (pilot s42) | 8 | 0.9543 | 0.8395 | 8 | 0.9543 | 0.8395 | 64,788 |
| `A2_no_g2` | 8 | 0.9495 | 0.8324 | 8 | 0.9495 | 0.8324 | 64,788 |
| `A6_no_page_memory` | 7 | 0.9301 | 0.8482 | 8 | 0.9111 | **0.8715** | 64,788 |
| `A7_no_boundary_gate` | 8 | 0.9491 | 0.8266 | 8 | 0.9491 | 0.8266 | 64,788 |
| `A10_dense` (=B4) | 8 | 0.9310 | 0.8413 | 8 | 0.9310 | 0.8413 | 64,788 |
| `A11_no_gates` | 6 | 0.9333 | 0.8466 | 8 | 0.8743 | 0.8074 | 64,788 |
| B1 roberta+MLP | 3 | 0.7130 | 0.5619 | 8 | 0.7018 | 0.5518 | 64,788 |
| B2 +layout | 4 | 0.6980 | 0.5523 | 8 | 0.6905 | 0.5592 | 64,788 |
| **B3 page-BiGRU** | 7 | 0.9261 | 0.8805 | 8 | 0.8950 | 0.8670 | **59,043** ⚠️ |

**口径差异很重要**：D6 冻结口径是"报 last epoch，acc 与 macro F1 成对"。
在 last-epoch 口径下，macro F1 高于 delivered 的只有 `A6`(0.8715) 与 B3(匹配分母 0.8605)；
在 best-epoch 口径下，`A6`(0.8482)、`A11`(0.8466)、`A10`(0.8413) 三者都超过 delivered(0.8362)。

**`A11_no_gates` 需要单独注意**：best 在 ep6（0.9333/0.8466），ep8 掉到 0.8743/0.8074，
晚期坍塌幅度约 5.9 pp acc。它的"消融损失最大"这个结论在 best-epoch 口径下并不成立。

## 3. B3 分母问题与更正（重要）

**根因**：`scripts/paper/baselines.py:199` 把页截断为 `page.blocks[:48]`，导致 B3 的验集
只评到 59,043 块，而其余所有行都是 64,788 块（**−8.9%**）。这不是模型差异，是评测口径差异。

**复评验证**（同一份权重、不动训练，脚本 `/root/autodl-tmp/b3_eval_cap.py`）：

| B3 复评 | acc | macro F1 | 验集块数 |
|---|---|---|---|
| cap=48（对照组） | 0.8950 | 0.8670 | 59,043 |
| cap=3000（匹配分母） | **0.8871** | **0.8605** | **64,788** |

对照组与训练日志逐位一致（0.8950 / 0.8670 / 59,043）→ 复评方法可信。
匹配分母后：**vs `full`(delivered)，acc 输 6.05 pp，macro F1 只赢 0.59 pp**
（低于中位噪声底 1.04 pp）→ **平局**。

**早前"B3 macro F1 赢 6.2 pp"的说法作废，不得再引用。**

**尚存的错配**：上面 0.8871/0.8605 是"cap=48 训练、cap=3000 评测"的时长错配。
严格版本需要 `--max_blocks_per_page 3000` 重训（≈6 h）。方向未知，
但不可能把 0.59 pp 的平局翻成 6 pp。

## 4. 结论（可用于论文）

- 对最强基线 B3：主模型 **acc 领先 6.05 pp**，macro F1 **持平**。主张 3 成立。
- `A6_no_page_memory` 与 B3 指向同一真实模式：**页内上下文抬 macro F1，跨页记忆换 acc**。
  这是可写的发现，不是漏洞。
- 对 B1/B2：主模型在两个指标上均大幅领先（acc +24.6/+25.7 pp，macro F1 +29.5/+29.5 pp）。
- **对 B3 的写法建议**（依据 §5 分层表）：B3 的 overall macro F1 平局来自 10 个小类
  （只占 4.3% 的块）；在占 95.5% 块的 `1k-5k` + `5k+` 两档上主模型领先
  10.6 pp / 4.8 pp。**按"数据主体上更强"来写主张 3，不要依赖 overall 平局。**

## 5. 支持度分层 macro F1（离线补算，D6 要求）

口径说明：D6 冻结口径是 **last epoch**；而仓库里 `support_stratified.py` 默认按
**val acc 最优 epoch** 配对。两者结论不同（见 5.2），故两版都列。

### 5.1 last epoch（D6 口径）

| 行 | overall | <100 | 100-999 | 1k-5k | 5k+ |
|---|---|---|---|---|---|
| `full` (delivered) | 0.8546 | 0.3985 | 0.8846 | 0.9350 | 0.9516 |
| `full` (pilot s42) | 0.8395 | 0.3988 | 0.8535 | 0.9325 | 0.9622 |
| `A2_no_g2` | 0.8324 | 0.4606 | 0.8284 | 0.9305 | 0.9625 |
| `A6_no_page_memory` | 0.8715 | 0.4546 | 0.9287 | 0.8812 | 0.9457 |
| `A7_no_boundary_gate` | 0.8266 | 0.2625 | 0.8597 | 0.9257 | 0.9598 |
| `A10_dense` (=B4) | 0.8413 | 0.3872 | 0.8887 | 0.8816 | 0.9325 |
| `A11_no_gates` | 0.8074 | 0.2987 | 0.8485 | 0.9147 | 0.8662 |
| B1 roberta+MLP | 0.5518 | 0.2275 | 0.5543 | 0.6491 | 0.6299 |
| B2 +layout | 0.5592 | 0.2397 | 0.5656 | 0.6606 | 0.6158 |
| B3 page-BiGRU（cap48 评测） | 0.8670 | 0.6063 | 0.9132 | 0.8529 | 0.9054 |
| **B3 page-BiGRU（cap3000 匹配分母）** | **0.8605** | 0.6063 | 0.9110 | 0.8290 | 0.9035 |

分层构成：`<100` 2 类 / 86 块；`100-999` 10 类 / 2,797 块；`1k-5k` 4 类 / 8,506 块；`5k+` 3 类 / 53,399 块。

### 5.2 关键读数：B3 的 macro F1 "优势"几乎全部来自稀有类

在匹配分母（64,788 块）下，B3 vs `full`(delivered)：

- **B3 领先**：`<100` **+20.8 pp**（0.6063 vs 0.3985）、`100-999` **+2.6 pp**（0.9110 vs 0.8846）；
- **B3 落后**：`1k-5k` **−10.6 pp**（0.8290 vs 0.9350）、`5k+` **−4.8 pp**（0.9035 vs 0.9516）。

macro F1 是 19 个类的**未加权**均值，而 19 个类里有 10 个落在 `100-999` 档——
该档只占验集块的 4.3%。所以 B3 靠"大量小类"把 overall 抬到 0.8605，
但**在占验集 95.5% 块的两档（`1k-5k` + `5k+`）上是明显落后的**。

这比"overall 打平"更有信息量：**主张 3 建议按"主模型在数据主体上更强"来写，
而不是依赖 overall 的平局。**

### 5.3 best-epoch 版本（仓库脚本的默认口径）

| 行 | ep | overall | <100 | 100-999 | 1k-5k | 5k+ |
|---|---|---|---|---|---|---|
| `full` (delivered) | 6 | 0.8362 | 0.3251 | 0.8640 | 0.9176 | 0.9757 |
| `full` (pilot s42) | 8 | 0.8395 | 0.3988 | 0.8535 | 0.9325 | 0.9622 |
| `A2_no_g2` | 8 | 0.8324 | 0.4606 | 0.8284 | 0.9305 | 0.9625 |
| `A6_no_page_memory` | 7 | 0.8482 | 0.3545 | 0.9018 | 0.9004 | 0.9290 |
| `A7_no_boundary_gate` | 8 | 0.8266 | 0.2625 | 0.8597 | 0.9257 | 0.9598 |
| `A10_dense` (=B4) | 8 | 0.8413 | 0.3872 | 0.8887 | 0.8816 | 0.9325 |
| `A11_no_gates` | 6 | 0.8466 | 0.4486 | 0.8741 | 0.9159 | 0.9276 |
| B1 roberta+MLP | 3 | 0.5619 | 0.2920 | 0.5543 | 0.6599 | 0.6364 |
| B2 +layout | 4 | 0.5523 | 0.2503 | 0.5531 | 0.6510 | 0.6190 |
| B3 page-BiGRU（cap48 评测） | 7 | 0.8805 | 0.6027 | 0.9167 | 0.8993 | 0.9197 |

复现命令（离线、不需要 GPU）：

```
python scripts\paper\support_stratified.py --dir remote_pull_2026-09-29\outputs\path3_clean --epoch_only
```

生成物：`remote_pull_2026-09-29\support_stratified.md`（含 B3 匹配分母版）。
## 6. 未决事项

1. **B3 是否按 cap=3000 重训**（≈6 h）。建议：先不做，留作审稿应对。
2. **基线公平性（唯一实质缺口）**：B1/B2/B3 用**无类别权重的 CE**
   （`baselines.py:322,344`），主模型用 `--class_weight_mode inverse`。
   这系统性压低了基线的 macro F1，恰好是基线现在最接近的指标。
   建议重跑 B1–B3 对齐权重（≈14 h）。
3. **D7 种子范围**：grey 4 行（`A2`/`A6`/`A7`/`A10`）≈27.7 h；建议先 `A6`+`A7` ≈13.8 h。
   按冻结口径，**种子不自动排队**。
4. **Stage 5 重跑**（≈80 min）。
5. AutoDL 余额（停机前面板反复警告不足 24 h）。

## 7. 本地产物

拉回位置：`outputs\hmsan-bsa\remote_pull_2026-09-29\`

- `outputs\path3_clean\<行>\{history.json, result.json, train.log}` × 11 行
- `outputs\path3_clean\summary.{csv,json}`（⚠️ 只覆盖队列里的 4 行，不完整）
- `outputs\path3_diag\{corpus_index.json, split_manifest.json, run_clean_eval.log, _smoke_leak_diag.json}`
- `b3_cap_eval.json` — 上面 §3 的复评原始输出
- `stage5_clean_eval_partial.log` — 被截断的 Stage 5 日志（只含 A10_dense/best）
- `comparison_table.{md,json}` — 由本文件 §2 脚本从 history.json 现算
- `support_stratified.md` — §5 的分层表（离线补算，含 B3 匹配分母版）

## 8. 本机环境（2026-09-30 实测，全绿）

**结论：本机可以直接训练，不需要重建环境。**
早前"本机环境没了"的判断是错的 —— 系统 Python
（`AppData\Local\Programs\Python\Python311`）里确实没有 torch，但项目一直用的是另一个 venv。

### 8.1 本机 venv（这才是项目环境）

```
C:\hmsan\.venv\Scripts\python.exe
```

- Python 3.11.9 / **torch 2.13.0+cu126（CUDA 可用）/ transformers 5.15.1** / 113 个包
- 与 `requirements-cloud.txt` 及 runbook §5.2 的"本地已验证组合"完全一致
- `C:\hmsan` 下只有 `.venv` 和 `wheels`，**仓库仍在本 workspace**，用该 venv 的解释器跑本仓库脚本即可
- 唯一缺的包是 `matplotlib`（本就不在 `requirements-cloud.txt` 里），已补装

### 8.2 五步自检（全部通过）

| 步骤 | 命令 | 结果 |
|---|---|---|
| A GPU | `torch.cuda.get_device_name(0)` | ✅ RTX 3060 12 GB |
| B 语料与切分 | `scripts\verify_split_clean.py` | ✅ **PASS** |
| C 设备防线 | `scripts\verify_device_guards.py` | ✅ ALL PASS |
| D 单元测试 | `pytest tests -q` | ✅ **18 passed in 31 s** |
| E 断点续跑 | `scripts\verify_ckpt_resume.py` | ✅ All F13 checks passed |

步骤 B 的详细输出：

```
Corpus: 434 segments, 456602 blocks, 132 source PDFs
train 306 segs / val 66 / test 62
  train segment share = 0.7051, val = 0.1521, test = 0.1429
PASS: no source PDF straddles two splits.
val blocks  = 64,788    <- 与远端矩阵分母完全一致
test blocks = 79,943（labeled 79,851）
```

新生成的 manifest 与 09-25 的 `split_manifest.json` **逐字段完全一致**，包括
`zips_sha256_16`（语料指纹）⇒ 本机语料与远端跑出全部结果所用的语料**同源**。

### 8.3 必须先设的环境变量（否则会"假死"）

不设离线模式时，`transformers` / `huggingface_hub` 会联网检查更新，表现为
**测试卡住十几分钟或随机失败**（实测：`test_text_encoder` 第一次失败、第二次通过，
`test_image_encoder` 卡死）。两个编码器权重其实都已在本地缓存
（ViT 330 MB / RoBERTa 393 MB）。

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_HUB_DISABLE_TELEMETRY = "1"
```

加上之后 `pytest` 从"卡 15 分钟以上"变成 **31 秒 18 通过**。

### 8.4 一个无害的怪现象

用 `Start-Process` 启动 `C:\hmsan\.venv\Scripts\python.exe` 时，任务管理器里会出现
**两个 python 进程**：父进程是 venv 存根（CPU 恒为 0），子进程是 base 解释器
（命令行完全相同）。这是 Windows venv 的启动器机制，**正常**，不要误判为"环境串了"。
## 9. 恢复步骤（下次开机）

```bash
# 1. 连上实例
ssh -p <端口> root@connect.bjb1.seetacloud.com

# 2. 环境（缺一不可；Xet 经代理不通）
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1
PY=/root/autodl-tmp/envs/hmsan/bin/python   # 注意：python3 不存在

# 3. 仓库
cd /root/autodl-tmp/hmsan-bsa

# 4. Stage 5 干净复评（纯推理，只写 outputs/path3_diag/leak_diag_clean.json）
screen -dmS cleaneval bash -c "export HF_HOME=/root/autodl-tmp/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1; bash scripts/paper/run_clean_eval.sh > /root/autodl-tmp/clean_eval.log 2>&1"
# 监控：tail -f /root/autodl-tmp/clean_eval.log   （约 80 min，24 个 checkpoint）
```

**所有产物都在数据盘 `/root/autodl-tmp` 下**（仓库、HF 缓存、screen 日志、复评脚本），
关机不会丢。**请用"关机"不要用"释放"**；若想保留 SSH 但不占 GPU，用**无卡模式开机**
（Stage 5 需要 GPU，无卡模式跑不了）。

## 10. 已知坑（避免重复踩）

- `python3` 在远端**不存在**，必须用 `/root/autodl-tmp/envs/hmsan/bin/python`。
- PowerShell 会吃掉 `$(...)`、`$i`、heredoc：**写本地 .py → scp → `ssh '<python> /path.py'`**（ssh 参数用单引号）。
- SSH 会不定期掉线，加 `-o ConnectTimeout=25` 重试。
- `nstat.sh` 显示的是最后一条 **mid_epoch** 记录，不是 epoch 末；epoch 末要用 `phase == None` 的记录。
- 不要跨不同验集分母比较任何指标（本次 B3 就栽在这里）。

## 11. 本机校准实测（2026-09-30 新增）

把剩余工作搬回本机之前，先跑了一遍与远端**完全同配置**的 1-epoch 冒烟
（`stage1_smoke_full`：冷启动、seed 42、`--val_every_steps 75`、`--class_weight_mode inverse`、
`--image_freeze`、ViT 缓存命中），输出在 `outputs\path3_local_calib\stage1_smoke_full`，
控制台 `outputs\_local_calib_console.log`。墙钟 10:22:13 → 12:33:11。

### 11.1 六条验收：全部 PASS

| # | 判据 | 结果 |
|---|---|---|
| 1 | 日志含 `COLD START`，argv 里没有 `--init_from` | PASS |
| 2 | `Split: train=306, val=66, test=62` | PASS |
| 3 | `history.json` 有 `phase=mid_epoch`（step 75/150/225/300），`val` 同时含 `boundary` 与 `section`，`boundary` 含 `section_start_f1` | PASS |
| 4 | `result.json` 的 `status=completed` | PASS |
| 5 | 全程无 `Traceback` / `CUDA out of memory` / `mat2 is on cpu` | PASS |
| 6 | 确实只跑了 1 个 epoch（`epochs_completed=1`，1 条 epoch 末记录） | PASS |

### 11.2 与远端同配置逐条对照（epoch 末）

| 指标 | 本机 RTX 3060 | 远端 4090 | 差 |
|---|---|---|---|
| val accuracy | 0.6602 | 0.6562 | +0.0040 |
| val macro F1 | 0.2049 | 0.2031 | +0.0018 |
| val `valid_count` | 64788 | 64788 | 0 |
| peak VRAM | 9201 MB | 8932 MB | +269 MB |
| epoch 墙钟 | **2.183 h** | **1.248 h** | **1.75×** |

第 75 步的 mid-epoch 记录同样对得上（macro F1 差 0.0014、val_loss 差 0.0064），
说明**本机管线忠实复现远端，本机产出的数字可以直接入表**。代价是每 epoch 慢 1.75 倍。

### 11.3 由此推出的本机预算（每 epoch = 2.183 h）

| 任务 | epochs | 本机 | 远端 4090 |
|---|---|---|---|
| 单条架构行（`full` / `A2` / `A6` / `A7` / `A10` / `A11`） | 8（D2） | 17.5 h | 10.0 h |
| 基线 B1 / B2 / B3（各一条） | 3 | 各 6.5 h | 各 3.7 h |
| B3 `cap=3000`（块量 +9.7%，长页另有开销） | 3 | ≈11.5 h | ≈6.5 h |
| grey 区补种子（`A6`+`A7`，各 2 个种子） | 8×4 | ≈70 h | ≈40 h |
| Stage 5 干净复评（纯推理） | — | 不可在本机（缺 27 GB checkpoint） | ≈80 min |

**结论：本机适合跑"单条、≤12 h"的任务；种子误差棒（约 70 h）和基线权重对齐（约 20–30 h）
在租用卡上做明显更划算。**

### 11.4 已在跑

`b3_page_bigru_cap3000`（B3 稳定性测试）：把 `baselines.py` 默认的
`--max_blocks_per_page 48` 抬到 3000，同时去掉训练侧和评估侧的截断。
验集分母因此从 59,043 块恢复到 **64,788 块**，与主模型可比。
启动脚本 `outputs\_launch_b3_cap3000.ps1`，控制台 `outputs\_b3_cap3000_console.log`，
输出 `outputs\path3_local_calib\b3_page_bigru_cap3000`。
注意 `baselines.py` **没有 resume**，崩了只能从零重来。

启动方 2026-09-30 12:34:57，`Split: train=306, val=66, test=62`，
`train examples=37299  val examples=7941`。


## 12. 基线口径更正 + B3 截断实验（2026-09-30 晚）

### 12.1 重要更正：基线跑的是 8 epochs，不是 manifest 写的 3

`experiments/path3_clean_manifest.json` 里 b1/b2/b3 三行标的是 `"epochs": 3`，
但**实际交付的产物是 8 epochs**：

| 行 | result.json 的 epochs | best acc | best macro F1 | last acc | last macro F1 | 验集块数 | 墙钟 |
|---|---|---|---|---|---|---|---|
| b1_roberta_mlp | 8 | 0.7130 | 0.5619 | 0.7018 | 0.5518 | 64788 | 4.2 h |
| b2_roberta_layout_mlp | 8 | 0.6980 | 0.5790 | 0.6905 | 0.5592 | 64788 | 3.9 h |
| b3_page_bigru | 8 | 0.9261 | 0.9167 | 0.8950 | 0.8670 | **59043** | 6.1 h |

后果：`run_matrix.py --group baselines` 会照 manifest 生成 3 epochs 的 argv，
跑出来的行与已交付的基线行、以及 8-epoch 的主模型**都不可比**。
manifest 已就地更正为 `"epochs": 8` 并留下说明。

### 12.2 B3 的分母问题与截断实验

`baselines.py:199` 把每页截断到 `max_blocks_per_page`（默认 48）。同一个 ep8 checkpoint、
两种评估口径（`remote_pull_2026-09-29/b3_cap_eval.json`）：

| 口径 | 验集块数 | acc | macro F1 |
|---|---|---|---|
| cap=48（默认，即表中报告的 0.8950 / 0.8670 的来源） | 59,043 | 0.8950 | 0.8670 |
| cap=3000（匹配分母） | 64,788 | 0.8871 | 0.8605 |

本机又补了一个「**训练侧也去掉截断**」的对照
（`outputs\path3_local_calib\b3_page_bigru_cap3000`，3 epochs，未加权，墙钟 4.0 h）：

| epoch | acc | macro F1 |
|---|---|---|
| 1 | 0.8911 | 0.8021 |
| 2 | 0.9072 | 0.8564 |
| 3 | 0.8709 | 0.8550 |

⚠️ 这一行只有 **3** epochs，不能与 8-epoch 的远端基线直接比；8-epoch 的干净版本已排进队列
（`b3_page_bigru_cap3000_8ep`）。

### 12.3 baselines.py 的两处改动

1. `--class_weight_mode {none,inverse}`，默认 `none`，所以 2026-09-30 之前的所有 B1-B3
   产物仍然逐位可复现。权重直接调用 `train.compute_class_weights`，与主模型同一个函数、
   同一条规则（`1/count` 后归一化到均值 1）。
   完整训练集上的 inverse 权重：
   `0.672,0.120,0.581,0.109,0.420,0.089,0.709,1.194,0.018,0.024,1.484,9.562,0.006,0.424,1.161,0.630,1.438,0.034,0.328`。
2. `--resume PATH`：从 `last.pt` 续跑到下一个 epoch，并且 `last.pt` 现在同时保存
   optimizer / scheduler 状态，续跑接的是同一条轨迹而不是重启 Adam 的动量。
   `baselines.py` 原本**没有任何续跑能力**，在会蓝屏的机器上这是每行 6-10 h 的裸奔。

### 12.4 已排队（本机 RTX 3060，串行，一次一行）

启动脚本 `outputs\_launch_baseline_weights.ps1`，控制台 `outputs\_baseline_weights_console.log`。

| # | 输出目录 | 配置 | 目的 | 预计墙钟 |
|---|---|---|---|---|
| 1 | `b3_page_bigru_cap3000_invw` | b3, 8 ep, inverse, cap3000 | 论文表 3 的 B3 行 | ≈10.5 h |
| 2 | `b1_roberta_mlp_invw` | b1, 8 ep, inverse | 补上权重公平性 | ≈7.4 h |
| 3 | `b2_roberta_layout_mlp_invw` | b2, 8 ep, inverse | 补上权重公平性 | ≈6.8 h |
| 4 | `b3_page_bigru_cap3000_8ep` | b3, 8 ep, **none**, cap3000 | 单独隔离「截断」这一个变量 | ≈10.5 h |

时间基准：本机校准实测 2.183 h/epoch（远端 4090 是 1.248 h/epoch，本机慢 1.75×）。


## 13. 基线权重对齐：第一个结果出来了（2026-10-01 11:30）

### 13.1 行 1 完成：B3 + inverse 权重 + cap3000，8 epochs

`outputs\path3_local_calib\b3_page_bigru_cap3000_invw`，17:13:14 → 10-01 05:56:58，
墙钟 45824 s = **12.73 h**（8 epoch，每 epoch 89-101 min），验集 64,788 块。

| epoch | acc | macro F1 | train loss | wall |
|---|---|---|---|---|
| 1 | 0.8832 | 0.8014 | 0.4913 | 98.9 min |
| 2 | 0.8946 | 0.8433 | 0.2531 | 98.9 min |
| 3 | 0.8637 | 0.8420 | 0.1906 | 95.7 min |
| 4 | 0.8750 | 0.8538 | 0.1576 | 100.5 min |
| 5 | 0.9103 | 0.8868 | 0.1199 | 101.2 min |
| 6 | **0.9137** | **0.9063** | 0.1065 | 89.7 min |
| 7 | 0.9079 | 0.8653 | 0.0983 | 89.5 min |
| 8 | 0.8950 | 0.8573 | 0.0939 | 89.0 min |

best acc 0.9137（ep6，同一 epoch 的 macro F1 = 0.9063）；
last（ep8，D6 主口径）0.8950 / 0.8573。

### 13.2 与「未加权 B3」的匹配分母对照 → 结论是**平局**

所有数字都在 64,788 块上：

| B3 配置 | last acc | last macro F1 |
|---|---|---|
| 未加权（远端 ep8 checkpoint，cap3000 复评） | 0.8871 | 0.8605 |
| **inverse 权重（本机行 1）** | 0.8950 | 0.8573 |
| Δ | **+0.79 pp** | **−0.32 pp** |

按 D7 噪声底（<1 pp 平局 / 1–3 pp 灰区 / >3 pp 真实），两个差都是**平局**。

**所以：把基线的损失权重对齐到主模型的 inverse，并没有改变 B3 的结论。**
B3 的 macro F1 不是"无权重 CE 压出来的假象"——这个公平性疑虑可以排除了，
而这恰好是论文里最该写出来的那种否定结果。

参考（未加权 cap48 训/评、59,043 块，即原表报告的 0.8950 / 0.8670）：
best acc 0.9261（ep7，同 epoch F1 0.8805），历史 best F1 0.9167（ep5）。

### 13.3 早期信号：B1 加 inverse 权重反而**明显更差**

行 2 正在跑。前三个 epoch 与远端未加权 B1（同 64,788 块）逐 epoch 对照：

| epoch | 加权 acc | 加权 macro F1 | 未加权 acc | 未加权 macro F1 |
|---|---|---|---|---|
| 1 | 0.5278 | 0.4681 | 0.7034 | 0.5338 |
| 2 | 0.5345 | 0.4422 | 0.7060 | 0.5452 |
| 3 | 0.5632 | 0.4856 | 0.7130 | 0.5619 |

acc 差 **−15 到 −18 pp**，macro F1 差 −7 到 −8 pp，远超噪声底。

机理上说得通：`1/count` + 归一化到均值 1 之后，稀有类 11 拿 9.562、最大类 12 只拿 0.006，
相当于让模型几乎放弃占 95% 的常见类。主模型扛得住是因为它本身强得多；
一个 RoBERTa+MLP 的基线扛不住。

⚠️ 只有 3/8 个 epoch，且加权版的 acc 仍在缓慢上升（0.5278→0.5632），
所以**还不能定论**；要按 D6 口径写进论文就得等 ep8。行 2 预计 10-01 20:10 结束。

### 13.4 修正后的 ETA

每 epoch 实测：B3 约 1.6 h，B1 约 1.78 h（比先前按 4090 × 1.75 推的估法慢，
`baselines.py` 的 DataLoader 是 `num_workers=0`，块级任务在小 batch 上更吃单核）。

| # | 行 | 状态 | 预计结束 |
|---|---|---|---|
| 1 | `b3_page_bigru_cap3000_invw` | ✅ 完成 10-01 05:56 | — |
| 2 | `b1_roberta_mlp_invw` | 跑到 ep3/8 | 10-01 约 20:10 |
| 3 | `b2_roberta_layout_mlp_invw` | 排队 | 10-02 约 09:30 |
| 4 | `b3_page_bigru_cap3000_8ep` | 排队 | 10-02 约 22:00 |


### 13.5 行 1 的支持度分层（last epoch，64,788 块）

行 1 的验集分母已与主模型完全一致，所以这张表可以直接和 `delivered` 比：

| 档 | 类数 | 块数 | 占验集 | `full`(delivered) | B3 行1（加权 cap3000） | B3 − 主模型 |
|---|---|---|---|---|---|---|
| `<100` | 2 | 86 | 0.1% | 0.3985 | 0.5475 | **+14.90 pp** |
| `100-999` | 10 | 2,797 | 4.3% | 0.8846 | 0.9113 | **+2.68 pp** |
| `1k-5k` | 4 | 8,506 | 13.1% | 0.9350 | 0.8394 | **−9.56 pp** |
| `5k+` | 3 | 53,399 | 82.4% | 0.9516 | 0.9076 | **−4.39 pp** |
| overall | 19 | 64,788 | 100% | 0.8546 | 0.8573 | +0.27 pp |

结论与 §5.2 一致、且更硬：B3 的 overall macro F1 优势集中在 `100-999` 档（10 个类、
只占 4.3% 的块），而在占验集 **95.5%** 块的 `1k-5k` + `5k+` 两档上落后 **9.56 / 4.39 pp**。
**主张 3 必须按"主模型在数据主体上更强"来写。**

### 13.6 行 1 关闭了论文草稿里的三个待办

`docs\path3_thesis_draft_2026-09-30.md`：

| 草稿位置 | 原状态 | 行 1 之后 |
|---|---|---|
| §6.2 基线类别权重一致性 | "该质疑对 B3 仍部分成立，对齐权重后的重跑列为进一步工作" | B3 已完成；B1/B2 由行 2/3 完成 |
| §6.3 评测分母一致性 | "存在训练/评测序列长度的轻微错配；匹配上限重训列为进一步工作" | 行 1 就是匹配上限重训（训练侧也 cap=3000）|
| §8 待办「B3 匹配上限重训」 | 可选，消除错配 | 已完成 |

**一个必须如实说明的残留混淆**：行 1 相对草稿里的 B3 同时动了三件事
（训练侧 cap 48→3000、加入 inverse 权重、全新的本机重跑），因此 **+0.79 pp 的 acc 提升
不能归因到任何单一因素**。本来能拆开这一项的正是被取消的行 4。
不过这不影响主张 3：三种 B3 口径下 overall macro F1 分别是 0.8573 / 0.8605 / 0.8670，
分层上 `1k-5k` 与 `5k+` 的落后幅度都在 4-10 pp 量级，结论对手上没有翻盘余地。


## 14. 2026-10-01 14:54 关机事故与续跑验证

**事实**：本机在 **14:54:25** 发生一次关机（System 日志 Kernel-Power 事件 109，
不是蓝屏 bugcheck；`LastBootUpTime` = 14:54:55）。当时行 2（B1）刚跑完 ep4
（`last.pt` 时间戳 13:05:13），ep5 的验证正好落在关机窗口里，被整段掐掉。
队列的 pwsh 包装进程一起死掉，所以既没有 `exit=` 行也没有 "queue done"——
`outputs\_baseline_weights_console.log` 停在 ep4 就是这样造成的。
GPU 从 13:05 空转到 16:08，约 3 h。

**恢复**：16:07:43 重新运行同一个 `outputs\_launch_baseline_weights.ps1`：

- 行 1 → `result.json present, skipping`
- 行 2 → `resuming from last.pt` → 日志出现
  `[resume] ... -> starting at epoch 5 (best=0.5634, carried 4 epoch(s))`
- 行 4 → 会命中 §13 之前的取消占位 `result.json`

**这次事故顺手验证了两件事**：

1. `baselines.py` 的 `--resume` 在真实事故里可用，只损失了 1 个 epoch（约 1.8 h）
   而不是整行 14 h。加这个功能是对的。
2. 队列脚本的"`result.json` 存在则跳过"规则同时充当了取消开关（行 4 就是这么摘掉的）
   和幂等重入口（行 1 重跑不会重训）。

**注意**：重跑会**覆盖** `outputs\_baseline_weights_console.log`，所以第二次启动
写到了 `outputs\_baseline_weights_console_20261001_1610.log`。
每行自己的 `train.log` 是追加写的、不受影响，也是权威记录。

**修正后的 ETA**：行 2 于 16:08 从 ep5 续起，约 7.1 h → 10-01 约 23:15；
行 3 约 14.2 h → 10-02 约 13:30。


## 15. 类别权重复跑：行 2 定稿、行 3 进行中（2026-10-02 10:30 追加）

### 15.1 行 2 `b1_roberta_mlp_invw` 已完成（8/8 epoch）

10-01 23:31:33 收尾，含 §14 那次关机后的续跑（ep5-ep8 由 `--resume` 接上）。
与远端未加权 B1 分母一致（64,788 块），逐轮对照：

| epoch | 加权 acc | 加权 macro F1 | 未加权 acc | 未加权 macro F1 | Δacc | Δmacro F1 |
|---|---|---|---|---|---|---|
| 1 | 0.5278 | 0.4681 | 0.7034 | 0.5338 | −17.56 pp | −6.57 pp |
| 2 | 0.5345 | 0.4422 | 0.7060 | 0.5452 | −17.15 pp | −10.30 pp |
| 3 | 0.5632 | 0.4856 | 0.7130 | 0.5619 | −14.98 pp | −7.63 pp |
| 4 | 0.5634 | 0.4899 | 0.7036 | 0.5478 | −14.02 pp | −5.79 pp |
| 5 | 0.5846 | 0.4734 | 0.6978 | 0.5565 | −11.32 pp | −8.31 pp |
| 6 | 0.5610 | 0.4383 | 0.6972 | 0.5558 | −13.62 pp | −11.75 pp |
| 7 | 0.6079 | 0.4855 | 0.6963 | 0.5525 | −8.84 pp | −6.26 pp |
| 8 | 0.5743 | 0.4805 | 0.7018 | 0.5518 | **−12.75 pp** | **−7.13 pp** |

按 D6 的 last-epoch 主口径：**acc −12.75 pp、macro F1 −7.13 pp**，两项都远超噪声底（中位 1.04 pp）。
按逐轮最优口径同样成立：末轮最优 acc 0.6079（ep7）对未加权 0.7130（ep3）＝ **−10.51 pp**；
全程最高 macro F1 0.4899（ep4）对未加权 0.5619（ep3）＝ **−7.20 pp**。
加权版每一个 epoch 都同时输掉两项指标，不存在"换个口径就翻盘"的余地。

**§13.3 的"还不能定论"至此关闭：对 B1，`inverse` 类别权重是有害的，不是早期波动。**
机理与 §13.3 的判断一致：`1/count` 归一化后稀有类 11 拿 9.562、最大类 12 只拿 0.006，
而 RoBERTa+MLP 这种容量/归纳偏置都弱的基线扛不住这种再平衡，于是连占 95% 的常见类一起崩。

### 15.2 行 3 `b2_roberta_layout_mlp_invw` 进行中（ep5/8）

10-01 23:32:06 起跑，10-02 08:51:40 完成 ep5，实测 ~112.6 min/epoch。
对照远端未加权 B2（同 64,788 块）：

| epoch | 加权 acc | 加权 macro F1 | 未加权 acc | 未加权 macro F1 | Δacc | Δmacro F1 |
|---|---|---|---|---|---|---|
| 1 | 0.4751 | 0.4433 | 0.6831 | 0.5248 | −20.80 pp | −8.15 pp |
| 2 | 0.5286 | 0.4549 | 0.6932 | 0.5374 | −16.46 pp | −8.25 pp |
| 3 | 0.5437 | 0.4759 | 0.6909 | 0.5790 | −14.72 pp | −10.31 pp |
| 4 | 0.5812 | 0.4844 | 0.6980 | 0.5523 | −11.68 pp | −6.79 pp |
| 5 | 0.5871 | 0.4587 | 0.6850 | 0.5769 | −9.79 pp | −11.82 pp |

与 B1 完全同一形态（加权版在长达 5 轮的区间里稳定落后 10-20 pp，且差距在缓慢收敛）。
剩余 ep6-ep8 预计 **10-02 约 14:30** 结束（原估 16:30 偏保守）。

### 15.3 这一组的总结论（三条基线，同一处理）

| 基线 | 加权 vs 未加权（last epoch） | Δ macro F1 | 判定 |
|---|---|---|---|
| B3 页内 BiGRU（cap3000、双方同分母） | +0.79 pp acc | −0.32 pp | **平局**（都在 <1 pp 内）|
| B1 RoBERTa+MLP | −12.75 pp acc | −7.13 pp | **加权有害** |
| B2 RoBERTa+布局+MLP | 待 ep8 | 待 ep8 | 形态同 B1 |

**对论文的意义**：`1/count` 类别加权是主模型与基线之间一处真实的实现差异，
补做之后它（a）**不足以解释 B3 的 macro F1 优势**——B3 的差距不是无权重 CE 压出来的；
（b）**对 B1/B2 反向生效**，加权反而把这两行拉得更远。
因此 §6.2 的公平性疑虑对三条基线全部关闭，且额外得到一条可写的否定结果：
**在不平衡语料上对弱基线施加 `1/count` 加权，会同时牺牲整体准确率与 macro F1。**