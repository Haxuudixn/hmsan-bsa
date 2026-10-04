# HMSAN-BSA 租用 GPU 训练交接单（2026-09-27）

> 目的：把「方案 A 干净重跑」整套实验搬到租用 GPU 上跑，结果回传后我做汇总与写作。
> 唯一事实来源仍是 `docs/path3_clean_rerun_plan_2026-09-25.md`；本文件只负责**怎么搬、怎么跑、跑完发我什么**。

---

## 0. 一页速览

1. **本地机器第 6 次崩溃**：2026-09-27 14:08:51，`0x0000013A`
   （KERNEL_MODE_HEAP_CORRUPTION），转储 `C:\Windows\Minidump\092726-7562-01.dmp`。
   阶段 1 死在 09-26 12:33:42，崩在**写 checkpoint 的中途**。→ 换租用 GPU，判断正确。
2. **要上传两样东西**：
   - 代码包 `hmsan_path3_cloud_bundle.tar.gz`（124 MiB，**已经打好**，在仓库上一级目录）；
   - 语料 `final_train\` 的 **38 个 zip / 36.86 GB**（**一个都不能少**，见 §3）。
3. **到机器上按 §5 自检 → §6 跑 4 个阶段 → §7 把产物发我。**
4. **现在还不能跑的**：8 行矩阵（`full` + 5 消融 + `delivered`）。
   它的 epoch 预算（决策 D2）由**阶段 2 pilot 的曲线**决定，必须先跑 pilot。

---

## 1. 为什么搬（证据，不必再看第二遍）

| 时间 | 事件 | 后果 |
|---|---|---|
| 09-08 / 09-15 / 09-17 / 09-18 / 09-24 | 5 次蓝屏（`0x101` / `0x4E` / WHEA Cache Hierarchy） | 长跑随时被打断 |
| 09-25 23:25 | 训练崩，exit `-1073741819`（`0xC0000005`），原生崩溃，**无 Python traceback** | 丢整轮 |
| 09-26 10:15 | 同上，step 78 | 只丢 3 步（F13 生效） |
| 09-26 12:33:42 | **机器整机崩溃**，崩在 `last.pt` 原子写的中途 | 见下 |
| 09-27 14:08:51 | `0x13A` 蓝屏 | 停机 |

**09-26 那次的实际损失（已核实）**：阶段 1 目录里留着 `last.pt.tmp`（1.30 GB，
目标 1.60 GB）= 写了一半被切断；`history.json` 只到 step 75，但控制台里
**step 150 的验证点已经算完并打印了**（`acc=0.6278 macro_f1=0.1417`），
因为当时顺序是「先写 1.6 GB checkpoint、再 flush history」，那一个点丢了。
→ 已修（§8 第 2 条）。

本机 `stage1_smoke_full` 现状：**未完成**。有 `last.pt`(step 125) 和 1 条
`mid_epoch`，**没有 `result.json`**。所以阶段 1 必须在租用机上重跑（很快）。

---

## 2. 实验总清单

| 阶段 | 内容 | 行数 | 本机实测/估算 | 租用机（按 4–5× 推算） | 出口条件 |
|---|---|---|---|---|---|
| **1 冒烟** | `full` × 1 epoch，seed 42，冷启动 | 1 | ≈1.9 h | ≈25–30 min | 6 条验收判据全绿（§6.1） |
| **2 pilot** | `full` × 8 epoch，seed 42（**定 D2**） | 1 | ≈15 h | ≈3–4 h | val acc 曲线出现平台期 |
| **3 矩阵** | 6 行（含参照行 `full`）+ `delivered` | 7 | 1.9 h/epoch × N | 同比例 | 每行 `status=completed` |
| **4 基线** | B1 / B2 / B3（`baselines.py`） | 3 | 待标定 | 待标定 | 三行都出 `result.json` |
| **5 评估** | 干净 test **一次性**打分 + 分层 | 1 次 | ≈0.2 h | ≈5 min | 主表口径冻结 |

**阶段 3 的 7 行**（决策 D3 已拍板：核心 6 行 + `delivered`）：

| # | run id | 说明 |
|---|---|---|
| 3.1 | `full` | 参照行（全门控），所有消融差值都对它读 |
| 3.2 | `A11_no_gates` | 全部门控关掉 |
| 3.3 | `A2_no_g2` | 去掉 G2 value gate |
| 3.4 | `A6_no_page_memory` | 去掉 Infini 页记忆 |
| 3.5 | `A7_no_boundary_gate` | 去掉边界感知重置 |
| 3.6 | `A10_dense` | 稠密注意力上界（**同时充当基线 B4，不用另跑**） |
| 3.7 | `delivered` | 论文主模型，**长 schedule**（≥8 epoch，旧跑就是 8） |

**阶段 4 的 3 个基线**（决策 D4 已拍板）：`b1_roberta_mlp` / `b2_roberta_layout_mlp` /
`b3_page_bigru`。硬要求：**必须端到端微调文本编码器**（脚本已按此实现），
且必须用与主模型**完全相同**的切分——`baselines.py` 会自己断言
`train=306 val=66 test=62`，对不上直接拒绝训练。

**暂缓**（别跑）：`A1_no_g1` / `A3_fixed_k` / `A4_no_gated_fusion` / `A5_standard_ffn` /
`A8_no_interpage_gate` / `A9_local_window` / `frozen_both`，
以及 `both_unfrozen` / `image_unfrozen`（12 GB 显存墙，不要碰）。

---

## 3. 数据来源：必须上传什么

**源目录**：`C:\Users\Administrator\Desktop\训练数据\final_train\`
**内容**：**38 个** `training_data_*.zip`，合计 **36.86 GB**（精确 **39,576,859,849 字节**）

### 3.1 为什么一个 zip 都不能少

`load_documents_from_zips()` 把文档键定为 **`(zip 路径, pdf_name)`**，
所以「同一个 PDF 出现在两个导出里」会被当成**两个独立文档**。
这意味着：**删掉任何一个 zip，语料规模、切分结果、指纹全部改变**，
和本地跑出来的数字不可比。必须整目录上传。

### 3.2 校验锚点（上机后必须核对）

| 项 | 期望值 |
|---|---|
| zip 个数 | **38** |
| `zips_sha256_16` | **`aa3cb23bcddcc621`** |
| 语料段数 / 块数 / 源 PDF 数 | 434 / 456,602 / 132 |
| train | 306 段（95 个源 PDF） |
| val | 66 段（20 个源 PDF） |
| test | 62 段（17 个源 PDF） |
| 跨切分 PDF | **0** |

> `zips_sha256_16` 的算法是「文件名 + 文件大小」排序后 sha256 取前 16 位，
> 能抓「少传/改名」，**抓不到「传坏了」**。所以 37 GB 的搬运还要做下面这一步。

### 3.3 内容级完整性校验（强烈建议）

```powershell
# 本地：生成清单（读 37 GB，几分钟）
C:\hmsan\.venv\Scripts\python.exe scripts\paper\data_manifest.py `
    --data_dir "C:\Users\Administrator\Desktop\训练数据\final_train" `
    --out data_manifest_local.json
```

```bash
# 租用机：上传完成后生成清单，再比对
python scripts/paper/data_manifest.py --data_dir "$DATA" --out data_manifest_cloud.json
python scripts/paper/data_manifest.py --compare data_manifest_local.json data_manifest_cloud.json
```

必须打印 **`IDENTICAL`**；打印 `MISMATCH` 就不要开训（它会指名哪个文件坏了）。

### 3.4 怎么把这 37 GB 搬上去（这是整个方案最大的摩擦）

- **优先走平台侧导入**：主流国内平台（AutoDL / 恒源云 / 矩池云 / 蓝耘 等）都支持
  「网盘导入」（阿里云盘 / 百度网盘 / 夸克）或「文件存储 / 数据盘」跨实例复用。
  先把 38 个 zip 塞进网盘，再从平台侧导入，通常比家宽直传快得多。
- **直传就用 rsync（支持断点续传，37 GB 断一次别重头来）**：

```bash
rsync -avP --partial --append-verify \
  "/cygdrive/c/Users/Administrator/Desktop/训练数据/final_train/" \
  root@<host>:/root/autodl-tmp/final_train/
```

- **只传一次**：把语料放在持久化存储/数据盘上，之后改代码不要重传语料。
- zip 已经是压缩包，**不要再压**，按原大小估时间：

| 上行带宽 | 36.86 GB 耗时 |
|---|---|
| 10 Mbps | ≈8.4 h |
| 20 Mbps | ≈4.2 h |
| 50 Mbps | ≈1.7 h |
| 100 Mbps | ≈0.8 h |

---

## 4. 代码与资产：上传什么

### 4.1 已经打好的包

```
..\hmsan_path3_cloud_bundle.tar.gz        124.1 MiB  (148 entries)
sha256 7CB48BD618162C50D1B51E08F3A422305E23C3AE6C740EE30241FB96F30F6341
```

（`..` = 仓库的上一级，即 `...\https\`；同目录还有 `.manifest.json` 记录大小与哈希。
需要重打就运行 `pwsh -NoProfile -File scripts\paper\package_for_cloud.ps1`。）

**包里有什么**：`train.py` / `predict.py` / `experiments.py` / `pyproject.toml` /
`requirements-cloud.txt` / `src/` / `configs/` / `scripts/` / `experiments/` /
`tests/` / `docs/` / **ViT 缓存**。

**包是「工作区快照」，不是 `git clone`** —— 这点很重要：`train.py` 等 10 多个文件
带着未提交的 F1–F13 改动，而 `.gitignore` 排除了 `cache/` 和 `*.zip`。
用 git 拉代码会拿到**错的代码**且没有缓存。

**包里故意没有**：`outputs/`（见 §9 第 1 条）和语料。

### 4.2 ViT 缓存一定要带上

```
cache/google__vit-base-patch16-224-bf16/     238 MB
  features.f32.npy   keys.txt   index.json   failed.txt
```

- 只有 238 MB，但**每个 epoch 省约 48 分钟**（旧矩阵 13 行 × 3 epoch 共省约 27 h）。
- 在云端重建要花 **≈38 分钟 GPU**（`scripts/paper/build_vit_cache.py`，实测 35.7 img/s）。
  传输成本 ≈ 重建成本的 1/10，所以**传**。
- 必须连 `failed.txt`（3,489 个无法解码的 key）一起传，否则行为与冻结口径不一致。
- 上机后核对：`python scripts/paper/verify_vit_cache.py --cache_dir cache/<...>`

---

## 5. 环境与上机自检

### 5.1 平台要求

| 项 | 要求 | 说明 |
|---|---|---|
| GPU 显存 | **≥12 GB，建议 24 GB** | 训练峰值实测约 5.4–7.0 GB（受 3000 块 / 150 页上限约束）；24 GB 有富余，也不会更贵太多 |
| GPU 数量 | **单卡** | 代码没有多卡支持 |
| CPU | ≥8 核 | 想用 `--num_workers` 就要有核 |
| 内存 | ≥16 GB | 本机 8 GB 曾经把系统拖卡 |
| 磁盘 | ≥150 GB | 语料 37 + 输出（7 行 × 约 1.5 GB checkpoint） |
| 存储 | **要持久化/数据盘** | 否则实例一释放就要重传 37 GB |
| 价格参考 | 见 `docs/gpu_rental_cost_2026-09-21.md` | RTX 4090 ≈ $0.34/h，整套路径 3 ≈ $15–25 |

> **真正的成本风险不是算力，是「忘关机」**：4090 跑满一天 ≈ $8。
> 以及「`num_workers=0` 让 GPU 空转」（§8 第 1 条）。

### 5.2 环境

本地已验证的组合：**Python 3.11.9 / torch 2.13.0+cu126 / transformers 5.15.1**，
完整快照见包里的 `requirements-cloud.txt`。

```bash
# 1) 解包
mkdir -p ~/hmsan-bsa && tar -xzf hmsan_path3_cloud_bundle.tar.gz -C ~/hmsan-bsa
cd ~/hmsan-bsa
export REPO=$PWD
export DATA=/root/autodl-tmp/final_train          # 改成你的实际路径
export CACHE=$REPO/cache/google__vit-base-patch16-224-bf16

# 2) 环境：如果镜像里已经装好 torch，跳过第 2 步
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
pip install -r requirements-cloud.txt \
  --extra-index-url https://download.pytorch.org/whl/cu126

# 3) 编码器权重（首次会自动从 HuggingFace 下载，约 0.71 GB）
#    hfl/chinese-roberta-wwm-ext + google/vit-base-patch16-224
export HF_HUB_DISABLE_TELEMETRY=1
```

### 5.3 上机自检（**5 步全过才开训**）

```bash
cd $REPO

# A. GPU 可见
python -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0))"

# B. 语料与切分（最关键的一步）—— 期望 38 个 zip、434 段、PASS、指纹 aa3cb23bcddcc621
python scripts/verify_split_clean.py --data_dir "$DATA" \
    --manifest_out outputs/path3_diag/split_manifest.json

# C. 设备防线（F9）
python scripts/verify_device_guards.py            # 期望 ALL PASS

# D. 单元测试
pytest tests -q                                   # 期望 18 passed

# E. 断点续跑机制（F13）
python scripts/verify_ckpt_resume.py              # 期望 All F13 checks passed
```

**B 的实际期望输出**（本地刚复核过，逐字对照）：

```
Loading 38 ZIP(s)...
Corpus: 434 segments, 456602 blocks, 132 source PDFs
split     segs  pdfs    blocks   labeled
train      306    95    311871    311870
val         66    20     64788     64788
test        62    17     79943     79851
PASS: no source PDF straddles two splits.
```

对不上就**停**，把 `zip_count` / `zips_sha256_16` / `straddling_pdfs` 三个数字发我。

---

## 6. 怎么跑

### 6.1 阶段 1：冒烟（**闸门**，必须先绿）

```bash
cd $REPO
# 顺便标定 num_workers：先 0，再 4，比较 s/it
python scripts/paper/run_matrix.py --only stage1_smoke_full --num_workers 0
python scripts/paper/run_matrix.py --only stage1_smoke_full --num_workers 4 --force
```

看 `outputs/path3_clean/stage1_smoke_full/train.log` 里的 `17.66s/it` 那类读数对比。

**6 条验收判据（全满足才准进阶段 2）**：

1. 日志含 `COLD START`，启动 argv 里**没有** `--init_from`；
2. `Split: train=306, val=66, test=62`；
3. `history.json` 至少 1 条 `phase=mid_epoch`，且其 `val` 里同时有
   `boundary` 与 `section` 子字典，`boundary` 里要有 `section_start_f1`；
4. `result.json` 的 `status=completed`；
5. 全程无 `Traceback`、无 `CUDA out of memory`、无 `mat2 is on cpu`；
6. 确实只跑了 1 个 epoch。

> 参考值：本地在 step 75 得到 `acc=0.5608`、step 150 得到 `acc=0.6278`
> （boundary_f1 0.2500 / section_f1 0.0352 / start_f1 0.0000）。
> 你的数字应该落在这个量级附近。

### 6.2 阶段 2：pilot（**决定 D2 epoch 预算**）

```bash
python scripts/paper/run_matrix.py --only stage2_pilot_full_seed42 --num_workers 4
```

跑 **8 个 epoch**（比原计划上限 6 多，因为原上限只是「本机赶不上 08:00」的产物；
租用机上多跑两轮很便宜，而曲线越完整 D2 定得越准）。

**跑完把 `history.json` 发我**，我要读：val acc 在第几 step 到顶、折算第几个 epoch、
第 8 轮是在涨还是已经平了。**D2 定了才开阶段 3。**

### 6.3 阶段 3：矩阵（等 D2）

D2 出来之后（假设预算 = N 轮）：

```bash
# 核心 6 行（full + 5 消融）
python scripts/paper/run_matrix.py --core --epochs N --num_workers 4

# 第 7 行：论文主模型（长 schedule，≥8 轮）
python scripts/paper/run_matrix.py --only delivered --epochs 8 --num_workers 4
```

先 `--dry_run` 看一眼要跑什么（不实际训练）：

```bash
python scripts/paper/run_matrix.py --core --epochs N --dry_run
```

### 6.4 阶段 4：基线

```bash
python scripts/paper/run_matrix.py --group baselines --num_workers 4
```

### 6.5 阶段 5：干净 test 一次性打分

矩阵全部结束后再做，**test 只碰这一次**（跑法见计划文档 §6 阶段 5；
需要时我再给具体命令）。

---

## 7. 跑完发我什么

**要发的**（每个 run 一个目录）：

| 文件 | 用途 |
|---|---|
| `history.json` | 逐 step 的 mid-epoch 曲线 + 每轮 val（**最重要**） |
| `result.json` | 状态、墙钟、best/last 指标 |
| `run_meta.json` | 参数量、ablation signature、epochs_completed |
| `train.log` | 现场证据（含 `[ckpt]` / `[mid ...]` 行） |

**汇总级**：

| 文件 | 位置 |
|---|---|
| `summary.json` / `summary.csv` | `outputs/path3_clean/` |
| `split_manifest.json` | `outputs/path3_diag/` |
| `data_manifest_local.json` / `data_manifest_cloud.json` | 你指定的路径 |

**不要发的**：`*.pt`（每个 1.5 GB）。除非我要查某个具体 checkpoint，我会单独说。

**打包命令**（只收文本产物，很小）：

```bash
cd $REPO
tar -czf path3_results.tar.gz \
    $(find outputs/path3_clean -maxdepth 2 \( -name 'history.json' \
        -o -name 'result.json' -o -name 'run_meta.json' -o -name 'train.log' \) \
        -o -maxdepth 1 -name 'summary.*') \
    outputs/path3_diag/split_manifest.json \
    data_manifest_*.json
ls -lh path3_results.tar.gz
```

---

## 8. 为云端做的改动（含必须验证的 1 条）

| # | 改动 | 状态 |
|---|---|---|
| 1 | `train.py` 新增 `--num_workers`（**默认 0，行为不变**） | 已验证：数据加载器校验 + 18 项测试通过 |
| 2 | **先 `flush_history()`、再写 checkpoint**（修复 step 150 验证点丢失） | 已验证：`verify_ckpt_resume.py` 11 项全过 |
| 3 | `run_meta.epochs_completed` 只数 epoch 记录（原来把 `mid_epoch` 也数成 epoch） | 已验证 |
| 4 | `final_val_*` 只读**最后一个 epoch 记录**（原来读 `history[-1]`，可能是 mid-epoch） | 已验证 |
| 5 | 新增 `scripts/paper/run_matrix.py`（Linux 版矩阵执行器） | 已验证：argv 与 `run_train_final.ps1 -PrintOnly` **逐项等价** |
| 6 | 新增 `experiments/path3_clean_manifest.json`（干净协议清单） | 已校验 |
| 7 | 新增 `scripts/paper/data_manifest.py`（37 GB 内容校验） | 已验证：能抓出内容损坏并指名文件 |
| 8 | 新增 `scripts/paper/package_for_cloud.ps1` + `requirements-cloud.txt` | 已验证：产出 124 MiB 包 |

| 9 | `scripts/paper/baselines.py` 修两个真 bug（见下） | 已验证：B1/B2/B3 各跑完 1 epoch，写全 artifact |
| 10 | `--num_workers` 端到端校验（2/3 worker 与 0 worker 逐批完全一致） | 已验证 |

**第 9 条的细节（这两个 bug 不修，阶段 4 的数字会是错的）**：

1. **`block_text()` 把图片块喂成了文件路径**。文本块的内容在 `text`，
   而图片/混合块在 `text` 里存的是 ZIP 成员路径、真正文字在 `ocr_text`
   （见 `Block.block_type_id`）。原来的写法会喂 `blocks/block_0222.png` 这种字符串
   ——既不是文字，又**带着块序号，是可记忆的**。已改成与主模型一致：
   文本块取 `text`，图片/混合块取 `ocr_text`。修完后同一份 3 文档冒烟里
   B1 的 val acc 从 0.2722 变成 0.3890，说明影响不小。
2. **`evaluate_pages()` 少解一层**。B3 的 `collate_fn` 返回「页面列表」，
   所以一个 loader step 是**一批** `(texts, labels)`，不是一对；
   原来的写法把整个 tuple 直接送进 tokenizer，报
   `TypeError: TextEncodeInput must be Union[...]`，训练能跑、**一进验证就崩**。

> 冒烟结果（仅 3 个文档，**数字无意义**，只证明路径通）：
> B1 `acc=0.3890 / macro_f1=0.1616`、B2 `0.3655 / 0.1503`、
> B3 `0.5617 / 0.1080`，每个 ≈1.2–1.6 min，显存峰值 ≈1.9 GB。
> 真实 epoch 成本等你在机器上用 1 epoch 标定。

**第 1 条要你在机器上标定**：`--num_workers 0` 让「读 zip + 解 PNG」与 GPU 计算
**串行**。本机上这部分只占一步的 ≈9%（实测：解码 14.3 ms/图、ViT 前向 21.0 ms/图），
但在 4090/A100 上 GPU 快 4–5 倍、CPU 不变，**占比会放大**，不设 worker 就是在为
空转的 GPU 付钱。用 §6.1 的 0 vs 4 对比决定。

**验证记录（本地，2026-09-27）**：

| 检查 | 结果 |
|---|---|
| `pytest tests -q` | **18 passed** |
| `scripts/verify_device_guards.py` | **ALL PASS** |
| `scripts/verify_ckpt_resume.py` | **11/11 passed** |
| `scripts/verify_split_clean.py` | 复现 38 zip / 434 段 / 456,602 块，指纹 `aa3cb23bcddcc621` 与冻结值一致，0 跨切分 |
| `scripts/verify_data_loader.py`（单 zip） | 通过 |
| `run_matrix.py --dry_run` vs `run_train_final.ps1 -PrintOnly` | argv **逐项等价**（仅多出新增的 `--num_workers 0` 与输出目录差异） |
| `--num_workers` 2/3 vs 0 | 逐批指纹**完全一致**，无 worker 报错 |
| `baselines.py` B1/B2/B3 | 各 1 epoch 跑完，`status=completed`，artifact 齐全 |

> 本地 `outputs/_bsmoke*/`、`outputs/_baseline_smoke/` 里留了几份冒烟产物（含约
> 410 MB 的 `best_model.pt`），删不掉也不影响——它们已经被打包脚本排除在外，
> 不会被传到租用机。想清理可以手动删。

---

## 9. 风险与注意

1. **不要把 `outputs/` 传上去**。本地 `outputs/path3_clean/stage1_smoke_full/`
   有 `last.pt`(step 125) 和 `.tmp`；执行器看到 `last.pt` 会自动加 `--resume`，
   于是阶段 1 会从**本机那个半截 checkpoint 续跑**，不再是干净的一整轮。
   要干净就从空目录开始。
2. **绝不要传 `--init_from`**。决策 D1 = 方案 A 冷启动，这是整套重跑的立身之本。
   `run_matrix.py` 在设计上不会加这个参数。
3. **同一时刻只跑一个训练**。执行器天然串行，别手工并行。
4. **`class_weight_mode` 从 `none` 改成了 `inverse`**。旧矩阵用 `none`，
   而 19 类里第 11 类只有 97 块 / 310k 块——不加权就会砸掉**正在报告的那个指标**
   （macro F1）。这是有意的口径变更，论文里要写明。
5. **不要在 Windows 侧用 `run_comparison.ps1` 汇总干净重跑**：它会把
   `mid_epoch` 记录当成 epoch 来算 best/final（mid-epoch 的 acc 有可能高于所有轮末值），
   汇总会被污染。用 `run_matrix.py`，它只读 epoch 边界。
6. **实例销毁前先确认产物已下载**：`result.json` + `history.json` 极小，
   但 checkpoint 有 1.5 GB，没传下来就只能重跑。
7. **崩了怎么办**：`--ckpt_every_steps 25` 每 25 步落一次盘，崩了重跑同一行会自动
   `--resume`（执行器最多重试 2 次）。看到 `last.pt.tmp` 说明上次崩在写盘中途，
   `last.pt` 本身仍是上一个完好的点，无害。
8. **显存/温度**：显存 >11 GB 或温度 >80 ℃ 就停手告诉我。
9. **pilot 之前不要开矩阵**。前面所有返工都来自「没冒烟就铺开」。

---

## 10. 我需要你回传后才能做的

- **D2**（epoch 预算）：读阶段 2 的曲线 → 补进计划文档 §8；
- 阶段 3 主表 + 支持度分层（`overall / <100 / 100–999 / 1k–5k / 5k+`）；
- 阶段 4 基线对照表；
- 把所有结果写进 `docs/path3_overnight_2026-09-25.md`（时间线、验收逐条、
  曲线、分层表、D2 建议）。
