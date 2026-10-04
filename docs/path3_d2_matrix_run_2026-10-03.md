# D2 重跑执行方案（2026-10-03）：7 行 × 10 epoch，本机 + 远端分工

> ⛔ **已于 2026-10-04 07:19 停机，D2 回退为 8 epoch**（10 轮的依据被本方案自己的两条实测证伪）。
> 停机记录、证据与结论见 §10；最终决策见 `docs/path3_clean_rerun_plan_2026-09-25.md`
> §8「D2 —— ✅ 最终（2026-10-04）」。本文件其余章节保留为执行记录。

上游：`docs/path3_clean_rerun_plan_2026-09-25.md` §8（D2/D3/D6 决策）、
`docs/path3_overnight_2026-10-03.md`（16 轮延长跑的实测依据）。

---

## 1. 本次拍板的决策

| 项 | 旧值 | 新值 | 依据 |
|---|---|---|---|
| D2 epoch 预算 | 8 | ~~10~~ → **8**（10-04 回退） | ~~依据：`delivered` 8→16 轮实测，macro F1 包络 ep10 到顶（0.89177）~~ **被本方案自己的两条 10 轮实测证伪**：`delivered` / `full` 的包络峰值都在 ep8，ep9–10 不涨反跌。见 §10.2 |
| LR 调度 `patience` | 3（硬编码） | **1** | 16 轮里 `patience=3` 最多只触发 1 次，LR 几乎全程恒定；ep10/ep16 末的验证塌陷是高 LR 失稳 |
| `min_lr` 下限 | 无 | **1e-5** | `patience=1` 触发 2 次（ep8、ep10）、末期 LR 为 25%；加下限把衰减封顶（head 最多 3 次、encoder 1 次）。〔2026-10-04 更正：初稿写的「降 7 次 / 剩 0.78%」是 `>=` 判据下的错算，见 §10.3〕 |
| 输出根 | `outputs/path3_clean` | **`outputs/path3_d2_10ep`** | 旧根里仍存着 8 轮的 `last.pt`，`run_matrix.py` 会自动 `--resume` 它 —— 那等于拿 10 轮协议去续 8 轮的 checkpoint。新根同时保留 8 vs 10 轮的对照 |
| 多 seed | — | **等 D2 结果再定** | 用户 2026-10-03 决定：先看 10 轮结果是否落在灰区，再决定补不补 seed 43/44 |

7 行 = D3 的核心 6（`--core`）+ `delivered`：
`full`、`A11_no_gates`、`A2_no_g2`、`A6_no_page_memory`、`A7_no_boundary_gate`、`A10_dense`、`delivered`。
注意 `delivered` 与 `full` 是**同配置同 seed 的重复行**（这就是噪声底的来源），所以这 7 行 = 6 个条件 + 1 个重复。

---

## 2. 代码改动（2026-10-03）

1. **`train.py`**
   - 新增 `--lr_patience`（默认 **3**，保持旧行为）与 `--lr_min_lr`（默认 **0**，无下限）。
     原来 `patience=3` 是硬编码在第 1040 行的，现在协议可显式声明。
   - 启动时会打印 `LR scheduler: plateau factor=0.5 patience=... epoch-end checks, min_lr=...`，
     两个值也写进 `run_meta.json` 的 `optimizer` 块，便于事后核对。
   - 顺带修掉一个既有 bug：`--num_workers` 帮助文本里的裸 `%` 会让 `train.py --help`
     直接抛 `TypeError: %o format ...`（argparse 的 `%` 格式化），已转义为 `%%`。
   - 备份：`train.py.bak_20261003`。
2. **`scripts/paper/run_matrix.py`**
   - 新增 `--lr_patience` / `--lr_min_lr`，为 None 时不传（保持历史协议）；
   - 子进程输出改为**实时 tee**（原来是 `subprocess.run(stdout=PIPE)`，要等行跑完才写 `train.log`，
     一行 10–22 小时，等于整段看不到进度、崩溃还会丢掉整份日志）。`history.json` 仍是权威记录。
   - 备份：`scripts/paper/run_matrix.py.bak_20261003`。
3. **`scripts/paper/run_d2_matrix.sh`**（新）：远端 7 行队列，SLOTS=2 + 显存/磁盘守卫 + 断点续跑。
4. **`outputs/_launch_d2_local.ps1`**（新，带 BOM）：本机单行启动器。

---

## 3. 本机 / 远端分工（按吞吐，不是按"难度"）

先说一个事实：**7 行的计算量基本相同**（同一份语料、同一套 caps、同一模型规模），
所谓"难跑的行"并不存在。真正的差别是两台机器的吞吐：

| 机器 | 卡 | 并行 | 单行单 epoch（实测） | 聚合吞吐 |
|---|---|---|---|---|
| 租用机 | 4090 24 GB | 2 行（每行峰值 ~8.9 GB） | **~1.2 h** | **1.67 行·epoch/h** |
| 本机 | 3060 12 GB | 1 行（12 GB 放不下 2 行） | **~2.25 h**（ep13–16：2h15–2h37） | **0.44 行·epoch/h** |

7 行 × 10 epoch = **70 行·epoch**。几种分法的完工时间：

| 分配（本机 + 远端） | 本机耗时 | 远端耗时 | 完工 |
|---|---|---|---|
| 0 + 7 | — | 42 h | 42 h |
| **1 + 6** | **22.5 h** | **36 h** | **36 h** ← 选它 |
| 2 + 5 | 45 h | 30 h | 45 h |
| 7 + 0 | 158 h | — | 158 h |

⇒ **本机 1 行、远端 6 行**。本机那行跑完（约 22.5 h）后可以空出来，
正好用作后续补 seed 或做 test 评估，不会浪费。

**本机跑哪一行**：`A7_no_boundary_gate`。理由：(a) D8 已把"边界"从论文口径里去掉，
它是 7 行里论文优先级最低的；(b) 本机 10-03 00:41 刚强杀过一次 9 小时的任务、且有多次蓝屏史，
放最不重要的一行，损失最小。**想换行随时可换**（改 `--only` 一个参数）。

**远端行序**（SLOTS=2，按 paper 重要性排序，早停也能拿到头条数字）：

| 时段（相对远端启动 T0） | 槽位 | 完成 |
|---|---|---|
| T0 → T0+12h | `delivered` + `full`（主模型 + 噪声底重复对） | T0+12h |
| T0+12h → T0+24h | `A6_no_page_memory` + `A2_no_g2`（争议最大的两条消融） | T0+24h |
| T0+24h → T0+36h | `A10_dense` + `A11_no_gates` | T0+36h |

---

## 4. 远端开机后的准备清单

远端实例现在是**关机**状态（`connect.bjb1.seetacloud.com:52544` 连接被拒），需要在 AutoDL 控制台手动开机。

1. 开机后先跑数据指纹校验：远端 `outputs/path3_diag/split_manifest.json` 的
   `data_fingerprint` 必须等于清单里的 `aa3cb23bcddcc621`，且 `Split: train=306, val=66, test=62`。
   不匹配就**不要**开跑（切分不是冻结的那一份）。
2. 同步三个改动过的文件到 `/root/autodl-tmp/hmsan-bsa`：
   `train.py`、`scripts/paper/run_matrix.py`、`scripts/paper/run_d2_matrix.sh`。
   （远端若有同名备份，先留档。）
3. 确认 ViT 特征缓存存在：`cache/google__vit-base-patch16-224-bf16`（81317 张、768-d）。
4. 确认磁盘：数据盘 105 GB，语料 37 GB，7 行 × ~6.4 GB（last.pt + best_model.pt + 3 个每 5 轮的 epoch 检查点）
   ≈ 45 GB → 够，但脚本仍有 `MIN_FREE_GB=14` 守卫，跑完一行会删掉 `checkpoint_epoch5.pt`。
5. `screen -dmS d2matrix bash scripts/paper/run_d2_matrix.sh`，然后 `tail -f /root/autodl-tmp/d2_matrix.log`。
6. 先确认第一行打印出 `LR scheduler: plateau factor=0.5 patience=1 epoch-end checks, min_lr=1e-05`
   —— 这是"新协议真的传下去了"的唯一硬证据。

本机（已经在跑，18:45:52 启动）：
`outputs/_launch_d2_local.ps1` → `run_matrix.py --only A7_no_boundary_gate --epochs 10 --lr_patience 1 --lr_min_lr 1e-5 --num_workers 4`，
日志 `outputs/_d2_local_console.log`，行日志 `outputs/path3_d2_10ep/A7_no_boundary_gate/`。

---

## 5. 监控口径

- **权威记录是每行的 `history.json`**（train.py 每 75 步做一次中间验证就写一次），
  不是控制台日志。注意本机这条是用**改动前**的 `run_matrix.py` 启动的，
  它的 `train.log` 要等行结束才会落盘（子进程输出被全缓冲）；远端用的是改动后的版本，日志实时。
- 每行结束写 `result.json`（`status=completed` / `crashed`）。
- 卡死判据：`history.json` 或 `last.pt` 的 mtime 超过 **45 分钟**没动，或显存 > 11 GB（本机）/ > 22 GB（远端），
  或温度 > 80 ℃。
- 崩溃处置：`run_matrix.py --max_attempts 2` 会自动用该行自己的 `last.pt` 重试一次。

---

## 6. 完工后的收尾

1. 拉回 6 行结果 + 本机 1 行，跑 `scripts/paper/summarize_comparison.py` 与
   `scripts/paper/support_stratified.py`（支持度分层是强制的，D6）。
2. 与现存 8 轮结果（`outputs/path3_clean/*`、`remote_pull_2026-09-29/`）做 8 vs 10 轮对照，
   确认新协议（patience=1）没有把早期轮次压坏。
3. 按 D7 的预登记判据判断是否需要补 seed 43/44；此时本机已空闲，可用于补 seed。
4. 更新 `docs/path3_thesis_draft_2026-09-30.md`（由 `scripts/paper/make_thesis_draft.py` 生成，
   改生成器而不是手改产物）：换入 10 轮主表数字，并写明「训练协议变更：`patience` 3→1 + `min_lr` 1e-5」。
---

## 7. 启动记录（2026-10-03 19:07）

### 7.1 取消灰区种子行（用户决定）

用户决定：**先撤掉远端的种子行**，等 D2 结果出来再决定要不要补种子。

- 涉及目录：`outputs/path3_clean/A2_no_g2_s43`（8 轮跑完、**无** `result.json`）、
  `A6_no_page_memory_s43`（7 轮）、`A10_dense_s43`（空目录，从未启动）。
- 三条整体移到 `outputs/_cancelled_seeds_2026-10-03/`；其中 `history.json` / `run_meta.json` / `train.log`
  另存到 `_meta/<row>/` 留证，**1.6 GB 的 `.pt` 全部删除**（腾出 9 GB）。
- 4 个 Dead screen（`greenseeds` / `cleaneval` / `wp` / `pacelog`）已 `screen -wipe` 清理；
  远端无 crontab、无 `/etc/rc.local`，**开机不会自动拉起队列**（已核实）。
- 影响：两条种子行是 8 轮协议，与新协议（10 轮 + `patience=1`）不可比，
  所以"取消"和"重做"是一回事，不算浪费；真要补种子时按新协议整批跑。

### 7.2 磁盘

清完 **36 GB** 空闲（清理前 27 GB，处于 `MIN_FREE_GB=14` 守卫生效边缘）。
6 行 × 3.0 GiB ≈ 18 GiB，守卫全程不会触发。

### 7.3 本机 / 远端分工

**4090 只跑 6 行，A7 由本机 3060 跑。** 队列脚本新增 `D2_ROWS` 覆写：

```bash
D2_ROWS="delivered full A6_no_page_memory A2_no_g2 A10_dense A11_no_gates" \
  bash scripts/paper/run_d2_matrix.sh
```

理由：吞吐是 本机 0.44 行·轮/h vs 4090 1.67 行·轮/h，6+1 分成 36 h，
全远端 42 h，本机 2 行 + 远端 5 行 45 h。A7 是论文里不报告的那一行（D8：边界概念已移除），
放在会蓝屏的机器上损失最小。

### 7.4 启动

同步（md5 与本地一致，远端原件备份为 `*.bak_20261003_remote`，已 `bash -n` 通过）：

| 文件 | md5 |
| --- | --- |
| `train.py` | `20a74347e071e677191580e93cae3204` |
| `scripts/paper/run_matrix.py` | `b871012fd742c135b0c0e0e75e8d910d` |
| `scripts/paper/run_d2_matrix.sh` | `aa06bd9c74169797151ca179362d8e55` |

```bash
cd /root/autodl-tmp/hmsan-bsa
screen -dmS d2matrix bash -lc 'D2_ROWS="delivered full A6_no_page_memory A2_no_g2 A10_dense A11_no_gates" bash scripts/paper/run_d2_matrix.sh'
```

- 19:07:57 起，`screen` 名 `d2matrix`；第一波 `delivered` + `full` 于 19:08:00 开跑。
- 起跑时显存 8.5 GB / 24 GB、GPU 88% 满负荷。
- **实测每 epoch ≈50–60 min**（比计划的 1.2 h 快），6 行预计 **~30 h** 收口（约 10-05 凌晨）。

### 7.5 验收硬证据

`outputs/path3_d2_10ep/<row>/train.log`：

```
Split: train=306, val=66, test=62
LR scheduler: plateau factor=0.5 patience=1 epoch-end checks, min_lr=1e-05
```

`run_meta.json`：`"epochs": 10`, `"lr_patience": 1`, `"lr_min_lr": 1e-05`；
日志含 `COLD START`；数据指纹 `aa3cb23bcddcc621`（= 冻结清单），切分 306/66/62 一致。

**日志口径提醒**：`/root/autodl-tmp/d2_matrix_<row>.log` 因 python stdout 重定向到文件而被块缓冲、会滞后；
**行内 `outputs/path3_d2_10ep/<row>/train.log` 是实时的**（新 `run_matrix.py` 的 tee），权威记录仍是 `history.json`。

### 7.6 本机那一路

18:45:52 启动的 `A7_no_boundary_gate`（`outputs/_d2_local_console.log`）同期在跑，与远端互不重叠。
---

## 8. 仪表盘改指向 D2（2026-10-03 19:18）

`scripts\paper\status.ps1` 原来盯 `outputs\path3_matrix`（旧矩阵，已被干净重跑取代），
刷出来写「预计还需 86.6 d」，是废数据。已重写：

- 现在盯 **`outputs\path3_d2_10ep`**（本机 1 行）+ **租用 4090 上的同一矩阵**（6 行）。
- 新增 `scripts\paper\status_remote_probe.py`：在远端执行、打印一个 JSON
  （每行的 result/run_meta/history/train.log 摘要 + GPU / 磁盘 / screen / 队列日志尾巴），
  整个刷新只需**一次 ssh 往返**（`ssh` 用 `BatchMode=yes`，连不上会快速失败而不是卡住）。
- 新增开关 `-NoRemote`：跳过远端探测，只出本机部分。
- **修掉旧脚本一个真实缺陷**：本机这行是用*改动前*的 `run_matrix.py` 启动的，子进程 stdout 被块缓冲，
  `train.log` 可静默十几分钟，旧脚本据此判「可能卡住」。新版改看
  `train.log / history.json / last.pt / best_model.pt` 里**最新的一份**，并用 `history.json` 的
  mid-epoch 记录回推 epoch（本机行因此能正确显示 ep1 而不是 ep0）。
- 进度条改为**含部分轮次**（不再整轮跳）；ETA 用整行平均速率，不用 tqdm 的瞬时值。
- 旧版留档：`scripts\paper\status.ps1.bak_20261003`。

刷新方式：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File scripts\paper\status.ps1          # 打印 + 写 docs\path3_status.md
pwsh -NoProfile -ExecutionPolicy Bypass -File scripts\paper\status.ps1 -Watch   # 每 60 s 刷新
```
---

## 9. 远端磁盘清理（2026-10-03 19:23）

远端数据盘 105 GB，D2 开跑时 73 GB 已用 / 33 GB 可用，逼近脚本的 `MIN_FREE_GB=14` 守卫。

**删掉的是 8 轮那批模型权重**：`outputs/path3_clean/<row>/{best_model.pt,last.pt}`，
7 个目录共 14 个文件、**20.83 GB**（审计清单留在远端 `/tmp/d2_prune_audit.txt`）。
涉及行：`delivered` `A2_no_g2` `A6_no_page_memory` `A10_dense` `A11_no_gates` `A7_no_boundary_gate` `stage2_pilot_full_seed42`。

理由：

1. 这 7 行**全部**在 D2 里以 10 轮新协议重跑，8 轮的权重被取代。
2. D2 特意用新根 `outputs/path3_d2_10ep`，本来就不会 `--resume` 这批 `last.pt`，删掉**不影响**正在跑的队列。
3. 8 vs 10 对照只需要 `history.json`；四类小文件
   （`history.json` / `result.json` / `run_meta.json` / `train.log`，合计约 4.5 MB）**全部保留**。

**代价**：以后若要重跑 8 轮模型在 test 集上的评估，得重新训练。判断是值得——
最终进论文的 test 评估应该在 D2 的 10 轮模型上做，8 轮那批的读数已经取完
（记在 `docs/path3_results_summary_2026-10-02.md`）。

结果：

| | 清理前 | 清理后 |
|---|---|---|
| 已用 | 73 GB（69%） | **52 GB（50%）** |
| 可用 | 33 GB | **54 GB** |

D2 剩余 5 行约需 15 GB，跑完仍有 ~39 GB 余量。

**保留未动**：`final_train`（37 GB 语料）、`envs`（6.8 GB）、`hmsan-bsa/cache`、`outputs/path3_diag`、
`path3_clean` 的 json/log、`b1/b2/b3` 基线权重（2.35 GB）、`wheels`（836 MB）、
`hf_cache` + `hf_models_symlink_20260927.tar`（各 724 MB）、`hmsan_path3_cloud_bundle.tar.gz`（125 MB）。
后几项是重装环境 / 重建实例的保险，54 GB 余量下没必要动。

---

## 10. 停机与结论（2026-10-04 07:19，用户拍板）

**用户 2026-10-04 07:19 决定：停止 D2 重跑；D2 回退为 8 epoch；论文主表用 8 轮那套**
（`docs/path3_results_summary_2026-10-02.md`）。

### 10.1 停机时的状态

| 行 | 位置 | 停在 | 落地文件 |
|---|---|---|---|
| `delivered` | 远端 | **10/10 完成**，07:06:26 `exit=0` | `history.json` / `result.json` / `run_meta.json` / `train.log` + 3 个 `.pt`（随实例关机留在数据盘） |
| `full` | 远端 | **10/10 完成**，07:03:02 `exit=0` | 同上 |
| `A6_no_page_memory` | 远端 | 07:03 起跑，ep1 进行中（约 15 min） | `last.pt`（ep1 未完成）+ `train.log` |
| `A2_no_g2` | 远端 | 07:06 起跑，ep1 进行中（约 13 min） | 同上 |
| `A10_dense`、`A11_no_gates` | 远端 | **未启动**（排队中） | — |
| `A7_no_boundary_gate` | **本机** | **ep5 完成、ep6 进行中**（10-03 18:45:52 起跑，累计约 12.6 h） | `best_model.pt` / `checkpoint_epoch5.pt` / `last.pt` / `history.json` |

停机动作：

- 远端：`screen -S d2matrix -X quit` + 清残留进程，`nvidia-smi` 回到 **1 MiB / 0% / 39 ℃**；随后执行 `shutdown`，**已确认关机**（再连为 `Connection refused`）。数据盘 `/root/autodl-tmp` 保留：10 轮两行的 `.pt`、8 轮全部 json/log、`path3_clean` 的 json/log、b1/b2/b3 基线权重等。
- 本机：`Stop-Process` 杀 A7 进程树，回到 **739 MiB / 4% / 51 ℃**（0 个 `python.exe`）。
- 结果快照已拉回本机：`outputs/path3_d2_10ep/_remote_snapshot/`（`d2_pre_stop.tar.gz` / `d2_post_stop.tar.gz` + 解包后的四行 json/log，不含 `.pt`）。

### 10.2 停机的直接原因：10 轮依据被自己的两条实测证伪

| 运行（均为 `full` preset + seed 42） | ep8 包络 macro F1 | ep10 包络 macro F1 | 包络峰值轮 |
|---|---|---|---|
| `delivered`（8 轮原跑，10-02） | 0.8668 | — | ep8 |
| `delivered`（10 轮重跑，本次） | **0.8692** | 0.8615 | **ep8** |
| `full`（10 轮重跑，本次） | **0.8704** | 0.8526 | **ep8** |
| `delivered_ext16`（10-03 的修订依据） | 0.8668 | **0.89177** | ep10 |

- 两条新跑的包络峰值**都在第 8 轮**；三次独立运行在 ep8 上只差 **0.36 pp**。
- 10-03 依据里的 ep10 = 0.89177 **未被复现**（两个新跑分别低 2.9 / 3.9 pp）。
- ⇒ 「8 轮少报 2.5 pp」应读作**单次运行的散布**（且 `delivered_ext16` 是 `--resume` 续跑，优化器/调度器状态被重置，本就不等价于连跑 16 轮），**不是 epoch 预算效应**。

### 10.3 连带更正：`patience=1` 的触发次数

本文件 §1 与 `docs/path3_overnight_2026-10-03.md` §7 原写「`patience=1` 在 16 轮里降 7 次、末期 LR 剩 0.78%」。
**这是错的** —— `ReduceLROnPlateau` 的判据是 `num_bad_epochs > patience`（严格大于），`patience=1`
要**连续两轮**不进步才降一次。用真实调度器对象（`torch 2.13.0+cu126`）回放 `delivered` / `full`
两条 10 轮实测曲线，`patience=1` 都只在 **ep8、ep10** 触发（末期 LR = ×0.25）。

**对论文主表的影响：没有影响。** 8 轮矩阵用的是默认 `patience=3`；8 轮内 `patience=3` 一次都不触发，
而 `patience=1` 只会在 **ep8 末**（最后一次验证之后）触发一次 ⇒ **两种协议在 8 轮内的 LR 轨迹完全相同**，
8 轮的 val / test 读数不受这次协议改动影响。

### 10.4 遗留

- 远端数据盘保留的 10 轮权重（`delivered` / `full` 各 3 个 `.pt`，约 9.6 GB）与 A6/A2 的 `last.pt`
  未删；下次开机若要腾空间，优先删这些（8 轮的 json/log 不可删）。
- 下次开机只需「无卡模式」即可改代码/取文件，不必占用 GPU。
- `docs/path3_status.md` 仪表盘仍指向 `path3_d2_10ep`，下次刷新时会显示「已停机」状态。