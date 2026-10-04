# 路径 3 停机 / 恢复说明（2026-09-24 23:30，用户重启加装内存条）

## 1 为什么停机
用户需要重启电脑并加装两条内存条。训练以"脱离会话"方式运行，重启会中断它，因此主动停机。

## 2 停机时的状态（全部为可信的完成态）
- 已完成 **9 行**（val acc / val macro F1，成对取自 val acc 最高轮）：
  | 行 | val acc | val macro F1 | wall |
  |---|---|---|---|
  | full_3ep | 0.9694 (ep1) | 0.8416 | 5.31 h |
  | A6_no_page_memory | 0.9653 (ep2) | 0.8774 | 5.37 h |
  | A7_no_boundary_gate | 0.9574 (ep1) | 0.8739 | 7.40 h |
  | A9_local_window | 0.9742 (ep3) | 0.8747 | 10.90 h |
  | A11_no_gates | 0.9367 (ep2) | 0.8199 | 6.06 h |
  | frozen_both | 0.9766 (ep3) | 0.8856 | 2.32 h |
  | A1_no_g1 | 0.9563 | 0.8790 | 1.34 h（续跑） |
  | A2_no_g2 | 0.9685 | 0.9035 | 5.58 h |
  | A3_fixed_k | 0.9696 | 0.8690 | 5.14 h |
- **剩余 4 行**：A4_no_gated_fusion / A5_standard_ffn / A8_no_interpage_gate / A10_dense
- A4 停在上线后约 13 分钟（目录里只有 config.json 与 train.log，**无任何 checkpoint**），恢复后从零开始，**没有损失**
- 停机动作：先停 runner（pid 41608，防止它触发 120 s 自动重试），再停 run_train_final 包装进程（72736）、GPU 遥测（68972）、python train.py（3744）。停机后 `python.exe` 计数 0，GPU 445 MiB / 4%
- 监控定时任务 `hmsan-bsa-matrix` 已置为 **PAUSED**（否则会在关机窗口里反复报警）

## 3 重启后如何恢复（两步）

### 3.1 启动训练（必须用 -Command 传数组，不能用 -File）
`pwsh -File` 只会给 `-Only` 绑定一个值，其余值会按位置落到 `-BaseOutput`/`-MaxAttempts` 上并报
`Cannot convert value "A2_no_g2" to type "System.Int32"`（2026-09-24 11:07 的失败日志：
`outputs\path3_matrix\runner_err-20260924-110759.log`）。正确写法：

```powershell
$repo='C:\Users\Administrator\Documents\Codex\2026-08-20\https\outputs\hmsan-bsa'
$stamp=Get-Date -Format 'yyyyMMdd-HHmmss'
$cmd="& '$repo\run_comparison.ps1' -Only 'A11_no_gates','frozen_both','A1_no_g1','A2_no_g2','A3_fixed_k','A4_no_gated_fusion','A5_standard_ffn','A8_no_interpage_gate','A10_dense' -StopAfter 9"
Start-Process -FilePath 'pwsh' -WindowStyle Hidden -WorkingDirectory $repo `
  -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-Command',$cmd) `
  -RedirectStandardOutput "$repo\outputs\path3_matrix\runner_out-$stamp.log" `
  -RedirectStandardError  "$repo\outputs\path3_matrix\runner_err-$stamp.log" -PassThru
```

已完成的行会被 `skip ... (already completed)` 跳过，A4 从头开始。启动后 30 s 内 `matrix.log`
应出现新的 `skip` 行与 `start A4_no_gated_fusion`。

### 3.2 恢复监控定时任务
`hmsan-bsa-matrix`（cron，每 30 分钟，model=deepseek-v4-pro）在重启后需要把 status 改回 `ACTIVE`，
并把 prompt 里的"当前状态"更新为当时实际进度。

## 4 内存加装后可以顺手确认的事
- 用 `(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB` 看物理内存是否已从 8 GB 提升
- 若提升明显，`both_unfrozen`（FF 口径，当初因 12 GB 显存 + 7.95 GB 主机内存双缺而失败，
  见执行计划 11.6）可以重新评估是否解禁；**但不要自动启动，先问用户**

## 5 硬性约束
- 恢复后不要再修改 `run_comparison.ps1` / `run_train_final.ps1` / `train.py`
- 全部 15 行跑完后按 `docs\path3_final_review_brief.md` 做整体评估，不要只跑汇总脚本
---

## 6 恢复记录（2026-09-25 09:59）

### 6.1 内存变更结果（用户 2026-09-24 加装 2×8 GB）
| 槽位 | 通道 | 容量 | 标称 | 实际 | P/N |
|---|---|---|---|---|---|
| DIMM 1 | P0 CHANNEL A | 8 GB | 2400 | 2400 | KHX2400C15/8G |
| DIMM 0 | P0 CHANNEL B | 8 GB | 2400 | 2400 | KHX2400C15/8G |
| DIMM 1 | P0 CHANNEL B | 8 GB | 2400 | 2400 | KHX2400C15/8G |

- 三根同型号，A/B 两通道均有内存（16 GB 双通道 + 8 GB 单通道，Flex 模式），总计 **23.95 GB**，全部实际运行 2400 MT/s（未掉频）

### 6.2 稳定性风险（未完全排除，务必留意）
- 2026-09-24 23:31 一次 Kernel-Power 41（BugcheckCode=0，无 Bugcheck）
- 2026-09-24 23:33 **0x0000004E PFN_LIST_CORRUPT** 蓝屏，转储 `C:\Windows\Minidump\092426-16609-01.dmp`（伴随 volmgr 162 转储写入异常）
- 更早历史：09-17 / 09-18 各一次 **0x101 CLOCK_WATCHDOG_TIMEOUT**；09-17 22:20 一条 **WHEA Cache Hierarchy 致命硬件错误**；三周内共 5 个转储（09-08 / 09-15 / 09-17 / 09-18 / 09-24）
- 主板 BIOS 仍为 **2.60（2019-12-30）**，未更新
- **未做内存测试**（MemTest86 / mdsched 均未运行）
- 判断依据：自 2026-09-24 23:33 开机后连续稳定 10 h 25 min，无蓝屏 / 无 WHEA / 无新转储，据此恢复训练

### 6.3 恢复动作
- 2026-09-25 09:59:03 以脱离会话方式恢复训练（runner pid=15700，日志 `outputs\path3_matrix\runner_out-20260925-095902.log`）
- 自动跳过已完成 9 行，从 `A4_no_gated_fusion` 开始；剩余 A4 / A5 / A8 / A10
- 监控定时任务 `hmsan-bsa-matrix` 已恢复为 ACTIVE，并在 prompt 里加入"WHEA / Kernel-Power 41 / 新转储文件"告警条件

### 6.4 若再次出现硬件稳定性问题，处理顺序
1. 先跑 MemTest86（≥2 轮）确认是否内存不稳
2. 升级到最新**非 Beta** BIOS（新 AGESA 对 3 根混插与 2400 的兼容性有实质改善）
3. 或退回 2 根（A2+B2，16 GB 纯双通道；剩余 4 行在 8 GB 时即可运行，16 GB 充裕）
4. 或放宽 Command Rate 到 2T / 内存降到 2133 再测