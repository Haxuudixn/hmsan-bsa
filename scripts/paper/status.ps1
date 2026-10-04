# HMSAN-BSA D2 progress dashboard (2026-10-03).
#
# Read-only.  Watches the CURRENT work: the D2 matrix, 7 rows x 10 epochs at
# `patience=1`, whose row directories live in `outputs\path3_d2_10ep` here and
# in the same relative path on the rented 4090.  It never writes inside a run
# directory.
#
# Replaces the old dashboard, which pointed at `outputs\path3_matrix` - the
# matrix that the clean re-run superseded (ft3 warm start + split leakage).
#
#   .\scripts\paper\status.ps1              # print + refresh docs\path3_status.md
#   .\scripts\paper\status.ps1 -Watch       # refresh every 60 s in this console
#   .\scripts\paper\status.ps1 -NoWrite     # print only
#   .\scripts\paper\status.ps1 -NoRemote    # skip the ssh probe (no network / fast)
param(
    [string]$OutFile = "",
    [int]$IntervalSec = 60,
    [switch]$Watch,
    [switch]$NoWrite,
    [switch]$NoRemote
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$root = Join-Path $repo "outputs\path3_d2_10ep"
$probePath = Join-Path $PSScriptRoot "status_remote_probe.py"
if (-not $OutFile) { $OutFile = Join-Path $repo "docs\path3_status.md" }

$remoteHost = "connect.bjb1.seetacloud.com"
$remotePort = 52544
$remoteUser = "root"
$remotePython = "/root/autodl-tmp/envs/hmsan/bin/python"
$remoteRoot = "/root/autodl-tmp/hmsan-bsa/outputs/path3_d2_10ep"
$remoteSlots = 2

$epochsPlanned = 10
# One forward-only pass over the val split, per epoch (measured 2026-09-20).
$valSecondsPerEpoch = 273.0
# Fallback when train.log has not printed a progress bar yet (fresh start, or a
# run whose stdout is still block-buffered).
$fallbackStepsPerEpoch = 306

$rows = @(
    [pscustomobject]@{ Id = "delivered";           Host = "remote"; Role = "主模型（论文主表）" }
    [pscustomobject]@{ Id = "full";                Host = "remote"; Role = "架构基线 / 噪声底重复对" }
    [pscustomobject]@{ Id = "A6_no_page_memory";   Host = "remote"; Role = "跨页记忆消融" }
    [pscustomobject]@{ Id = "A2_no_g2";            Host = "remote"; Role = "value gate 消融" }
    [pscustomobject]@{ Id = "A10_dense";           Host = "remote"; Role = "稠密对照（兼作 B4 基线）" }
    [pscustomobject]@{ Id = "A11_no_gates";        Host = "remote"; Role = "全门控移除" }
    [pscustomobject]@{ Id = "A7_no_boundary_gate"; Host = "local";  Role = "顺序记忆写入门控移除（论文不报告）" }
)

$reProg = "Training:\s+\d+%\s*\|.*\|\s*(\d+)/(\d+)\s*\[([0-9:]+)<([0-9:]+),\s*([0-9.]+)s/it"
$reEpoch = "--- Epoch (\d+)/(\d+) ---"
$reMid = "\[mid ep(\d+) step (\d+)\]\s*acc=([0-9.]+)\s+macro_f1=([0-9.]+)"
$evidenceFiles = @("train.log", "history.json", "last.pt", "best_model.pt")

function Format-Hours([double]$Hours) {
    if ($Hours -lt 1.0) { return ("{0:N0} min" -f ($Hours * 60)) }
    if ($Hours -lt 48.0) { return ("{0:N1} h" -f $Hours) }
    return ("{0:N1} d" -f ($Hours / 24))
}

function Format-Bar([double]$Percent, [int]$Width = 24) {
    $filled = [int][Math]::Round($Percent / 100.0 * $Width)
    if ($filled -lt 0) { $filled = 0 }
    if ($filled -gt $Width) { $filled = $Width }
    return ("[" + ("#" * $filled) + ("." * ($Width - $filled)) + "]")
}

# train.log is being appended to while we read it, so open it shared.
function Read-LogText([string]$Path) {
    if (-not (Test-Path $Path)) { return "" }
    $stream = [System.IO.File]::Open($Path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
    $reader = New-Object System.IO.StreamReader($stream)
    $text = $reader.ReadToEnd()
    $reader.Dispose()
    $stream.Dispose()
    return $text
}

function Get-FreshestAgeMin([string]$Dir) {
    $ages = @()
    foreach ($name in $evidenceFiles) {
        $p = Join-Path $Dir $name
        if (Test-Path $p) { $ages += ((Get-Date) - (Get-Item $p).LastWriteTime).TotalMinutes }
    }
    if ($ages.Count -eq 0) { return $null }
    return [Math]::Round((($ages | Measure-Object -Minimum).Minimum), 1)
}

function Get-LocalRow([string]$Id) {
    $dir = Join-Path $root $Id
    $state = [ordered]@{
        id = $Id; host = "local"; exists = (Test-Path $dir); running = $false
        result_status = $null; started_at = $null; epochs_done = 0; curve = @()
        best_epoch = $null; best_acc = $null; best_f1 = $null
        last_epoch = $null; last_acc = $null; last_f1 = $null; mid = $null
        phase = ""; epoch_index = 0; epochs_planned = 0
        step_done = 0; step_total = 0; s_per_it = $null
        log_age_min = $null; fresh_min = $null; oom_count = 0; peak_vram_mb = $null
        has_last_pt = $false; has_best_pt = $false
    }
    if (-not $state.exists) { return [pscustomobject]$state }

    $resultPath = Join-Path $dir "result.json"
    $metaPath = Join-Path $dir "run_meta.json"
    $historyPath = Join-Path $dir "history.json"
    $logPath = Join-Path $dir "train.log"

    if (Test-Path $resultPath) {
        try { $state.result_status = "$((Get-Content $resultPath -Raw -Encoding UTF8 | ConvertFrom-Json).status)" } catch { $state.result_status = "unreadable" }
    }
    if (Test-Path $metaPath) {
        try { $state.started_at = (Get-Content $metaPath -Raw -Encoding UTF8 | ConvertFrom-Json).started_at } catch { }
    }
    if (Test-Path $historyPath) {
        try {
            # One record per epoch end (no "phase" key) plus three mid-epoch
            # probes per epoch ("phase" = "mid_epoch").
            $hist = @(Get-Content $historyPath -Raw -Encoding UTF8 | ConvertFrom-Json)
            $ends = @($hist | Where-Object { $null -eq $_.phase })
            $mids = @($hist | Where-Object { "$($_.phase)" -eq "mid_epoch" })
            $state.epochs_done = $ends.Count
            $state.curve = @($ends | ForEach-Object {
                [pscustomobject]@{ epoch = [int]$_.epoch; acc = [double]$_.val.accuracy; f1 = [double]$_.val.macro_f1 }
            })
            if ($ends.Count -gt 0) {
                $best = $ends | Sort-Object { [double]$_.val.accuracy } -Descending | Select-Object -First 1
                $last = $ends[-1]
                $state.best_epoch = [int]$best.epoch
                $state.best_acc = [double]$best.val.accuracy
                $state.best_f1 = [double]$best.val.macro_f1
                $state.last_epoch = [int]$last.epoch
                $state.last_acc = [double]$last.val.accuracy
                $state.last_f1 = [double]$last.val.macro_f1
                $state.peak_vram_mb = $last.peak_vram_mb
            }
            if ($mids.Count -gt 0) {
                $mid = $mids[-1]
                $state.mid = [pscustomobject]@{
                    epoch = [int]$mid.epoch; step = [int]$mid.step
                    acc = [double]$mid.val.accuracy; macro_f1 = [double]$mid.val.macro_f1
                }
            }
        } catch { $state.epochs_done = 0 }
    }

    $text = Read-LogText $logPath
    if ($text) {
        try { $state.log_age_min = [Math]::Round(((Get-Date) - (Get-Item $logPath).LastWriteTime).TotalMinutes, 1) } catch { }
        $state.oom_count = ([regex]::Matches($text, "\[oom\]")).Count
        $lines = $text -split "[\r\n]+" | Where-Object { $_ -ne "" }
        foreach ($line in $lines) {
            if ($line -match $reEpoch) {
                $state.epoch_index = [int]$Matches[1]; $state.epochs_planned = [int]$Matches[2]
            } elseif ($line -match $reProg) {
                $state.phase = "Training"; $state.step_done = [int]$Matches[1]
                $state.step_total = [int]$Matches[2]; $state.s_per_it = [double]$Matches[5]
            } elseif ($line -match "Validation:") {
                $state.phase = "Validation"
            }
        }
        $midMatches = [regex]::Matches($text, $reMid)
        if ($midMatches.Count -gt 0) {
            $m = $midMatches[$midMatches.Count - 1]
            $state.mid = [pscustomobject]@{
                epoch = [int]$m.Groups[1].Value; step = [int]$m.Groups[2].Value
                acc = [double]$m.Groups[3].Value; macro_f1 = [double]$m.Groups[4].Value
            }
        }
    }
    # The local row runs under an older run_matrix.py that block-buffers the
    # child, so train.log can sit minutes stale while history.json keeps
    # updating.  Fall back to the mid-epoch probe for "which epoch", and judge
    # liveness on the freshest artifact rather than on train.log alone.
    if ([int]$state.epoch_index -eq 0 -and $state.mid) { $state.epoch_index = [int]$state.mid.epoch }
    $state.fresh_min = Get-FreshestAgeMin $dir
    $state.has_last_pt = Test-Path (Join-Path $dir "last.pt")
    $state.has_best_pt = Test-Path (Join-Path $dir "best_model.pt")
    return [pscustomobject]$state
}

function Get-RemoteProbe {
    if ($NoRemote) { return $null }
    if (-not (Test-Path $probePath)) { return $null }
    $sshExe = Join-Path $env:SystemRoot "System32\OpenSSH\ssh.exe"
    if (-not (Test-Path $sshExe)) { $sshExe = "ssh" }
    $py = [IO.File]::ReadAllText($probePath)
    $b64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes(($py -replace "`r", "")))
    $target = "$remoteUser@$remoteHost"
    $cmd = "echo $b64 | base64 -d | $remotePython - $remoteRoot"
    try {
        # BatchMode: fail fast instead of blocking on a password prompt.
        $raw = & $sshExe -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=12 -p $remotePort $target $cmd 2>$null
    } catch { return $null }
    if (-not $raw) { return $null }
    try { return (($raw -join "`n") | ConvertFrom-Json) } catch { return $null }
}

$now = Get-Date
$localRows = @{}
foreach ($row in $rows) { if ($row.Host -eq "local") { $localRows[$row.Id] = Get-LocalRow $row.Id } }

$trainProcs = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match "train\.py" })
$localRunning = @{}
foreach ($row in $rows) {
    if ($row.Host -ne "local") { continue }
    $dir = Join-Path $root $row.Id
    $hit = @($trainProcs | Where-Object { $_.CommandLine -like "*$dir*" })
    $localRunning[$row.Id] = ($hit.Count -gt 0)
}

$remoteProbe = $null
$remoteFailed = $false
if (-not $NoRemote) {
    $remoteProbe = Get-RemoteProbe
    if ($null -eq $remoteProbe) { $remoteFailed = $true }
}
$remoteRows = @{}
if ($remoteProbe) {
    foreach ($entry in @($remoteProbe.rows)) { if ($entry) { $remoteRows["$($entry.id)"] = $entry } }
}

# --- merge into one per-row view -------------------------------------------
$views = @()
foreach ($row in $rows) {
    if ($row.Host -eq "local") {
        $s = $localRows[$row.Id]
        $isRunning = [bool]$localRunning[$row.Id]
    } else {
        if ($remoteRows.ContainsKey($row.Id)) {
            $s = $remoteRows[$row.Id]
            $isRunning = [bool]$s.running
        } else {
            $s = [pscustomobject]@{ id = $row.Id; exists = $false; epochs_done = 0 }
            $isRunning = $false
        }
    }
    $status = "pending"
    if ("$($s.result_status)" -like "completed*") { $status = "completed" }
    elseif ($isRunning) { $status = "running" }
    elseif ($s.exists -and ($s.epochs_done -gt 0 -or $s.started_at)) { $status = "stalled" }
    elseif ($s.exists) { $status = "pending" }

    $total = if ([int]$s.step_total -gt 0) { [int]$s.step_total } else { $fallbackStepsPerEpoch }
    $stepDone = [int]$s.step_done
    if ($stepDone -eq 0 -and $s.mid) { $stepDone = [int]$s.mid.step }

    $elapsedH = $null
    if ($status -eq "running" -and $s.started_at) {
        try { $elapsedH = ($now - [datetime]::Parse("$($s.started_at)")).TotalHours } catch { $elapsedH = $null }
    }
    $etaH = $null
    if ($status -eq "running") {
        $curEpoch = [Math]::Max(1, [int]$s.epoch_index)
        $doneIters = ($curEpoch - 1) * $total
        if ("$($s.phase)" -eq "Validation") { $doneIters += $total } else { $doneIters += $stepDone }
        $rate = $null
        if ($doneIters -gt 0 -and $elapsedH -and $elapsedH -gt 0) { $rate = $elapsedH * 3600.0 / $doneIters }
        elseif ($s.s_per_it) { $rate = [double]$s.s_per_it }
        if ($rate) {
            $remainIters = $epochsPlanned * $total - $doneIters
            $valLeft = $valSecondsPerEpoch * ($epochsPlanned - $curEpoch + 1)
            $etaH = [Math]::Round(($remainIters * $rate + $valLeft) / 3600.0, 2)
        }
    }

    $frac = 0.0
    if ($status -eq "completed") { $frac = 1.0 }
    else {
        $doneEpochs = [Math]::Min($epochsPlanned, [int]$s.epochs_done)
        $partial = 0.0
        if ($total -gt 0) { $partial = [Math]::Min(1.0, $stepDone / [double]$total) }
        $frac = [Math]::Min(1.0, ($doneEpochs + $partial) / [double]$epochsPlanned)
    }

    $views += [pscustomobject]@{
        Row = $row; State = $s; Status = $status; Running = $isRunning
        ElapsedH = $elapsedH; EtaH = $etaH; Total = $total; StepDone = $stepDone; Frac = $frac
    }
}

$doneRowEpochs = 0.0
foreach ($v in $views) { $doneRowEpochs += $v.Frac * $epochsPlanned }
$totalRowEpochs = $rows.Count * $epochsPlanned
$progressPct = if ($totalRowEpochs -gt 0) { [Math]::Round($doneRowEpochs / $totalRowEpochs * 100.0, 1) } else { 0.0 }

$runningViews = @($views | Where-Object { $_.Status -eq "running" })
$completedViews = @($views | Where-Object { $_.Status -eq "completed" })

$remoteRowsLeft = @($views | Where-Object { $_.Row.Host -eq "remote" -and $_.Status -ne "completed" }).Count
$rowHours = ($views | Where-Object { $_.Row.Host -eq "remote" -and $_.EtaH } | Measure-Object EtaH -Maximum).Maximum
$remoteQueueEta = $null
if ($rowHours -and $remoteRowsLeft -gt 0) {
    # Pipelined: when a slot frees, the next pending row starts immediately, so
    # the queue drains in ceil(rows-left / slots) waves of one row each.
    $remoteQueueEta = [Math]::Ceiling($remoteRowsLeft / [double]$remoteSlots) * $rowHours
}

$report = New-Object System.Collections.Generic.List[string]
function Add-Line([string]$Text) { $script:report.Add($Text) }

Add-Line "# HMSAN-BSA D2 矩阵进度（10 epoch / patience=1）—— 已停机"
Add-Line ""
Add-Line "> ⛔ **D2 已于 2026-10-04 07:19 停机（用户拍板），epoch 预算回退为 8。** 远端实例已关机。"
Add-Line "> 下面的「总览 / 当前在跑 / 逐行状态」是**历史记录**，远端行因探测不到会显示成未跑，不要据此重启队列。"
Add-Line "> 真实结果看下方「停机时的最终状态」；论文主表 = ``docs\path3_results_summary_2026-10-02.md``（8 轮矩阵），"
Add-Line "> 停机记录 = ``docs\path3_d2_matrix_run_2026-10-03.md`` §10。"
Add-Line ""
Add-Line "> 由 ``scripts\paper\status.ps1`` 自动生成，每次运行覆盖。检查时间：$($now.ToString('yyyy-MM-dd HH:mm:ss'))"
Add-Line "> 盯的是 **D2 矩阵**：``outputs\path3_d2_10ep``（本机）+ 租用 4090 上的同一矩阵。协议 ``--epochs 10 --lr_patience 1 --lr_min_lr 1e-5``、冷启动。"
Add-Line ""
Add-Line "## 总览"
Add-Line ""
Add-Line "``$(Format-Bar $progressPct)`` **$progressPct%**   $("{0:N1}" -f $doneRowEpochs) / $totalRowEpochs 行·epoch（$($rows.Count) 行 × $epochsPlanned epoch）"
Add-Line ""
$runningIdsText = if ($runningViews.Count -gt 0) { ($runningViews | ForEach-Object { "``$($_.Row.Id)``" }) -join '、' } else { '无' }
Add-Line "| 指标 | 值 |"
Add-Line "|---|---|"
Add-Line "| 已完成行 | $($completedViews.Count) / $($rows.Count) |"
Add-Line "| 进行中 | $runningIdsText |"
Add-Line "| 分工 | 本机 1 行（``A7_no_boundary_gate``）+ 远端 6 行（$remoteSlots 路并行） |"
if ($remoteQueueEta) { Add-Line "| 远端整队跑完 | 约 $(Format-Hours $remoteQueueEta)（$remoteRowsLeft 行待跑，$remoteSlots 路并行；预计 $($now.AddHours($remoteQueueEta).ToString('MM-dd HH:mm'))） |" }
$localEtaView = @($views | Where-Object { $_.Row.Host -eq "local" -and $_.EtaH })
if ($localEtaView.Count -gt 0) { Add-Line "| 本机跑完 | 约 $(Format-Hours $localEtaView[0].EtaH)（预计 $($now.AddHours($localEtaView[0].EtaH).ToString('MM-dd HH:mm'))） |" }
if ($remoteFailed) { Add-Line "| 远端探测 | **失败**（ssh 连不通或未开机；本机行仍照常显示） |" }
Add-Line ""

Add-Line "## 当前在跑"
Add-Line ""
if ($runningViews.Count -eq 0) {
    Add-Line "当前没有训练进程在跑。"
} else {
    foreach ($v in $runningViews) {
        $s = $v.State
        $where = if ($v.Row.Host -eq "local") { "本机" } else { "远端" }
        $rateText = if ($s.s_per_it) { "（$($s.s_per_it) s/样本）" } else { "" }
        $phaseText = if ("$($s.phase)" -eq "Validation") { "Validation" } elseif ($v.StepDone -gt 0) { "Training $($v.StepDone)/$($v.Total)$rateText" } else { "启动中（尚未写出首个进度条 / 首个中间验证）" }
        Add-Line "### $where · ``$($v.Row.Id)``"
        Add-Line ""
        Add-Line "- epoch $([int]$s.epoch_index)/$epochsPlanned   $phaseText"
        if ($s.last_epoch) { Add-Line "- 已完成轮次的 val：ep$($s.last_epoch) acc **$("{0:N4}" -f $s.last_acc)** / macro F1 $("{0:N4}" -f $s.last_f1)（最好 ep$($s.best_epoch) acc $("{0:N4}" -f $s.best_acc) / F1 $("{0:N4}" -f $s.best_f1)）" }
        if ($s.mid) { Add-Line "- 最近中间验证：ep$($s.mid.epoch) step $($s.mid.step) acc $("{0:N4}" -f $s.mid.acc) / F1 $("{0:N4}" -f $s.mid.macro_f1)" }
        if ($v.ElapsedH) {
            $etaBit = if ($v.EtaH) { "；预计还需 **" + (Format-Hours $v.EtaH) + "**（约 " + $now.AddHours($v.EtaH).ToString("MM-dd HH:mm") + "）" } else { "" }
            Add-Line ("- 已跑 " + ("{0:N2}" -f $v.ElapsedH) + " h" + $etaBit)
        }
        $fresh = if ($null -ne $s.fresh_min) { $s.fresh_min } else { $s.log_age_min }
        if ($null -ne $fresh) {
            $staleWarn = if ($fresh -gt 45) { "  **警告：可能卡住**" } else { "" }
            Add-Line "- 产出新鲜度：$fresh min（train.log / history.json / *.pt 里最新的一份）$staleWarn；[oom] 计数 $($s.oom_count)（应为 0）"
        }
        Add-Line ""
    }
}

Add-Line "## 逐行状态"
Add-Line ""
Add-Line "| # | 行 | 机器 | 说明 | 状态 | epoch | 最好 val acc | 最好 macro F1 | 峰值显存 | ETA |"
Add-Line "|---|---|---|---|---|---|---|---|---|---|"
$idx = 0
foreach ($v in $views) {
    $idx++
    $s = $v.State
    $where = if ($v.Row.Host -eq "local") { "本机" } else { "远端" }
    $statusText = switch ($v.Status) {
        "completed" { "完成" }
        "running"   { "**进行中**" }
        "stalled"   { "**中断/停滞**" }
        default     { "待跑" }
    }
    $epochText = "$([int]$s.epochs_done)/$epochsPlanned"
    $accText = if ($s.best_acc) { "$("{0:N4}" -f $s.best_acc) (ep$($s.best_epoch))" } else { "-" }
    $f1Text = if ($s.best_f1) { "$("{0:N4}" -f $s.best_f1) (ep$($s.best_epoch))" } else { "-" }
    $vramText = if ($s.peak_vram_mb) { "$("{0:N0}" -f $s.peak_vram_mb) MB" } else { "-" }
    $etaText = if ($v.EtaH) { Format-Hours $v.EtaH } else { "-" }
    Add-Line "| $idx | ``$($v.Row.Id)`` | $where | $($v.Row.Role) | $statusText | $epochText | $accText | $f1Text | $vramText | $etaText |"
}
Add-Line ""
Add-Line "val 两列**成对取自同一轮**（val acc 最高的那一轮，也就是 ``best_model.pt`` 对应模型），不是把两个指标各自的最大值拼成两个模型。"
Add-Line "D6：落表口径以**末轮**为主，acc 与 macro F1 永远成对出现，且必须附支持度分层。"
Add-Line ""

$curveViews = @($views | Where-Object { $_.State.curve -and @($_.State.curve).Count -gt 0 })
if ($curveViews.Count -gt 0) {
    Add-Line "## val 曲线（每轮末）"
    Add-Line ""
    $headerCells = (1..$epochsPlanned | ForEach-Object { "ep$_" }) -join ' | '
    $sepCells = ('---|' * ($epochsPlanned + 1))
    Add-Line "| 行 | 指标 | $headerCells |"
    Add-Line "|---|$sepCells"
    foreach ($v in $curveViews) {
        $byEpoch = @{}
        foreach ($p in @($v.State.curve)) { $byEpoch[[int]$p.epoch] = $p }
        $accCells = @(); $f1Cells = @()
        foreach ($e in 1..$epochsPlanned) {
            if ($byEpoch.ContainsKey($e)) {
                $accCells += ("{0:N4}" -f $byEpoch[$e].acc)
                $f1Cells += ("{0:N4}" -f $byEpoch[$e].f1)
            } else { $accCells += "-"; $f1Cells += "-" }
        }
        Add-Line "| ``$($v.Row.Id)`` | val acc | $($accCells -join ' | ') |"
        Add-Line "| | macro F1 | $($f1Cells -join ' | ') |"
    }
    Add-Line ""
}

$smiRaw = & nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader 2>$null
$localGpuText = "不可用"
if ($smiRaw) {
    $parts = @(($smiRaw -join " ") -split ",") | ForEach-Object { $_.Trim() }
    if ($parts.Count -ge 4) { $localGpuText = "$($parts[2] -replace ' %', '')% 利用率，$($parts[0] -replace ' MiB', '') / $($parts[1] -replace ' MiB', '') MiB，$($parts[3] -replace ' C', '') C" }
}
$localDiskText = "$([Math]::Round((Get-PSDrive C).Free / 1GB, 1)) GB"
$remoteGpuText = "未知"
$remoteDiskText = "未知"
if ($remoteProbe -and $remoteProbe.machine) {
    $m = $remoteProbe.machine
    if ($null -ne $m.gpu_util) { $remoteGpuText = "$($m.gpu_util)% 利用率，$($m.gpu_mem_used) / $($m.gpu_mem_total) MiB，$($m.gpu_temp) C" }
    if ($null -ne $m.disk_free_gb) { $remoteDiskText = "$($m.disk_free_gb) GB" }
}
$localProcText = if ($trainProcs.Count -gt 0) { "运行中（PID " + ($trainProcs.ProcessId -join ", ") + "）" } else { "无" }
$remoteScreenText = "未知"
if ($remoteProbe -and $remoteProbe.machine.screen) {
    $hit = @($remoteProbe.machine.screen -split "`n" | Where-Object { $_ -match "d2matrix" })
    if ($hit.Count -gt 0) { $remoteScreenText = ($hit[0] -replace "\s+", " ").Trim() } else { $remoteScreenText = "无 d2matrix 会话" }
}

Add-Line "## 机器状态"
Add-Line ""
Add-Line "| 项 | 本机（RTX 3060 12 GB） | 远端（RTX 4090 24 GB） |"
Add-Line "|---|---|---|"
Add-Line "| GPU | $localGpuText | $remoteGpuText |"
Add-Line "| 磁盘空闲 | $localDiskText | $remoteDiskText |"
Add-Line "| 训练进程 | $localProcText | $remoteScreenText |"
Add-Line ""

if ($remoteProbe -and $remoteProbe.machine -and $remoteProbe.machine.queue_log_tail) {
    Add-Line "## 远端队列日志（尾 8 行）"
    Add-Line ""
    Add-Line '```'
    foreach ($ln in @($remoteProbe.machine.queue_log_tail)) { Add-Line "$ln" }
    Add-Line '```'
    Add-Line ""
}

$snapRoot = Join-Path $root "_remote_snapshot"
if (Test-Path $snapRoot) {
    Add-Line "## 停机时的最终状态（本机快照 ``_remote_snapshot``）"
    Add-Line ""
    Add-Line "远端已关机，下表读的是本机留存的 json（快照不含 ``.pt``）。"
    Add-Line ""
    Add-Line "| 行 | result.status | epoch 末记录 | 轮末 val acc | 轮末 macro F1 | 包络最好 val acc | 包络最好 macro F1 |"
    Add-Line "|---|---|---|---|---|---|---|"
    foreach ($d in (Get-ChildItem $snapRoot -Directory -ErrorAction SilentlyContinue | Sort-Object Name)) {
        $hj = Join-Path $d.FullName "history.json"
        if (-not (Test-Path $hj)) { Add-Line "| ``$($d.Name)`` | （无 history.json） | - | - | - | - | - |"; continue }
        $h = $null
        try { $h = @(Get-Content $hj -Raw -Encoding UTF8 | ConvertFrom-Json) } catch { $h = $null }
        $st = "-"
        $rj = Join-Path $d.FullName "result.json"
        if (Test-Path $rj) { try { $st = "$((Get-Content $rj -Raw -Encoding UTF8 | ConvertFrom-Json).status)" } catch { $st = "?" } }
        if (-not $h -or $h.Count -eq 0) { Add-Line "| ``$($d.Name)`` | $st | 0（已起跑未完成 1 轮） | - | - | - | - |"; continue }
        $ends = @($h | Where-Object { -not $_.phase })
        $bestAcc = @($h | Sort-Object { [double]$_.val.accuracy } -Descending)[0]
        $bestF1 = @($h | Sort-Object { [double]$_.val.macro_f1 } -Descending)[0]
        $la = "-"; $lf = "-"
        if ($ends.Count -gt 0) { $la = "{0:N4}" -f [double]$ends[-1].val.accuracy; $lf = "{0:N4}" -f [double]$ends[-1].val.macro_f1 }
        Add-Line "| ``$($d.Name)`` | $st | $($ends.Count) | $la | $lf | $("{0:N4}" -f [double]$bestAcc.val.accuracy) (ep$($bestAcc.epoch)) | $("{0:N4}" -f [double]$bestF1.val.macro_f1) (ep$($bestF1.epoch)) |"
    }
    Add-Line ""
    Add-Line "读法：**论文主表取 8 轮矩阵**（``docs\path3_results_summary_2026-10-02.md``）。本表的 10 轮数据只作预算敏感性证据 ——"
    Add-Line "两条跑满的行（``delivered`` / ``full``）的 macro F1 包络峰值都落在 **ep8**，即 10 轮没有换来增益。"
    Add-Line ""
}

Add-Line "## 现在该做什么"
Add-Line ""
if ($runningViews.Count -gt 0) {
    Add-Line "1. 正常推进，无需干预。远端队列是 ``screen -dmS d2matrix``，行序 ``delivered -> full -> A6_no_page_memory -> A2_no_g2 -> A10_dense -> A11_no_gates``（2 路并行，A7 由本机跑）。"
    Add-Line "2. 期间**不要**改 ``train.py`` / ``scripts\paper\run_matrix.py`` / ``scripts\paper\run_d2_matrix.sh``（正在被使用）。"
    Add-Line "3. 全部跑完后：拉回远端结果 -> ``scripts\paper\summarize_comparison.py`` + ``scripts\paper\support_stratified.py`` -> 与 8 轮结果对照 -> 按 D7 判据决定补不补 seed 43/44。"
} elseif ($completedViews.Count -eq $rows.Count) {
    Add-Line "1. 7 行全部完成：跑 ``summarize_comparison.py`` 与 ``support_stratified.py``。"
    Add-Line "2. 与 8 轮结果做 8 vs 10 对照，确认 patience=1 没把早期轮次压坏。"
} else {
    Add-Line "1. **D2 已停机，不要重启这个队列。** epoch 预算已回退为 **8**，论文主表 = ``docs\path3_results_summary_2026-10-02.md``（8 轮矩阵）。"
    Add-Line "2. 停机原因：10 轮的依据（`delivered_ext16` 的 ep10 包络 0.89177）**未被复现** —— 10 轮两条跑满的行都在 ep8 到顶。见 ``docs\path3_d2_matrix_run_2026-10-03.md`` §10。"
    Add-Line "3. 若要继续做实验，先与用户确认方向（补种子 / 基线权重对齐 / 图像分支消融），不要凭本仪表盘的历史状态自行开跑。"
}
Add-Line ""
Add-Line "---"
Add-Line ""
Add-Line "查看方式：``& $repo\scripts\paper\status.ps1``（每次重新生成）；``-Watch`` 每 60 s 刷新；``-NoWrite`` 只打印；``-NoRemote`` 跳过远端探测。"
Add-Line "权威记录是每行的 ``outputs\path3_d2_10ep\<row>\history.json``（train.py 每 75 步写一次），不是控制台日志。"

$reportText = ($report -join "`n")
if ($Watch) {
    while ($true) {
        Clear-Host
        & pwsh -NoProfile -ExecutionPolicy Bypass -File $PSCommandPath -NoWrite
        Write-Host ""
        Write-Host "[status] $((Get-Date).ToString('HH:mm:ss')) - $IntervalSec 秒后刷新（Ctrl+C 退出）"
        Start-Sleep -Seconds $IntervalSec
    }
}

Write-Host $reportText
if (-not $NoWrite) {
    [System.IO.File]::WriteAllText($OutFile, $reportText + "`n", (New-Object System.Text.UTF8Encoding($false)))
    Write-Host ""
    Write-Host "[status] 已刷新 $OutFile"
}