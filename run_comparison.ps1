# HMSAN-BSA comparison-matrix runner (Phase 0, item W2 / E0.4).
#
# Reads experiments\path3_manifest.json and runs every enabled row serially,
# one output directory each, with retry-on-crash resume from last.pt.  After
# each row it writes result.json (status, wall time, attempts, val metrics,
# peak VRAM) so the summary script never has to parse a log.
#
#     .\run_comparison.ps1 -DryRun                 # show the plan, touch nothing
#     .\run_comparison.ps1 -StopAfter 1            # calibration run (M0)
#     .\run_comparison.ps1                         # run everything pending
#     .\run_comparison.ps1 -Only full_3ep A6_no_page_memory
#
# Protocol locks enforced here:
#   * every row uses the same split, optimizer, caps and seed from the manifest
#   * a row only counts as done when run_meta.json says status=completed
#   * result.json records the ablation signature, so a row trained with the
#     wrong switches cannot be mistaken for the real thing
param(
    [string]$Manifest = "",
    [string]$BaseOutput = "",
    [string[]]$Only = @(),
    [int]$MaxAttempts = 3,
    [int]$RetryDelaySec = 120,
    [int]$StopAfter = 0,
    [switch]$Force,
    [switch]$DryRun,
    [switch]$Prune,
    [switch]$NoMonitor,
    [switch]$LowPriority
)

$ErrorActionPreference = "Stop"
$repo = $PSScriptRoot
$Py = "C:\hmsan\.venv\Scripts\python.exe"
$runner = Join-Path $repo "run_train_final.ps1"
$monitorScript = Join-Path $repo "scripts\paper\monitor_gpu.ps1"

if (-not $Manifest) { $Manifest = Join-Path $repo "experiments\path3_manifest.json" }
if (-not $BaseOutput) { $BaseOutput = Join-Path $repo "outputs\path3_matrix" }
if (-not (Test-Path $Manifest)) { Write-Error "manifest missing: $Manifest"; exit 1 }
if (-not (Test-Path $Py)) { Write-Error "venv missing: $Py"; exit 1 }

New-Item -ItemType Directory -Force -Path $BaseOutput | Out-Null
$matrixLog = Join-Path $BaseOutput "matrix.log"
$monitorCsv = Join-Path $BaseOutput "monitor.csv"

function Log([string]$message) {
    $line = "[matrix] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $message"
    Write-Host $line
    $line | Add-Content -Path $matrixLog -Encoding utf8
}

$manifestData = Get-Content $Manifest -Raw -Encoding UTF8 | ConvertFrom-Json
$protocol = $manifestData.protocol
$defaultCacheDir = Join-Path $repo "cache\google__vit-base-patch16-224-bf16"

$runs = @($manifestData.runs) |
    Where-Object { -not ($_.PSObject.Properties.Name -contains 'enabled' -and $_.enabled -eq $false) } |
    Sort-Object order
if ($Only.Count -gt 0) {
    $runs = $runs | Where-Object { $Only -contains $_.id }
}
if ($runs.Count -eq 0) { Write-Error "no run selected"; exit 1 }

Log "manifest=$Manifest"
Log "output=$BaseOutput"
Log "rows selected: $($runs.Count)  (MaxAttempts=$MaxAttempts RetryDelaySec=$RetryDelaySec)"

# ── dry run: show what would happen and stop ──
if ($DryRun) {
    foreach ($run in @($runs)) {
        $cache = ""
        if ($run.vit_cache -eq $true) { $cache = $defaultCacheDir }
        "{0,3}  {1,-24} preset={2,-22} text_freeze={3,-5} image_freeze={4,-5} cache={5} est={6} h" -f `
            $run.order, $run.id, $run.ablation_preset, $run.text_freeze, $run.image_freeze, `
            $(if ($cache) { "yes" } else { "no" }), $run.est_hours
    }
    exit 0
}

# ── telemetry sampler for the whole matrix ──
$monitor = $null
if (-not $NoMonitor) {
    $monitor = Start-Process pwsh -ArgumentList @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $monitorScript,
        "-OutFile", $monitorCsv, "-IntervalSec", "30"
    ) -WindowStyle Hidden -PassThru
    Log "gpu monitor pid=$($monitor.Id) -> $monitorCsv"
}

$summary = @()
try {
    $completed = 0
    foreach ($run in @($runs)) {
        $id = $run.id
        $outDir = Join-Path $BaseOutput $id
        $resultPath = Join-Path $outDir "result.json"
        $runMetaPath = Join-Path $outDir "run_meta.json"

        if ((Test-Path $resultPath) -and -not $Force) {
            $previous = Get-Content $resultPath -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($previous.status -eq "completed") {
                Log "skip $id (already completed: val_acc=$($previous.val_best_accuracy))"
                $summary += $previous
                $completed++
                continue
            }
        }

        $epochs = if ($null -ne $run.epochs) { [int]$run.epochs } else { [int]$protocol.epochs }
        $seed = if ($null -ne $run.seed) { [int]$run.seed } else { [int]$protocol.split.seed }
        $textFreeze = [bool]$run.text_freeze
        $imageFreeze = [bool]$run.image_freeze
        $preset = "$($run.ablation_preset)"
        $wantCache = ($run.vit_cache -eq $true)
        $cacheDir = ""
        if ($wantCache) {
            $cacheDir = if ($env:HMSAN_VIT_CACHE) { $env:HMSAN_VIT_CACHE } else { $defaultCacheDir }
            if (-not (Test-Path $cacheDir)) {
                Log "ABORT $id : requires a ViT cache but $cacheDir does not exist"
                Log "  build it first: $Py scripts\paper\build_vit_cache.py"
                exit 1
            }
        }
        if ((-not $imageFreeze) -and $wantCache) {
            Log "ABORT $id : manifest asks for a cache with image_freeze=false"
            exit 1
        }

        New-Item -ItemType Directory -Force -Path $outDir | Out-Null
        $config = [ordered]@{
            id = $id
            group = "$($run.group)"
            note = "$($run.note)"
            ablation_preset = $preset
            text_freeze = $textFreeze
            image_freeze = $imageFreeze
            vit_cache = $cacheDir
            epochs = $epochs
            seed = $seed
            est_hours = $run.est_hours
            manifest = $Manifest
        }
        [System.IO.File]::WriteAllText(
            (Join-Path $outDir "config.json"),
            ($config | ConvertTo-Json -Depth 4),
            (New-Object System.Text.UTF8Encoding($false))
        )

        Log "start $id  preset=$preset text_freeze=$textFreeze image_freeze=$imageFreeze epochs=$epochs seed=$seed cache=$(if ($cacheDir) { 'yes' } else { 'no' })"
        $startedAt = Get-Date
        $startedIso = $startedAt.ToString("yyyy-MM-ddTHH:mm:ss")
        $attempts = 0
        $code = 1

        while ($attempts -lt $MaxAttempts -and $code -ne 0) {
            $attempts++
            # Empty strings vanish across the `pwsh -File ... @splat`
            # boundary and the following token gets swallowed as their value,
            # so optional parameters are only added when they have a value.
            # The freeze flags travel as explicit 'true'/'false' strings for the
            # same reason: a [bool] would be rendered as -Name:True here.
            $invoke = @{
                Epochs = $epochs
                OutputDir = $outDir
                TextFreeze = $(if ($textFreeze) { "true" } else { "false" })
                ImageFreeze = $(if ($imageFreeze) { "true" } else { "false" })
                Seed = $seed
            }
            if ($preset) { $invoke["AblationPreset"] = $preset }
            if ($cacheDir) { $invoke["VitCacheDir"] = $cacheDir }
            if ($LowPriority) { $invoke["LowPriority"] = $true }
            if ($attempts -gt 1) { Log "  retry $attempts/$MaxAttempts for $id (previous exit=$code)" }
            & pwsh -NoProfile -ExecutionPolicy Bypass -File $runner @invoke
            $code = $LASTEXITCODE
            if ($code -ne 0 -and $attempts -lt $MaxAttempts) {
                Log "  $id exit=$code, retrying in ${RetryDelaySec}s"
                Start-Sleep -Seconds $RetryDelaySec
            }
        }

        $endedAt = Get-Date
        $history = @()
        $historyPath = Join-Path $outDir "history.json"
        if (Test-Path $historyPath) {
            try { $history = @(Get-Content $historyPath -Raw -Encoding UTF8 | ConvertFrom-Json) } catch { $history = @() }
        }
        # The reported pair must come from ONE epoch: the epoch with the highest
        # val accuracy, which is exactly the checkpoint train.py keeps as
        # best_model.pt (protocol section 3).  Taking max(accuracy) and
        # max(macro F1) independently would report two different models as one -
        # the first calibration run showed 0.9694 acc at epoch 1 and 0.8667
        # macro F1 at epoch 3.
        $bestAcc = 0.0; $bestF1 = 0.0; $bestEpoch = 0
        $finalAcc = 0.0; $finalF1 = 0.0; $peakVram = 0.0
        foreach ($entry in $history) {
            $epAcc = [double]$entry.val.accuracy
            $epF1 = [double]$entry.val.macro_f1
            if ($epAcc -gt $bestAcc) {
                $bestAcc = $epAcc; $bestF1 = $epF1; $bestEpoch = [int]$entry.epoch
            }
            $finalAcc = $epAcc
            $finalF1 = $epF1
            if ($entry.peak_vram_mb -gt $peakVram) { $peakVram = [double]$entry.peak_vram_mb }
        }

        $status = "crashed"
        $runMeta = $null
        if (Test-Path $runMetaPath) {
            $runMeta = Get-Content $runMetaPath -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($runMeta.status -eq "completed") { $status = "completed" }
        }
        if ($code -ne 0 -and $status -eq "completed") { $status = "completed-nonzero-exit" }

        $result = [ordered]@{
            id = $id
            group = "$($run.group)"
            note = "$($run.note)"
            status = $status
            exit_code = $code
            attempts = $attempts
            started_at = $startedIso
            ended_at = $endedAt.ToString("yyyy-MM-ddTHH:mm:ss")
            wall_hours = [math]::Round(($endedAt - $startedAt).TotalHours, 3)
            ablation_preset = $preset
            ablation_signature = if ($runMeta) { "$($runMeta.ablation_signature)" } else { "" }
            text_freeze = $textFreeze
            image_freeze = $imageFreeze
            epochs_requested = $epochs
            epochs_completed = if ($runMeta) { $runMeta.epochs_completed } else { @($history).Count }
            seed = $seed
            params_total = if ($runMeta) { $runMeta.params_total } else { $null }
            params_trainable = if ($runMeta) { $runMeta.params_trainable } else { $null }
            vit_cache = $cacheDir
            val_best_accuracy = $bestAcc
            val_best_macro_f1 = $bestF1
            val_best_epoch = $bestEpoch
            val_selection = "epoch with the highest val accuracy (= best_model.pt); macro F1 read from the same epoch"
            val_final_accuracy = $finalAcc
            val_final_macro_f1 = $finalF1
            peak_vram_mb = $peakVram
            est_hours = $run.est_hours
            history = if (@($history).Count -gt 0) { Join-Path $outDir "history.json" } else { "" }
        }
        [System.IO.File]::WriteAllText(
            $resultPath,
            (($result | ConvertTo-Json -Depth 4)),
            (New-Object System.Text.UTF8Encoding($false))
        )
        $summary += $result
        Log "done  $id  status=$status attempts=$attempts wall=$($result.wall_hours)h val_acc=$bestAcc val_f1=$bestF1"

        if ($status -eq "completed" -and $Prune) {
            $prune = Join-Path $repo "scripts\paper\prune_run_outputs.py"
            & $Py $prune --run_dir $outDir 2>&1 | ForEach-Object { Log "  prune: $_" }
        }

        if ($status -eq "completed") { $completed++ } 
        if ($StopAfter -gt 0 -and $completed -ge $StopAfter) {
            Log "StopAfter=$StopAfter reached, stopping"
            break
        }
    }
}
finally {
    if ($monitor) {
        Stop-Process -Id $monitor.Id -Force -ErrorAction SilentlyContinue
        Log "gpu monitor stopped"
    }
}

Log "summary:"
foreach ($item in $summary) {
    "  {0,-24} {1,-22} val_acc={2,-8} val_f1={3,-8} wall={4,7}h" -f `
        $item.id, $item.status, $item.val_best_accuracy, $item.val_best_macro_f1, $item.wall_hours |
        ForEach-Object { Log $_ }
}
$pending = @($runs).Count - $completed
Log "completed=$completed pending=$pending"
if ($pending -gt 0) { exit 1 }
exit 0