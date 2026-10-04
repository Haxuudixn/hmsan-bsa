# HMSAN-BSA single training run.
#
# Used by run_comparison.ps1 (one row of the path-3 ablation matrix) and by
# train_final_loop.ps1.
#
# COLD START BY DEFAULT (changed 2026-09-25).  -InitFrom used to default to
# outputs\hmsan_bsa_ft3\best_model.pt.  That checkpoint was trained on an older
# export whose documents overlap the current val/test splits, so warm-starting
# from it leaked eval content into every row; worse, ft3 is a ~35-epoch
# converged model, so each "3-epoch ablation" was really a 3-epoch fine-tune of
# an already-solved task, which hides whether a component is needed at all.
# Warm start is now OPT-IN: pass -InitFrom, and only with a checkpoint that has
# provably never seen an eval document.
#
# Resume policy: if <OutputDir>\last.pt exists the run continues from it.
# Otherwise it cold-starts from the pretrained encoders unless -InitFrom is set.
param(
    [int]$Epochs = 8,
    [int]$MaxBlocksPerSample = 3000,
    [int]$MaxPagesPerSample = 150,
    [string]$DataDir = "C:\Users\Administrator\Desktop\训练数据\final_train",
    [string]$OutputDir = "",
    [string]$InitFrom = "",
    # ── path-3 matrix parameters ──
    # The freeze flags are strings on purpose: this script is reached both from
    # a direct PowerShell command line and from `pwsh -File ... @splat` inside
    # run_comparison.ps1, and PowerShell stringifies booleans across that
    # boundary, so a [bool] parameter would fail before the body ever runs.
    [string]$AblationPreset = "",
    [string]$TextFreeze = "false",
    [string]$ImageFreeze = "true",
    [string]$VitCacheDir = "",
    [int]$Seed = 42,
    # Reported metric is macro F1 over 19 classes, so the objective has to be
    # weighted too: `none` lets the model ignore rare classes (class 11 has 97
    # train blocks out of 310k) and tanks exactly the metric being reported.
    [string]$ClassWeightMode = "inverse",
    # Mid-epoch validation on a step cadence, so the epoch budget can be read
    # off the curve instead of guessed from three epoch endpoints.
    [int]$ValEverySteps = 75,
    # 0 keeps early stopping off: the pilot needs the full curve to be visible.
    [int]$EarlyStopPatience = 0,
    # Step-level crash recovery (F13): write last.pt and flush history.json
    # every N optimizer steps, so a native crash (2026-09-25 23:25 died with
    # exit=-1073741819 at step 85/306) costs at most N steps instead of the
    # whole ~2 h epoch. 0 disables. A mid-epoch resume is approximate: the
    # shuffled batch order is not reproduced, so the skipped prefix is a
    # different draw than the one already trained on.
    [int]$CkptEverySteps = 25,
    [switch]$LowPriority,
    [switch]$PrintOnly
)

function ConvertTo-BoolFlag([string]$value, [string]$name, [bool]$default) {
    if ([string]::IsNullOrWhiteSpace($value)) { return $default }
    switch ($value.Trim().TrimStart('$').ToLowerInvariant()) {
        "true"  { return $true }
        "false" { return $false }
        "1"     { return $true }
        "0"     { return $false }
        "yes"   { return $true }
        "no"    { return $false }
        default {
            Write-Error "-$name must be true/false (got '$value')"
            exit 1
        }
    }
}

$repo = $PSScriptRoot
$Py = "C:\hmsan\.venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { Write-Error "venv missing: $Py"; exit 1 }
if (-not (Test-Path $DataDir)) { Write-Error "data dir missing: $DataDir"; exit 1 }
if (-not $OutputDir) { $OutputDir = Join-Path $repo "outputs\hmsan_bsa_final" }

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$last = Join-Path $OutputDir "last.pt"
$log = Join-Path $OutputDir "train.log"

$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"
$env:OMP_NUM_THREADS = "8"
$env:HF_HUB_DISABLE_TELEMETRY = "1"
# The 2026-09-20 crash loop was CUDA OOM on a 12 GB card with 14.25 GiB reported
# as allocated by PyTorch, i.e. allocator fragmentation on top of a genuinely
# oversized sample. The caps below are the real fix; this only removes the
# fragmentation half of the problem (PyTorch itself suggests this setting).
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$argv = @(
    "train.py",
    "--data_dir", $DataDir,
    "--output_dir", $OutputDir,
    "--label_config", "configs/labels.yaml",
    "--device", "cuda",
    "--epochs", "$Epochs",
    "--encoder_lr", "2e-5",
    "--head_lr", "1e-4",
    "--weight_decay", "1e-4",
    "--grad_clip", "1.0",
    "--scheduler", "plateau",
    "--class_weight_mode", $ClassWeightMode,
    "--val_every_steps", "$ValEverySteps",
    "--early_stop_patience", "$EarlyStopPatience",
    "--ckpt_every_steps", "$CkptEverySteps",
    "--max_blocks_per_sample", "$MaxBlocksPerSample",
    "--max_pages_per_sample", "$MaxPagesPerSample",
    "--train_ratio", "0.7",
    "--val_ratio", "0.15",
    "--seed", "$Seed"
)

# train.py uses BooleanOptionalAction: pass the freeze state explicitly rather
# than relying on its defaults, so a matrix row can never silently inherit the
# default setting.
$textFreezeValue = ConvertTo-BoolFlag $TextFreeze "TextFreeze" $false
$imageFreezeValue = ConvertTo-BoolFlag $ImageFreeze "ImageFreeze" $true
if ($textFreezeValue) { $argv += "--text_freeze" } else { $argv += "--no-text_freeze" }
if ($imageFreezeValue) { $argv += "--image_freeze" } else { $argv += "--no-image_freeze" }

if ($AblationPreset) { $argv += @("--ablation_preset", $AblationPreset) }

if ($VitCacheDir) {
    if (-not (Test-Path $VitCacheDir)) {
        Write-Error "ViT cache missing: $VitCacheDir (run scripts\paper\build_vit_cache.py)"
        exit 1
    }
    $argv += @("--vit_cache_dir", $VitCacheDir)
}

if (Test-Path $last) {
    $argv += @("--resume", $last)
    "[run] resume from $last" | Tee-Object -FilePath $log -Append
} elseif ($InitFrom) {
    $argv += @("--init_from", $InitFrom)
    "[run] WARM START from $InitFrom" | Tee-Object -FilePath $log -Append
} else {
    "[run] COLD START from pretrained encoders (no --init_from)" | Tee-Object -FilePath $log -Append
}

Set-Location $repo
"[run] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  epochs=$Epochs data=$DataDir" |
    Tee-Object -FilePath $log -Append
"[run] preset='$AblationPreset' text_freeze=$textFreezeValue image_freeze=$imageFreezeValue seed=$Seed" |
    Tee-Object -FilePath $log -Append
if ($VitCacheDir) {
    "[run] vit_cache=$VitCacheDir" | Tee-Object -FilePath $log -Append
}

if ($PrintOnly) {
    # Audit the exact command line without launching anything (used by
    # run_comparison.ps1 -DryRun and when documenting the protocol).
    $quoted = $argv | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }
    "[print] $Py $($quoted -join ' ')"
    exit 0
}

if ($LowPriority) {
    # Optional courtesy for a machine that is also being used for work: drop
    # the training process to below-normal CPU priority shortly after launch.
    Start-Job -ScriptBlock {
        Start-Sleep -Seconds 15
        Get-Process -Name python -ErrorAction SilentlyContinue | ForEach-Object {
            try { $_.PriorityClass = 'BelowNormal' } catch { }
        }
    } | Out-Null
}

& $Py @argv 2>&1 | Tee-Object -FilePath $log -Append
$code = $LASTEXITCODE
"[run] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  exit=$code" | Tee-Object -FilePath $log -Append
exit $code
