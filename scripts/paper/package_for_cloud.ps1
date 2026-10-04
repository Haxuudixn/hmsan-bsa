# Build the upload bundle for the rented GPU.
#
# The bundle is a tar.gz of the *working tree*, not a git archive: train.py and
# a dozen other files carry uncommitted F1-F13 work, and .gitignore excludes
# cache/ and *.zip, so a clone would ship the wrong code and no ViT cache.
#
# Usage:
#   pwsh -NoProfile -File scripts\paper\package_for_cloud.ps1
#   pwsh -NoProfile -File scripts\paper\package_for_cloud.ps1 -OutFile D:\bundle.tar.gz
#
# The corpus is deliberately NOT included: it lives outside the repo, it is
# 36.9 GB, and it should go up once into persistent storage rather than riding
# along with every code change.  -IncludeData would put it in the archive for
# a platform that has no other upload path.
param(
    [string]$OutFile = "",
    [string]$DataDir = "C:\Users\Administrator\Desktop\训练数据\final_train",
    [switch]$IncludeData,
    [switch]$NoCache
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if (-not $OutFile) {
    $OutFile = Join-Path (Split-Path $repo -Parent) "hmsan_path3_cloud_bundle.tar.gz"
}

$tar = Join-Path $env:SystemRoot "System32\tar.exe"
if (-not (Test-Path $tar)) { Write-Error "tar.exe not found at $tar"; exit 1 }

Set-Location $repo

# Code and configuration: everything the training path actually imports.
$items = @(
    "train.py", "predict.py", "experiments.py",
    "pyproject.toml", "requirements-cloud.txt",
    "src", "configs", "scripts", "experiments", "tests", "docs"
) | Where-Object { Test-Path $_ }

# The root run_*.ps1 drivers are part of the protocol: run_train_final.ps1
# -PrintOnly is the reference argv, and the E-gate test shells out to it, so a
# bundle without them dies with FileNotFoundError: run_train_final.ps1.
$items += Get-ChildItem -Path $repo -Filter "run_*.ps1" -File |
    ForEach-Object { $_.Name }

if (-not $NoCache) {
    $cacheDir = "cache\google__vit-base-patch16-224-bf16"
    if (Test-Path $cacheDir) { $items += $cacheDir }
    else { Write-Warning "ViT cache not found at $cacheDir - the rented machine will rebuild it (~38 min of GPU)" }
}

if ($IncludeData) {
    if (-not (Test-Path $DataDir)) { Write-Error "data dir missing: $DataDir"; exit 1 }
    $items += $DataDir
    Write-Warning "-IncludeData adds 36.9 GB to the archive; prefer persistent storage"
}

# Windows-only helpers that would only confuse a Linux box.
$exclude = @(
    "--exclude=__pycache__", "--exclude=*.pyc", "--exclude=.pytest_cache",
    "--exclude=.git", "--exclude=.venv", "--exclude=outputs",
    "--exclude=cache_smoke", "--exclude=*.pt", "--exclude=*.tmp",
    "--exclude=annotation_tool", "--exclude=data\processed"
)

Write-Host "repo    $repo"
Write-Host "items   $($items -join ', ')"
Write-Host "out     $OutFile"
Write-Host ""

if (Test-Path $OutFile) { Remove-Item -LiteralPath $OutFile -Force }
& $tar -czf $OutFile @exclude @items
if ($LASTEXITCODE -ne 0) { Write-Error "tar failed with exit $LASTEXITCODE"; exit 1 }

$info = Get-Item -LiteralPath $OutFile
$hash = (Get-FileHash -LiteralPath $OutFile -Algorithm SHA256).Hash
$mb = [math]::Round($info.Length / 1MB, 1)

Write-Host ""
Write-Host "bundle   $($info.FullName)"
Write-Host "size     $mb MiB"
Write-Host "sha256   $hash"
Write-Host ""
Write-Host "contents:"
& $tar -tzf $OutFile | ForEach-Object { "  $_" } | Select-Object -First 40
Write-Host "  ..."
$count = (& $tar -tzf $OutFile | Measure-Object).Count
Write-Host "  $count entries total"

$sidecar = [ordered]@{
    bundle        = $info.Name
    bytes         = $info.Length
    sha256        = $hash
    built_at      = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ss")
    includes_data = [bool]$IncludeData
    includes_cache = (-not $NoCache)
    files         = $count
} | ConvertTo-Json
$sidecarPath = "$OutFile.manifest.json"
[System.IO.File]::WriteAllText($sidecarPath, $sidecar, (New-Object System.Text.UTF8Encoding($false)))
Write-Host ""
Write-Host "sidecar  $sidecarPath"
