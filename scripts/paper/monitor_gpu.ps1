# GPU telemetry sampler for long unattended runs (Phase 0, item W10).
#
# The path-3 matrix keeps the GPU busy for ~4 days; if a row is lost to thermal
# throttling or a power cap, the evidence has to exist somewhere.  One CSV row
# per sample, appended forever, so a crash mid-run loses nothing:
#
#     timestamp,total_mb,used_mb,free_mb,util_pct,temp_c,power_w
#
# Started in the background by run_comparison.ps1 and killed when the matrix
# ends; -MaxMinutes is a safety net in case the parent dies.
param(
    [string]$OutFile = "",
    [int]$IntervalSec = 30,
    [int]$MaxMinutes = 4320
)

if (-not $OutFile) {
    $OutFile = Join-Path $PSScriptRoot "..\..\outputs\path3_matrix\monitor.csv"
}
$dir = Split-Path -Parent $OutFile
if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
if (-not (Test-Path $OutFile)) {
    "timestamp,total_mb,used_mb,free_mb,util_pct,temp_c,power_w" |
        Add-Content -Path $OutFile -Encoding utf8
}

$deadline = (Get-Date).AddMinutes($MaxMinutes)
while ((Get-Date) -lt $deadline) {
    $sample = & nvidia-smi --query-gpu=memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits 2>$null
    if ($LASTEXITCODE -eq 0 -and $sample) {
        $line = ($sample | Select-Object -First 1) -replace '\s', ''
        $stamp = Get-Date -Format 'yyyy-MM-ddTHH:mm:ss'
        "$stamp,$line" | Add-Content -Path $OutFile -Encoding utf8
    }
    Start-Sleep -Seconds $IntervalSec
}