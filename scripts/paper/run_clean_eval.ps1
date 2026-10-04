<#
    A' - re-score the surviving matrix checkpoints on the leak-free eval sets.

    Waits for the current matrix batch to finish (train.py gone + the gated row
    reports status=completed), then runs scripts/paper/leak_diag_eval.py over

      pass 1  every matrix row's best_model.pt and last.pt
      pass 2  the delivered checkpoint (separate invocation: a preset-signature
              mismatch there must not abort the matrix pass)

    Nothing is trained and no run directory is written to; the only outputs are
    outputs/path3_diag/leak_diag_clean.json and the log next to it.
#>
param(
    [string]$Gate = 'A9_local_window',
    [int]$PollSeconds = 60,
    [int]$SettleSeconds = 120,
    [string]$Out = 'outputs\path3_diag\leak_diag_clean.json'
)

$ErrorActionPreference = 'Continue'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repo

$python = 'C:\hmsan\.venv\Scripts\python.exe'
$script = 'scripts\paper\leak_diag_eval.py'
$logDir = 'outputs\path3_diag'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir 'run_clean_eval.log'

function Write-Log([string]$message) {
    $line = '[clean-eval] {0}  {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $message
    Add-Content -Path $log -Value $line -Encoding utf8
}

$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS = '8'

Write-Log "waiting for $Gate to report status=completed and for train.py to exit"

while ($true) {
    $resultPath = "outputs\path3_matrix\$Gate\result.json"
    $done = $false
    if (Test-Path $resultPath) {
        try {
            $result = Get-Content $resultPath -Raw -Encoding UTF8 | ConvertFrom-Json
            $done = ($result.status -eq 'completed')
        } catch {
            $done = $false
        }
    }
    $training = @(Get-CimInstance Win32_Process -Filter "Name like '%python%'" |
        Where-Object { $_.CommandLine -like '*train.py*' })
    if ($done -and $training.Count -eq 0) { break }
    Start-Sleep -Seconds $PollSeconds
}

Write-Log "gate satisfied; settling for $SettleSeconds s so the host frees memory"
Start-Sleep -Seconds $SettleSeconds

$freeMb = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1KB)
Write-Log "free physical memory: $freeMb MB"

Write-Log 'pass 1/2: every matrix row (best_model.pt + last.pt)'
& $python $script --out $Out *>&1 | Out-File -FilePath $log -Append -Encoding utf8
Write-Log "pass 1 exit=$LASTEXITCODE"

Write-Log 'pass 2/2: delivered checkpoint'
& $python $script --out $Out --checkpoint delivered_last8 *>&1 | Out-File -FilePath $log -Append -Encoding utf8
Write-Log "pass 2 exit=$LASTEXITCODE"

Write-Log "DONE  -> $Out"