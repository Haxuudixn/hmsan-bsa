<#
Drive PaddleOCR enrichment across the whole 0827 training-data batch.

Each ZIP is processed independently, written to a .partial file and only moved
into place once the run succeeds, so an interrupted batch never leaves a
truncated ZIP behind.  Per-ZIP JSONL checkpoints let a re-run skip blocks that
were already recognised.
#>
param(
    [string]$SourceDir = 'C:\Users\Administrator\Desktop\训练数据\0827',
    [string]$OutDir    = 'C:\Users\Administrator\Desktop\训练数据\0827_ocr',
    [string]$WorkDir   = 'C:\Users\Administrator\Documents\Codex\2026-08-20\https\work',
    [string]$Repo      = 'C:\Users\Administrator\Documents\Codex\2026-08-20\https\outputs\hmsan-bsa',
    [string]$Python    = 'C:\hmsan\.venv\Scripts\python.exe',
    [string]$Device    = 'gpu:0',
    [string]$Engine    = 'paddle',
    [int]$MaxAttempts  = 3
)

$ErrorActionPreference = 'Continue'
$env:PYTHONUTF8 = '1'

$OutDir  = Join-Path $OutDir ''
$ProgDir = Join-Path $WorkDir 'ocr_progress'
$LogDir  = Join-Path $WorkDir 'ocr_logs'
foreach ($d in @($OutDir, $ProgDir, $LogDir)) {
    if (-not (Test-Path -LiteralPath $d)) { New-Item -ItemType Directory -Force -Path $d | Out-Null }
}

$batchLog = Join-Path $LogDir 'batch.log'
function Write-BatchLog([string]$msg) {
    $line = '{0} {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg
    Write-Host $line
    Add-Content -LiteralPath $batchLog -Value $line -Encoding UTF8
}

$zips = Get-ChildItem -LiteralPath $SourceDir -Filter '*.zip' | Sort-Object Name
Write-BatchLog ("Batch start: {0} ZIP(s) | engine={1} device={2}" -f $zips.Count, $Engine, $Device)

$overall = [Diagnostics.Stopwatch]::StartNew()
$done = 0; $failed = @(); $skipped = 0

foreach ($zip in $zips) {
    $stem      = [IO.Path]::GetFileNameWithoutExtension($zip.Name)
    $finalZip  = Join-Path $OutDir  ($stem + '.zip')
    $partial   = Join-Path $OutDir  ($stem + '.partial.zip')
    $progress  = Join-Path $ProgDir ($stem + '.jsonl')
    $zipLog    = Join-Path $LogDir  ($stem + '.log')

    if (Test-Path -LiteralPath $finalZip) {
        Write-BatchLog ("SKIP (done)  {0}" -f $zip.Name)
        $skipped++
        continue
    }

    Write-BatchLog ("START        {0}  ({1:N1} MB)" -f $zip.Name, ($zip.Length / 1MB))
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $ok = $false

    for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
        if (Test-Path -LiteralPath $partial) { Remove-Item -LiteralPath $partial -Force }
        Add-Content -LiteralPath $zipLog -Encoding UTF8 -Value (
            "`n===== attempt $attempt @ $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') =====")

        & $Python (Join-Path $Repo 'scripts\ocr_preprocess.py') `
            --input $zip.FullName `
            --output $partial `
            --progress $progress `
            --engine $Engine `
            --device $Device *>&1 |
            Tee-Object -FilePath $zipLog -Append |
            Select-String -NotMatch 'ccache|warnings\.warn|Could not find files|Creating model|Model files already|redownload|InitGoogleLogging|gpu_resources|onnxruntime:|CleanUnused|Removing initializer'
        $code = $LASTEXITCODE

        if ($code -eq 0 -and (Test-Path -LiteralPath $partial)) {
            Move-Item -LiteralPath $partial -Destination $finalZip -Force
            $ok = $true
            break
        }
        if ($code -eq 0) {
            Write-BatchLog ("NOTE         {0} had nothing to OCR; marking done" -f $zip.Name)
            New-Item -ItemType File -Force -Path ($finalZip + '.nodata') | Out-Null
            $ok = $true
            break
        }
        Write-BatchLog ("RETRY        {0} attempt {1}/{2} exited {3}" -f $zip.Name, $attempt, $MaxAttempts, $code)
        Start-Sleep -Seconds 20
    }

    $sw.Stop()
    if ($ok) {
        $done++
        $sizeMB = if (Test-Path -LiteralPath $finalZip) { (Get-Item -LiteralPath $finalZip).Length / 1MB } else { 0 }
        Write-BatchLog ("DONE         {0}  in {1:N1} min -> {2:N1} MB" -f $zip.Name, $sw.Elapsed.TotalMinutes, $sizeMB)
    } else {
        $failed += $zip.Name
        Write-BatchLog ("FAILED       {0} after {1} attempts" -f $zip.Name, $MaxAttempts)
    }
}

$overall.Stop()
Write-BatchLog ("Batch end: {0} done, {1} skipped, {2} failed | elapsed {3:N1} min" -f `
    $done, $skipped, $failed.Count, $overall.Elapsed.TotalMinutes)
if ($failed.Count -gt 0) {
    Write-BatchLog ("Failed ZIPs: {0}" -f ($failed -join ', '))
}