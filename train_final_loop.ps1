# HMSAN-BSA 最终训练（崩溃自动重试）
# 机器内存只有 8GB，长时间训练可能被中断；本脚本在中断后自动从 last.pt 续跑。
param(
    [int]$Epochs = 8,
    [int]$MaxAttempts = 30,
    [int]$RetryDelaySec = 60
)

$repo = $PSScriptRoot
$runner = Join-Path $repo "run_train_final.ps1"
$outputDir = Join-Path $repo "outputs\hmsan_bsa_final"
$loopLog = Join-Path $outputDir "loop.log"
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
    "[loop] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  attempt $attempt/$MaxAttempts" |
        Tee-Object -FilePath $loopLog -Append

    & pwsh -NoProfile -ExecutionPolicy Bypass -File $runner -Epochs $Epochs
    $code = $LASTEXITCODE

    if ($code -eq 0) {
        "[loop] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  training finished normally" |
            Tee-Object -FilePath $loopLog -Append
        exit 0
    }
    "[loop] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  exit=$code, retry in ${RetryDelaySec}s" |
        Tee-Object -FilePath $loopLog -Append
    Start-Sleep -Seconds $RetryDelaySec
}
"[loop] attempts exhausted" | Tee-Object -FilePath $loopLog -Append
exit 1
