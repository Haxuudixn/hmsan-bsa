# 训练结束后的收尾流水线：测试集评测 -> 图表 -> 论文报告
# 用法（在项目根目录）：
#   pwsh -File run_eval_and_report.ps1
# 或在有沙箱限制的会话中由用户手动执行一次。

param(
    [string]$Python = 'C:\hmsan\.venv\Scripts\python.exe',
    [string]$DataDir = 'C:\Users\Administrator\Desktop\训练数据\final_train',
    [string]$OutputDir = '',
    [string]$OutJson = ''
)

$ErrorActionPreference = 'Continue'
$repo = $PSScriptRoot
if (-not $OutputDir) { $OutputDir = Join-Path $repo 'outputs\hmsan_bsa_final' }
if (-not $OutJson) { $OutJson = Join-Path $repo 'docs\paper_materials\eval_test.json' }

if (-not (Test-Path -LiteralPath $Python)) { Write-Error "venv python missing: $Python"; exit 1 }
if (-not (Test-Path -LiteralPath $OutputDir)) { Write-Error "output dir missing: $OutputDir"; exit 1 }

$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$log = Join-Path $OutputDir 'eval.log'
Set-Location $repo

function Step([string]$name, [string]$exe, [string[]]$argv) {
    # 注意：Tee-Object 的输出必须用 Out-Null 丢弃，否则函数会把这些字符串连同
    # 退出码一起返回，调用方拿到的就是数组而不是整数，导致 $code -ne 0 永真。
    "[step] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $name" |
        Tee-Object -FilePath $log -Append | Out-Null
    & $exe @argv 2>&1 | Tee-Object -FilePath $log -Append | Out-Null
    $code = $LASTEXITCODE
    "[step] $name exit=$code" | Tee-Object -FilePath $log -Append | Out-Null
    return $code
}

$fail = 0

$code = Step 'eval_test.py' $Python @(
    'scripts\eval_test.py',
    '--data_dir', $DataDir,
    '--output_dir', $OutputDir,
    '--checkpoints', 'best_model.pt', 'last.pt', 'checkpoint_epoch5.pt',
    '--out_json', $OutJson
)
if ($code -ne 0) { $fail++ }

$code = Step 'make_figures.py' $Python @(
    'scripts\paper\make_figures.py',
    '--stats', 'docs\paper_materials\dataset_stats.json',
    '--history', (Join-Path $OutputDir 'history.json'),
    '--eval', $OutJson,
    '--out_dir', 'docs\paper_materials\figures'
)
if ($code -ne 0) { $fail++ }

$code = Step 'build_report.py' $Python @(
    'scripts\paper\build_report.py',
    '--stats', 'docs\paper_materials\dataset_stats.json',
    '--history', (Join-Path $OutputDir 'history.json'),
    '--eval', $OutJson,
    '--out_dir', 'docs\paper_materials'
)
if ($code -ne 0) { $fail++ }

if ($fail -eq 0) {
    "[done] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  all steps ok" |
        Tee-Object -FilePath $log -Append
} else {
    "[done] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $fail step(s) failed" |
        Tee-Object -FilePath $log -Append
}
exit $fail