# HMSAN-BSA 训练启动脚本
# 用法示例:
#   powershell -ExecutionPolicy Bypass -File .\run_train.ps1
#   powershell -ExecutionPolicy Bypass -File .\run_train.ps1 -DataDir "D:\投标文件切片\新建文件夹" -Epochs 20
param(
    [string]$DataDir = "C:\Users\Administrator\Desktop\训练数据",
    [int]$Epochs = 20,
    [string]$Device = "cuda",
    [string]$OutputDir = (Join-Path $PSScriptRoot "outputs\hmsan_bsa_v1")
)

$Py = "C:\hmsan\.venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    Write-Error "未找到虚拟环境: $Py"
    exit 1
}

Set-Location $PSScriptRoot
& $Py train.py --data_dir $DataDir --output_dir $OutputDir --epochs $Epochs --device $Device --label_config "configs/labels.yaml"
exit $LASTEXITCODE

