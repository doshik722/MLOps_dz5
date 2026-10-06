$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$env:HF_HUB_OFFLINE = '1'
$env:PYTHONHASHSEED = '42'
$env:PATH = 'D:\downloads_programs\Git\usr\bin;C:\Users\Дарья\AppData\Local\Microsoft\WinGet\Packages\ezwinports.make_Microsoft.Winget.Source_8wekyb3d8bbwe\bin;' + $env:PATH
New-Item -ItemType Directory -Path 'logs' -Force | Out-Null

function Invoke-Stage {
    param([string]$Name, [string[]]$Arguments)
    Write-Output "$(Get-Date -Format o) START $Name"
    & uv @Arguments 2>&1 | Tee-Object -FilePath "logs/$Name.log"
    if ($LASTEXITCODE -ne 0) { throw "$Name failed: $LASTEXITCODE" }
    Write-Output "$(Get-Date -Format o) DONE $Name"
}

try {
    Invoke-Stage 'prepare_subset' @('run', 'python', '-u', '-m', 'src.prepare_subset')
    Invoke-Stage 'train_all_layers' @('run', 'python', '-u', '-m', 'src.train', '--variant', 'all_layers')
    Invoke-Stage 'train_freeze14' @('run', 'python', '-u', '-m', 'src.train', '--variant', 'freeze14')
    Invoke-Stage 'plot' @('run', 'python', '-m', 'src.plot')
    Invoke-Stage 'compare' @('run', 'python', '-m', 'src.compare', '--variant', 'all_layers')
    Invoke-Stage 'report' @('run', 'python', '-m', 'src.report')
    & git add .gitignore Makefile README.md dvc.yaml params.yaml pyproject.toml uv.lock src scripts tests docs
    if ($LASTEXITCODE -ne 0) { throw 'git add failed' }
    Write-Output "$(Get-Date -Format o) START check"
    & 'D:\downloads_programs\Git\bin\bash.exe' -c 'make check' 2>&1 | Tee-Object -FilePath 'logs/check.log'
    if ($LASTEXITCODE -ne 0) { throw 'make check failed; see logs/check.log' }
    Write-Output "$(Get-Date -Format o) HOMEWORK COMPLETE"
} catch {
    Write-Output "$(Get-Date -Format o) HOMEWORK FAILED: $_"
    exit 1
}
