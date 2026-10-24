$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

if (-not (Test-Path -LiteralPath ".venv\Scripts\python.exe")) {
    uv venv --python 3.11 .venv
    if ($LASTEXITCODE -ne 0) { throw "Failed to create .venv" }
}
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"

uv pip install --python $python `
    torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 `
    --index-url https://mirrors.nju.edu.cn/pytorch/whl/cu128
if ($LASTEXITCODE -ne 0) { throw "Failed to install CUDA PyTorch" }
uv pip install --python $python -e . `
    --default-index https://mirrors.nju.edu.cn/pypi/web/simple `
    --index https://mirrors.aliyun.com/pypi/simple `
    --index-strategy unsafe-best-match
if ($LASTEXITCODE -ne 0) { throw "Failed to install project dependencies" }

if (-not (Test-Path -LiteralPath ".env")) {
    Copy-Item -LiteralPath ".env.example" -Destination ".env"
    Write-Host "Created .env from .env.example. Fill in the LLM credentials before running."
}

& $python scripts/check_experiment.py
