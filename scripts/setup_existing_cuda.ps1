param(
    [string]$ExistingPython = "C:\software\python\python.exe"
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

if (-not (Test-Path -LiteralPath $ExistingPython)) {
    throw "Existing Python was not found: $ExistingPython"
}

& $ExistingPython -c "import torch; assert torch.cuda.is_available(); assert 'sm_120' in torch.cuda.get_arch_list(); print(torch.__version__, torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw "The selected Python does not have a compatible CUDA PyTorch installation" }

$venvPath = Join-Path $projectRoot ".venv-existing-cuda"
$python = Join-Path $venvPath "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    uv venv --python $ExistingPython --system-site-packages $venvPath
    if ($LASTEXITCODE -ne 0) { throw "Failed to create the shared-site-packages environment" }
}

uv pip install --python $python -e .
if ($LASTEXITCODE -ne 0) { throw "Failed to install project dependencies" }

if (-not (Test-Path -LiteralPath ".env")) {
    Copy-Item -LiteralPath ".env.example" -Destination ".env"
}

& $python scripts/check_experiment.py
