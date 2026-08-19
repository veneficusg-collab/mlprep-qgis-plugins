$ErrorActionPreference = "Stop"
$workspace = $PSScriptRoot

if ($env:MLDFI_CONFIG_PYTHON) {
    $python = $env:MLDFI_CONFIG_PYTHON
} elseif (Test-Path -LiteralPath (Join-Path $workspace ".venv\Scripts\python.exe")) {
    $python = Join-Path $workspace ".venv\Scripts\python.exe"
} elseif (Test-Path -LiteralPath (Join-Path $workspace ".venv-local\Scripts\python.exe")) {
    $python = Join-Path $workspace ".venv-local\Scripts\python.exe"
} else {
    $python = "python"
}

& $python (Join-Path $workspace "workflow_config.py") @args
exit $LASTEXITCODE
