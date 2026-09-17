param(
    [ValidateSet("lock", "minimum")]
    [string]$DependencySet = "lock",
    [string]$PythonExecutable = "",
    [switch]$ForceReinstall
)

$ErrorActionPreference = "Stop"
$installer = Join-Path $PSScriptRoot "setup_environment.py"
$installerArgs = @("--dependency-set", $DependencySet)
if (![string]::IsNullOrWhiteSpace($PythonExecutable)) {
    $installerArgs += @("--python-executable", $PythonExecutable)
}
if ($ForceReinstall) {
    $installerArgs += "--force-reinstall"
}

if (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3.13 $installer @installerArgs
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    & python $installer @installerArgs
} else {
    throw "Python was not found. Install 64-bit Python 3.13.12."
}
if ($LASTEXITCODE -ne 0) {
    throw "Environment setup failed with exit code $LASTEXITCODE."
}
