param(
    [Parameter(Mandatory = $true)]
    [string]$Step,

    [Parameter(Mandatory = $true)]
    [string]$LogPath,

    [Parameter(Mandatory = $true)]
    [string]$ExitCodePath,

    [switch]$Overwrite,

    [string[]]$ExtraArguments = @()
)

$ErrorActionPreference = "Stop"
$workspaceRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $workspaceRoot ".venv-local\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Workspace Python was not found: $python"
}

$logParent = Split-Path -Parent $LogPath
$exitParent = Split-Path -Parent $ExitCodePath
New-Item -ItemType Directory -Path $logParent -Force | Out-Null
New-Item -ItemType Directory -Path $exitParent -Force | Out-Null

# TauDEM's legacy GIS variables are scoped by the Step 4 launcher. They must
# not leak into rasterio, QGIS, or other Python stages.
Remove-Item Env:PROJ_LIB -ErrorAction SilentlyContinue
Remove-Item Env:PROJ_DATA -ErrorAction SilentlyContinue
Remove-Item Env:GDAL_DATA -ErrorAction SilentlyContinue
Remove-Item Env:GDAL_DRIVER_PATH -ErrorAction SilentlyContinue

Set-Location -LiteralPath $workspaceRoot
Write-Host "ML-DFI official run: $Step"
Write-Host "Workspace: $workspaceRoot"
Write-Host "Log: $LogPath"

$exitCode = 1
try {
    # Native GIS tools may write progress messages to stderr even on success.
    # Keep those visible and logged without converting them into terminating
    # PowerShell errors.
    $ErrorActionPreference = "Continue"
    $arguments = @(
        (Join-Path $workspaceRoot "workflow_config.py"),
        "run",
        $Step
    )
    if ($ExtraArguments.Count -gt 0) {
        $arguments += "--"
        $arguments += $ExtraArguments
    }
    if ($Overwrite) {
        if ($ExtraArguments.Count -eq 0) {
            $arguments += "--"
        }
        $arguments += "--overwrite"
    }
    & $python @arguments 2>&1 |
        Tee-Object -FilePath $LogPath
    if ($null -ne $LASTEXITCODE) {
        $exitCode = [int]$LASTEXITCODE
    }
}
catch {
    $_ | Out-String | Tee-Object -FilePath $LogPath -Append | Write-Host
    $exitCode = 1
}
finally {
    Set-Content -LiteralPath $ExitCodePath -Value $exitCode -Encoding ascii
}

Write-Host "Completed $Step with exit code $exitCode"
exit $exitCode
