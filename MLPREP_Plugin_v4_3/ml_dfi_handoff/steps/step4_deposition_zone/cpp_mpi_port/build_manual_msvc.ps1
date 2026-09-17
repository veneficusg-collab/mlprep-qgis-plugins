param(
    [string]$TaudemSrc = $env:TAUDEM_SRC_DIR,
    [string]$MpiRoot = "$PSScriptRoot\third_party\msmpi",
    [string]$GdalDll = $env:TAUDEM_GDAL_DLL,
    [string]$OutDir = "$PSScriptRoot\build_manual",
    [string]$VcVars64 = $env:VCVARS64
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($TaudemSrc)) {
    $TaudemSrc = Join-Path $PSScriptRoot "third_party\taudem\src"
}
$TaudemSrc = (Resolve-Path -LiteralPath $TaudemSrc).Path
if (!(Test-Path (Join-Path $TaudemSrc "commonLib.cpp"))) {
    throw "TauDEM commonLib.cpp was not found under: $TaudemSrc. Restore third_party\taudem or pass -TaudemSrc."
}
if (!(Test-Path (Join-Path $TaudemSrc "tiffIO.cpp"))) {
    throw "TauDEM tiffIO.cpp was not found under: $TaudemSrc"
}

if ([string]::IsNullOrWhiteSpace($GdalDll)) {
    $gdalCandidates = @(
        (Join-Path $PSScriptRoot "runtime\gdal.dll")
    )
    $GdalDll = $gdalCandidates |
        Where-Object { Test-Path -LiteralPath $_ } |
        Select-Object -First 1
}
if ([string]::IsNullOrWhiteSpace($GdalDll) -or !(Test-Path -LiteralPath $GdalDll)) {
    throw "A TauDEM-compatible gdal.dll is required. Pass -GdalDll or set TAUDEM_GDAL_DLL."
}
$GdalDll = (Resolve-Path -LiteralPath $GdalDll).Path

if ([string]::IsNullOrWhiteSpace($VcVars64)) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path -LiteralPath $vswhere) {
        $vsRoot = & $vswhere -latest -products * `
            -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
            -property installationPath
        if ($vsRoot) {
            $VcVars64 = Join-Path $vsRoot "VC\Auxiliary\Build\vcvars64.bat"
        }
    }
}
if ([string]::IsNullOrWhiteSpace($VcVars64) -or !(Test-Path -LiteralPath $VcVars64)) {
    throw "vcvars64.bat was not found. Pass -VcVars64 or set VCVARS64."
}
if (!(Test-Path (Join-Path $MpiRoot "Include\mpi.h"))) {
    throw "MPI header not found. Expected: $(Join-Path $MpiRoot 'Include\mpi.h')"
}
if (!(Test-Path (Join-Path $MpiRoot "Lib\x64\msmpi.lib"))) {
    throw "MPI import library not found. Expected: $(Join-Path $MpiRoot 'Lib\x64\msmpi.lib')"
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$defPath = Join-Path $OutDir "gdal.def"
$libPath = Join-Path $OutDir "gdal.lib"
$exePath = Join-Path $OutDir "step4_deposition_zone_mpi.exe"
$candidateExe = Join-Path $OutDir "step4_deposition_zone_mpi.candidate.exe"
$exports = @(
    "GDALAllRegister",
    "GDALOpen",
    "GDALClose",
    "GDALGetDatasetDriver",
    "GDALGetDriverByName",
    "GDALCreate",
    "GDALGetRasterBand",
    "GDALGetRasterXSize",
    "GDALGetRasterYSize",
    "GDALGetProjectionRef",
    "GDALSetProjection",
    "GDALGetGeoTransform",
    "GDALSetGeoTransform",
    "GDALGetRasterNoDataValue",
    "GDALSetRasterNoDataValue",
    "GDALGetRasterUnitType",
    "GDALGetRasterDataType",
    "GDALRasterIO",
    "GDALFlushCache",
    "CSLSetNameValue",
    "OSRNewSpatialReference",
    "OSRIsGeographic",
    "OSRGetLinearUnits",
    "OGR_DS_GetLayerCount",
    "OGR_DS_GetLayer",
    "OGR_L_GetName",
    "OGR_L_GetGeomType"
)
"LIBRARY gdal.dll`nEXPORTS`n$($exports -join "`n")" | Set-Content -Path $defPath -Encoding ASCII

$cmd = @"
call "$VcVars64"
if errorlevel 1 exit /b %errorlevel%
lib /nologo /machine:x64 /def:"$defPath" /out:"$libPath"
if errorlevel 1 exit /b %errorlevel%
cl /nologo /EHsc /O2 /std:c++17 /D_CRT_SECURE_NO_WARNINGS ^
  /I"$PSScriptRoot\gdal_compat" ^
  /I"$TaudemSrc" ^
  /I"$MpiRoot\Include" ^
  /FI"$PSScriptRoot\gdal_compat\taudem_compat.h" ^
  "$PSScriptRoot\step4_deposition_zone_mpi.cpp" ^
  "$TaudemSrc\commonLib.cpp" ^
  "$TaudemSrc\tiffIO.cpp" ^
  /Fe:"$candidateExe" ^
  /link /MANIFEST:NO /LIBPATH:"$MpiRoot\Lib\x64" /LIBPATH:"$OutDir" msmpi.lib gdal.lib
exit /b %errorlevel%
"@

$cmdPath = Join-Path $env:TEMP ("step4-build-" + [guid]::NewGuid().ToString("N") + ".cmd")
$cmd | Set-Content -LiteralPath $cmdPath -Encoding ASCII
Push-Location $OutDir
try {
    Remove-Item -LiteralPath $candidateExe -Force -ErrorAction SilentlyContinue
    & $env:ComSpec /d /s /c "call `"$cmdPath`""
    if ($LASTEXITCODE -ne 0) {
        throw "Step 4 C++/MPI compilation failed with exit code $LASTEXITCODE."
    }
} finally {
    Pop-Location
    Remove-Item -LiteralPath $cmdPath -Force -ErrorAction SilentlyContinue
}

if (!(Test-Path $candidateExe)) {
    throw "Build command finished but candidate executable was not created: $candidateExe"
}
Move-Item -LiteralPath $candidateExe -Destination $exePath -Force

@(
    $defPath,
    $libPath,
    (Join-Path $OutDir "gdal.exp"),
    (Join-Path $OutDir "commonLib.obj"),
    (Join-Path $OutDir "tiffIO.obj"),
    (Join-Path $OutDir "step4_deposition_zone_mpi.obj")
) | ForEach-Object {
    Remove-Item -LiteralPath $_ -Force -ErrorAction SilentlyContinue
}

Write-Host "Built: $exePath"
Write-Host "Runtime note: include this on PATH before running: $(Split-Path $GdalDll -Parent)"
