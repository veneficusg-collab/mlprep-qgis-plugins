@echo off
setlocal

if "%~1"=="" (
  if defined MLDFI_CONFIG_PYTHON (
    set "CONFIG_PYTHON=%MLDFI_CONFIG_PYTHON%"
  ) else if exist "%~dp0..\..\.venv\Scripts\python.exe" (
    set "CONFIG_PYTHON=%~dp0..\..\.venv\Scripts\python.exe"
  ) else if exist "%~dp0..\..\.venv-local\Scripts\python.exe" (
    set "CONFIG_PYTHON=%~dp0..\..\.venv-local\Scripts\python.exe"
  ) else (
    set "CONFIG_PYTHON=python"
  )
  "%CONFIG_PYTHON%" "%~dp0..\..\workflow_config.py" run step5_post_depositional_spread_PDS
  exit /b %ERRORLEVEL%
)

set "QGIS_PYTHON="
if defined OSGEO4W_ROOT if exist "%OSGEO4W_ROOT%\bin\python-qgis-ltr.bat" (
  set "QGIS_PYTHON=%OSGEO4W_ROOT%\bin\python-qgis-ltr.bat"
)
if not defined QGIS_PYTHON (
  for /f "delims=" %%I in ('where python-qgis-ltr.bat 2^>nul') do if not defined QGIS_PYTHON set "QGIS_PYTHON=%%~fI"
)
if not defined QGIS_PYTHON (
  echo ERROR: QGIS LTR Python launcher was not found.
  echo Set OSGEO4W_ROOT or add python-qgis-ltr.bat to PATH, then retry.
  exit /b 1
)

call "%QGIS_PYTHON%" "%~dp0step5_post_depositional_spread_PDS.py" %*
exit /b %ERRORLEVEL%
