@echo off
setlocal
if defined MLDFI_PYTHON (
  set "PYTHON_EXE=%MLDFI_PYTHON%"
) else if exist "%~dp0..\..\.venv-local\Scripts\python.exe" (
  set "PYTHON_EXE=%~dp0..\..\.venv-local\Scripts\python.exe"
) else (
  set "PYTHON_EXE=python"
)
if "%~1"=="" (
  "%PYTHON_EXE%" "%~dp0..\..\workflow_config.py" run step6_landslide_damming_potential_LDP
) else (
  "%PYTHON_EXE%" "%~dp0step6_landslide_damming_potential_LDP.py" %*
)
exit /b %ERRORLEVEL%
