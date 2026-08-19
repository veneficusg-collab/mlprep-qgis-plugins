@echo off
setlocal
if defined MLDFI_PYTHON (
  set "PYTHON_EXE=%MLDFI_PYTHON%"
) else if exist "%~dp0..\..\.venv\Scripts\python.exe" (
  set "PYTHON_EXE=%~dp0..\..\.venv\Scripts\python.exe"
) else if exist "%~dp0..\..\.venv-local\Scripts\python.exe" (
  set "PYTHON_EXE=%~dp0..\..\.venv-local\Scripts\python.exe"
) else (
  set "PYTHON_EXE=python"
)
if "%~1"=="" (
  "%PYTHON_EXE%" "%~dp0..\..\workflow_config.py" run step3_model_application
) else (
  "%PYTHON_EXE%" "%~dp0apply_ml_dfi_model_to_rasters.py" %*
)
exit /b %ERRORLEVEL%
