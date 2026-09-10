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
  "%PYTHON_EXE%" "%~dp0..\..\workflow_config.py" run step2_extract_pixel_samples
) else (
  "%PYTHON_EXE%" "%~dp0step2_extract_pixel_samples.py" %*
)
exit /b %ERRORLEVEL%
