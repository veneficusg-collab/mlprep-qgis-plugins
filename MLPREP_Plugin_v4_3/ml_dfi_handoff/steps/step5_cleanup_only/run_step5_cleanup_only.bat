@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
python "%SCRIPT_DIR%step5_cleanup_only.py" %*
exit /b %ERRORLEVEL%
