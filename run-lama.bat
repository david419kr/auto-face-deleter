@echo off
setlocal
cd /d "%~dp0"

call run.bat --lama %*
exit /b %errorlevel%
