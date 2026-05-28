@echo off
setlocal
cd /d "%~dp0"

call run.bat --eye-only %*
exit /b %errorlevel%
