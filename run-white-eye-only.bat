@echo off
setlocal
cd /d "%~dp0"

call run.bat --white --eye-only %*
exit /b %errorlevel%
