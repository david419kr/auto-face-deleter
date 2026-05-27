@echo off
setlocal
cd /d "%~dp0"

call run.bat --white %*
exit /b %errorlevel%
