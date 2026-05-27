@echo off
setlocal
cd /d "%~dp0"

call run.bat --crop %*
exit /b %errorlevel%
