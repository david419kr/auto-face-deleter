@echo off
setlocal
cd /d "%~dp0"

call run.bat --white --exclude-mouth %*
exit /b %errorlevel%
