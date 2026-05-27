@echo off
setlocal
cd /d "%~dp0"

where uv >nul 2>nul
if errorlevel 1 (
    echo uv not found. Run install.bat first.
    pause
    exit /b 1
)

if "%~1"=="" (
    if not exist input mkdir input
    echo Processing input\ to output\ ...
    uv run afd process input --output output --recursive --backend hybrid --device cuda --conf 0.25 --aggression normal --save-debug
    goto done
)

:loop
if "%~1"=="" goto done
echo Processing "%~1" to output\ ...
uv run afd process "%~1" --output output --recursive --backend hybrid --device cuda --conf 0.25 --aggression normal --save-debug
if errorlevel 1 goto failed
shift
goto loop

:failed
echo Processing failed.
pause
exit /b 1

:done
pause
