@echo off
setlocal
cd /d "%~dp0"

where uv >nul 2>nul
if errorlevel 1 (
    echo uv not found. Run install.bat first.
    pause
    exit /b 1
)

set "MODE_ARGS="
set "HAD_INPUT=0"

:loop
if "%~1"=="" goto after_args
if /I "%~1"=="--white" (
    set "MODE_ARGS=%MODE_ARGS% --white"
    shift
    goto loop
)
if /I "%~1"=="-w" (
    set "MODE_ARGS=%MODE_ARGS% --white"
    shift
    goto loop
)
if /I "%~1"=="--lama" (
    set "MODE_ARGS=%MODE_ARGS% --lama"
    shift
    goto loop
)
if /I "%~1"=="-l" (
    set "MODE_ARGS=%MODE_ARGS% --lama"
    shift
    goto loop
)
set "HAD_INPUT=1"
call :process_one "%~1"
if errorlevel 1 goto failed
shift
goto loop

:after_args
if "%HAD_INPUT%"=="0" (
    if not exist input mkdir input
    call :process_one input
    if errorlevel 1 goto failed
)
goto done

:process_one
echo Processing "%~1" to output\ ...
uv run afd process "%~1" --output output --recursive %MODE_ARGS% --save-debug
exit /b %errorlevel%

:failed
echo Processing failed.
pause
exit /b 1

:done
pause
