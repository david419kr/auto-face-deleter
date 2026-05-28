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

if "%~1"=="" (
    call :interactive_mode
    if errorlevel 1 goto cancelled
    if not exist input mkdir input
    call :process_one input
    if errorlevel 1 goto failed
    goto done
)

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
if /I "%~1"=="--crop" (
    set "MODE_ARGS=%MODE_ARGS% --crop"
    shift
    goto loop
)
if /I "%~1"=="-c" (
    set "MODE_ARGS=%MODE_ARGS% --crop"
    shift
    goto loop
)
if /I "%~1"=="--exclude-mouth" (
    set "MODE_ARGS=%MODE_ARGS% --exclude-mouth"
    shift
    goto loop
)
if /I "%~1"=="-m" (
    set "MODE_ARGS=%MODE_ARGS% --exclude-mouth"
    shift
    goto loop
)
if /I "%~1"=="--eye-only" (
    set "MODE_ARGS=%MODE_ARGS% --eye-only"
    shift
    goto loop
)
if /I "%~1"=="-e" (
    set "MODE_ARGS=%MODE_ARGS% --eye-only"
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
uv run --no-sync afd process "%~1" --output output --recursive %MODE_ARGS%
exit /b %errorlevel%

:interactive_mode
call :select_menu "Select mode" "default|white|crop"
if errorlevel 1 exit /b 1
if /I "%MENU_RESULT%"=="crop" (
    set "MODE_ARGS=--crop"
    exit /b 0
)
if /I "%MENU_RESULT%"=="white" (
    set "MODE_ARGS=--white"
) else (
    set "MODE_ARGS="
)

call :select_menu "Select cleanup" "faceless|exclude mouth|eye only"
if errorlevel 1 exit /b 1
if /I "%MENU_RESULT%"=="exclude mouth" set "MODE_ARGS=%MODE_ARGS% --exclude-mouth"
if /I "%MENU_RESULT%"=="eye only" set "MODE_ARGS=%MODE_ARGS% --eye-only"
exit /b 0

:select_menu
set "MENU_RESULT="
set "AFD_MENU_TITLE=%~1"
set "AFD_MENU_OPTIONS=%~2"
set "AFD_MENU_FILE=%TEMP%\afd_menu_%RANDOM%_%RANDOM%.txt"
if exist "%AFD_MENU_FILE%" del "%AFD_MENU_FILE%" >nul 2>nul
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $title=$env:AFD_MENU_TITLE; $items=$env:AFD_MENU_OPTIONS -split '\|'; $out=$env:AFD_MENU_FILE; $index=0; [Console]::CursorVisible=$false; try { while ($true) { Clear-Host; Write-Host $title; Write-Host ''; Write-Host 'Use Up/Down arrows and Enter. Esc cancels.' -ForegroundColor DarkGray; Write-Host ''; for ($i=0; $i -lt $items.Length; $i++) { if ($i -eq $index) { Write-Host ('> ' + $items[$i]) -ForegroundColor Cyan } else { Write-Host ('  ' + $items[$i]) } }; $key=[Console]::ReadKey($true).Key; if ($key -eq 'UpArrow') { if ($index -le 0) { $index=$items.Length-1 } else { $index-- } } elseif ($key -eq 'DownArrow') { if ($index -ge $items.Length-1) { $index=0 } else { $index++ } } elseif ($key -eq 'Enter') { Set-Content -LiteralPath $out -Value $items[$index] -Encoding ASCII; exit 0 } elseif ($key -eq 'Escape') { exit 2 } } } finally { [Console]::CursorVisible=$true }"
if errorlevel 1 (
    if exist "%AFD_MENU_FILE%" del "%AFD_MENU_FILE%" >nul 2>nul
    exit /b 1
)
if not exist "%AFD_MENU_FILE%" exit /b 1
set /p "MENU_RESULT="<"%AFD_MENU_FILE%"
del "%AFD_MENU_FILE%" >nul 2>nul
exit /b 0

:cancelled
echo Cancelled.
pause
exit /b 1

:failed
echo Processing failed.
pause
exit /b 1

:done
pause
