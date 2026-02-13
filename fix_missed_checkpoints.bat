@echo off
setlocal

:: Get the directory of this script
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

:: Default behavior: validate/fix SOURCE URLs for checkpoints
:: (not local install paths)
set "SYNC_ARGS=--checkpoints-only --check-live"
if not "%~1"=="" (
    set "SYNC_ARGS=%*"
)

:: Attempt to locate ComfyUI Embedded Python (Standard Installation)
set "PYTHON_EXE=..\..\..\python_embeded\python.exe"

if not exist "%PYTHON_EXE%" (
    echo [INFO] Embedded Python not found at standard relative path.
    echo [INFO] checking ..\..\..\ComfyUI\python_embeded\python.exe
    set "PYTHON_EXE=..\..\..\ComfyUI\python_embeded\python.exe"
)

echo Running source URL repair with args: %SYNC_ARGS%
echo.

if exist "%PYTHON_EXE%" (
    "%PYTHON_EXE%" sync_source_urls.py %SYNC_ARGS%
) else (
    echo WARNING: ComfyUI Python not found. Trying global 'python'...
    python sync_source_urls.py %SYNC_ARGS%
)

echo.
echo Finished.
pause
