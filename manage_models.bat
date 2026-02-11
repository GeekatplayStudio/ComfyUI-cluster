@echo off
setlocal

:: Get the directory of this script
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

:: Attempt to locate ComfyUI Embedded Python (Standard Installation)
set "PYTHON_EXE=..\..\..\python_embeded\python.exe"

if not exist "%PYTHON_EXE%" (
    echo [INFO] Embedded Python not found at standard relative path.
    echo [INFO] checking ..\..\..\ComfyUI\python_embeded\python.exe
    set "PYTHON_EXE=..\..\..\ComfyUI\python_embeded\python.exe"
)

if exist "%PYTHON_EXE%" (
    "%PYTHON_EXE%" sync_registry.py
) else (
    echo WARNING: ComfyUI Python not found. Trying global 'python'...
    python sync_registry.py
)

pause
