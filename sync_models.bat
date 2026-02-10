@echo off
setlocal

:: Get the directory of this script
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

:: Attempt to locate ComfyUI Embedded Python (Standard Installation)
:: Relative path from custom_nodes/ComfyUI-Cluster/ to python_embeded/
set "PYTHON_EXE=..\..\..\python_embeded\python.exe"

if not exist "%PYTHON_EXE%" (
    echo [INFO] Embedded Python not found at standard relative path.
    echo [INFO] checking ..\..\..\ComfyUI\python_embeded\python.exe
    set "PYTHON_EXE=..\..\..\ComfyUI\python_embeded\python.exe"
)

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Could not find ComfyUI python_embeded directory.
    echo Please make sure you are running this from the custom_nodes folder of a ComfyUI standalone install.
    echo You can try running this manually with your system python if available.
    pause
    exit /b 1
)

echo Using Python: %PYTHON_EXE%
echo Running Model Sync...
"%PYTHON_EXE%" sync_registry.py

echo.
echo Sync Complete.
pause
