@echo off
setlocal
title Geekatplay Studio - ComfyUI-Cluster Installer

:: Run from this script's folder (ComfyUI\custom_nodes\ComfyUI-Cluster)
cd /d "%~dp0"

:: Find Python: ComfyUI portable first, then whatever is on PATH
set "PYTHON_EXE=..\..\..\python_embeded\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=..\..\..\ComfyUI\python_embeded\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"

echo [1/2] Rebuilding the install manifest from model_registry.json...
"%PYTHON_EXE%" update_manifest.py
if errorlevel 1 echo WARNING: could not rebuild the manifest. Using the existing install_manifest.json.

echo [2/2] Opening the model installer...
powershell -NoProfile -ExecutionPolicy Bypass -File "gui_install.ps1"
if errorlevel 1 (
    echo.
    echo The installer reported a problem. See install_errors.log if it exists.
    echo For gated Hugging Face models, accept the licence on the model page and run:
    echo     set HF_TOKEN=your_token
    echo then start install.bat again.
)

echo.
pause
endlocal
