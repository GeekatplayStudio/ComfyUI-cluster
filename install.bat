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

:: 1. Update Manifest from Registry
echo Updating Install Manifest from Model Registry...
if exist "%PYTHON_EXE%" (
    "%PYTHON_EXE%" update_manifest.py
) else (
    echo WARNING: Python not found. Manifest update skipped. Using existing manifest.
    python update_manifest.py
)

:: 2. Launch GUI Installer
echo Launching GUI Installer...
powershell -NoProfile -ExecutionPolicy Bypass -File "gui_install.ps1"

pause

if errorlevel 1 (
  echo.
  echo Installer encountered an error. Set HF_TOKEN for gated Hugging Face models if needed.
  exit /b 1
)

REM Update extra_model_paths.yaml
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$yamlPath = Join-Path '%COMFY_ROOT%' 'extra_model_paths.yaml';" ^
  "if(Test-Path $yamlPath){" ^
  "  $content = Get-Content -Raw -Path $yamlPath;" ^
  "  if($content -notmatch '(?m)^ollama_cluster:'){" ^
  "    $base = ('%MODEL_ROOT%' -replace '\\\\','/');" ^
  "    $section = \"`nollama_cluster:`n     base_path: $base`n     is_default: true`n     checkpoints: checkpoints/`n     diffusion_models: diffusion_models/`n     unet: diffusion_models/`n     loras: loras/`n     text_encoders: text_encoders/`n     clip: text_encoders/`n     controlnet: controlnet/`n     vae: vae/`n     embeddings: embeddings/`n\";" ^
  "    Add-Content -Path $yamlPath -Value $section;" ^
  "    Write-Host '[yaml] Added ollama_cluster section to extra_model_paths.yaml';" ^
  "  } else { Write-Host '[yaml] ollama_cluster section already present'; }" ^
  "} else { Write-Host '[yaml] extra_model_paths.yaml not found; skipping'; }"

echo.
echo Script updated. Run install.bat locally to perform downloads.
echo.
exit /b 0
