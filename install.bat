@echo off
setlocal enabledelayedexpansion

REM Resolve paths
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\\..") do set "COMFY_ROOT=%%~fI"
set "DEFAULT_MODEL_ROOT=%COMFY_ROOT%\\models"
set "MANIFEST=%SCRIPT_DIR%install_manifest.json"

echo.
echo ComfyUI-Cluster Installer
echo -------------------------
echo ComfyUI Root: %COMFY_ROOT%
echo Default Model Root: %DEFAULT_MODEL_ROOT%
echo.

set /p MODEL_ROOT=Enter model root path (press Enter to use default): 
if "%MODEL_ROOT%"=="" set "MODEL_ROOT=%DEFAULT_MODEL_ROOT%"

echo.
echo Using model root: %MODEL_ROOT%
echo.

if not exist "%MANIFEST%" (
  echo ERROR: install_manifest.json not found at %MANIFEST%
  exit /b 1
)

REM Download models using PowerShell
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Stop';" ^
  "$manifest = Get-Content -Raw -Path '%MANIFEST%' | ConvertFrom-Json;" ^
  "$root = '%MODEL_ROOT%';" ^
  "function Download-File([string]$url,[string]$out){ if(Test-Path $out){ Write-Host '[skip]' $out; return } $headers=@{}; if($env:HF_TOKEN){ $headers['Authorization']='Bearer ' + $env:HF_TOKEN } Invoke-WebRequest -Uri $url -OutFile $out -Headers $headers }" ^
  "foreach($item in $manifest.downloads){ $dir = Join-Path $root $item.subdir; New-Item -ItemType Directory -Force -Path $dir | Out-Null; $out = Join-Path $dir $item.filename; Write-Host '[download]' $item.url '->' $out; Download-File $item.url $out }"

if errorlevel 1 (
  echo.
  echo Download step failed. If models require a license, set HF_TOKEN and retry.
  echo Example: set HF_TOKEN=your_hf_token
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
echo Done. Restart ComfyUI to load new models and custom nodes.
echo.
exit /b 0
