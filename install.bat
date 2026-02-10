@echo off
setlocal enabledelayedexpansion

REM Geekatplay Studio - ComfyUI Cluster Installer
REM Interactive selector for checkpoints, VAEs, text encoders, and LoRAs.
REM Shows descriptions + NSFW flag, downloads one-by-one, then re-checks availability.

REM Resolve paths
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\\..") do set "COMFY_ROOT=%%~fI"
set "DEFAULT_MODEL_ROOT=%COMFY_ROOT%\\models"
set "MANIFEST=%SCRIPT_DIR%install_manifest.json"

echo.
echo ComfyUI-Cluster Installer (Geekatplay Studio)
echo --------------------------------------------
echo ComfyUI Root: %COMFY_ROOT%
echo Default Model Root: %DEFAULT_MODEL_ROOT%
echo.

set /p MODEL_ROOT=Enter model root path (press Enter for default): 
if "%MODEL_ROOT%"=="" set "MODEL_ROOT=%DEFAULT_MODEL_ROOT%"

echo.
echo Using model root: %MODEL_ROOT%
echo.

if not exist "%MANIFEST%" (
  echo ERROR: install_manifest.json not found at %MANIFEST%
  exit /b 1
)

REM Interactive PowerShell workflow; not executed here per user request.
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Stop';" ^
  "$manifest = Get-Content -Raw -Path '%MANIFEST%' | ConvertFrom-Json;" ^
  "$root = '%MODEL_ROOT%';" ^
  "function Get-Items { param($downloads,$rootPath); foreach($d in $downloads){ $dir = Join-Path $rootPath $d.subdir; $target = Join-Path $dir $d.filename; $exists = Test-Path $target; $size = if($exists){ (Get-Item $target).Length } else { 0 }; [pscustomobject]@{ Name=$d.filename; Kind=$d.subdir; Description=$d.description; NSFW=[bool]$d.nsfw; Url=$d.url; Target=$target; Present=($exists -and $size -gt 0); SizeMB=if($size -gt 0){ [math]::Round($size/1MB,2) } else { 0 } } } }" ^
  "function Select-Items { param($items); $hasOGV = Get-Command Out-GridView -ErrorAction SilentlyContinue; if($hasOGV){ return $items | Out-GridView -Title 'Select models to download (checkbox). Present=true means already on disk.' -PassThru } else { Write-Host 'Out-GridView not available; falling back to console selection.' -ForegroundColor Yellow; $list = @(); foreach($i in $items){ $label = \"[$($i.Kind)] $($i.Name) :: $($i.Description)\"; if($i.NSFW){ $label += ' [NSFW]' }; $default = if($i.Present){'N'} else {'Y'}; $resp = Read-Host \"$label - download? (Y/N, default=$default)\"; if([string]::IsNullOrWhiteSpace($resp)){ $resp=$default }; if($resp.ToUpper() -eq 'Y'){ $list += $i } }; return $list } }" ^
  "function Download-File { param([string]$url,[string]$outPath); $headers=@{}; if($env:HF_TOKEN){ $headers['Authorization']='Bearer ' + $env:HF_TOKEN }; $outDir = Split-Path -Parent $outPath; if(-not (Test-Path $outDir)){ New-Item -ItemType Directory -Force -Path $outDir | Out-Null }; if(Test-Path $outPath -and (Get-Item $outPath).Length -gt 0){ Write-Host '[skip] already present:' $outPath; return }; Write-Host '[download]' $url '->' $outPath; Invoke-WebRequest -Uri $url -OutFile $outPath -Headers $headers -UseBasicParsing; $fi = Get-Item $outPath; if(-not $fi -or $fi.Length -lt 1024){ throw 'Download appears incomplete: ' + $outPath } }" ^
  "function Check-Status { param($items); $rows = @(); foreach($i in $items){ $exists = Test-Path $i.Target; $size = if($exists){ (Get-Item $i.Target).Length } else { 0 }; $rows += [pscustomobject]@{Name=$i.Name; Target=$i.Target; Present=($exists -and $size -gt 0); SizeMB=if($size -gt 0){ [math]::Round($size/1MB,2) } else { 0 }} }; return $rows }" ^
  "$items = Get-Items -downloads $manifest.downloads -rootPath $root;" ^
  "Write-Host 'Available downloads:'; $items | Select-Object Name,Kind,NSFW,Present,SizeMB,Description | Format-Table;" ^
  "$selected = Select-Items -items $items;" ^
  "if(-not $selected -or $selected.Count -eq 0){ Write-Host 'No items selected. Exiting.'; exit 0 }" ^
  "foreach($item in $selected){ Download-File -url $item.Url -outPath $item.Target }" ^
  "Write-Host ''; Write-Host 'Re-checking model availability...';" ^
  "$status = Check-Status -items $items;" ^
  "$status | Format-Table;" ^
  "if($status | Where-Object { -not $_.Present }){ Write-Warning 'Some items are still missing. Verify URLs and HF_TOKEN.' } else { Write-Host 'All selected items present.' -ForegroundColor Green }"

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
