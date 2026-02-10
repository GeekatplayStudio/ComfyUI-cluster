# Install Guide

## Prerequisites

- ComfyUI installed and running
- Ollama installed and running on `localhost:11434`
- required Ollama models pulled (examples):
  - `qwen2.5:7b`
  - `llava:7b`

## Automated Install

Run from this repo:

```bat
install.bat
```

The installer:
- reads `install_manifest.json`
- downloads model files into your selected model root
- appends an `ollama_cluster` section to ComfyUI `extra_model_paths.yaml` if missing

## Model Path Mapping Added by Installer

`install.bat` now writes these folders under `ollama_cluster`:
- `checkpoints`
- `diffusion_models`
- `unet`
- `loras`
- `text_encoders`
- `clip`
- `controlnet`
- `vae`
- `embeddings`

This supports both standard checkpoint workflows and flux split-component workflows.

## Gated Model Downloads

Some Hugging Face assets may require accepted licenses/authentication.

Before running install, set:

```bat
set HF_TOKEN=your_hf_token
```

Then rerun `install.bat`.

## Manual Validation

After install, verify files are in expected folders:
- `models/checkpoints/` for standard checkpoint workflows
- `models/diffusion_models/`, `models/vae/`, `models/text_encoders/` for flux split workflows
- `models/loras/` for LoRA files
