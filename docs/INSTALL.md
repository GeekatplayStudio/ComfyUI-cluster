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
  - **New:** Uses BITS for background downloading; UI remains responsive.
  - **New:** Supports pausing/cancelling mid-download.
- appends an `ollama_cluster` section to ComfyUI `extra_model_paths.yaml` if missing

## Managing Model Registry

To keep your `model_registry.json` in sync with what you have actually downloaded (so the node system knows which models are available), run:

```bat
manage_models.bat
```

This utility offers two modes:
1. **Sync with Disk**: Scans your model folders, enables found models, and disables missing ones in the registry.
2. **Reset Registry**: Re-enables ALL models in the registry (useful if you plan to download more or just want to see everything in the node options).

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
