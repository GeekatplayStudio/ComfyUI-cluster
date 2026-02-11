# ComfyUI-Cluster

Ollama-driven routing nodes and curated workflows for ComfyUI.

Maintained by Geekatplay Studio — the console logs and node helpers now surface the Geekatplay branding so you can spot them quickly in shared setups.

This repo is cleaned to a focused setup:
- four production workflows
- checkpoint + LoRA flow
- flux split-component flow (diffusion model + VAE + text encoders)
- image-guided variants using Ollama vision analysis

`model_registry.json` is intentionally kept as the long-term reference registry.

## Included Nodes

Exported custom nodes:
- `OllamaPromptPlanner`
- `OllamaVisionStylePlanner`
- `DynamicCheckpointLoader`
- `DynamicLoraStack`
- `PreviewTextMerge`
- `LiveStatus`

## Workflows

All workflows are in `workflows/`:
- `workflows/ollama_text_checkpoint_lora.json`
- `workflows/ollama_text_flux_split_lora.json`
- `workflows/ollama_image_checkpoint_lora.json`
- `workflows/ollama_image_flux_split_lora.json`

Details are documented in `docs/WORKFLOWS.md`.

## Install

1. Start Ollama and ensure your model is available (for example `qwen2.5:7b` and `llava:7b`).
2. Run:

```bat
install.bat
```

3. The installer will show all checkpoints/VAEs/text encoders/LoRAs from `install_manifest.json`. Pick what you want via the checkbox selector.
   
   **New Features (v2):**
   - **Visual Progress Bar**: Monitor BITS and WebClient downloads in real-time.
   - **Civitai Support**: Automatically resolves Civitai model page URLs to direct file links.
   - **Smart Path Detection**: Checks multiple ComfyUI folders (e.g., `diffusion_models` vs `checkpoints`) to avoid re-downloading existing models.
   - **Robust & Resilient**: 
     - Auto-retries with fallbacks (BITS -> WebClient).
     - Detects and kills stalled downloads (60s timeout).
     - Automatically cleans up 0-byte corrupt files on failure.
   - **Error Logging**: Failed downloads are logged to `install_errors.log` for review.

4. If gated Hugging Face models are required, set `HF_TOKEN` environment variable before running.

## Managing Models

If you manage your models in a custom directory (e.g., external drive), `sync_registry.py` now supports interactive configuration.

```bat
python sync_registry.py --sync
```

- **First Run**: It will ask for your models root path if not found, and save it to `cluster_config.json`.
- **Scanning**: It checks `checkpoints`, `diffusion_models`, `loras`, etc., and updates `model_registry.json` enabled status.
- **Logging**: A detailed scan report is written to `model_scan.log`.

Use `manage_models.bat` (wraps the python script) to quickly enable/disable nodes in the registry based on what you actually have installed.


Included `manage_models.bat` allows you to sync `model_registry.json` with your disk state:
- **Sync**: Disables models in the registry that are missing from disk.
- **Reset**: Re-enables all models in the registry.

Use this if you want the ComfyUI nodes to only offer models you actually have installed.

Installer behavior and model paths are documented in `docs/INSTALL.md`.

## Notes

- Flux split workflows expect:
  - diffusion model in `models/diffusion_models/`
  - VAE in `models/vae/`
  - text encoders in `models/text_encoders/`
- LoRA loading uses `models/loras/`.
- `DynamicCheckpointLoader` fallback now respects per-model `folder_type` from `model_registry.json`.
- Console logging is prefixed with `Geekatplay Studio | OllamaRouter` to make routing messages easy to spot.
