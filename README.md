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

3. The installer will show all checkpoints/VAEs/text encoders/LoRAs from `install_manifest.json` with descriptions and NSFW flags. Pick what you want via the checkbox selector (Out-GridView) or the console fallback.

4. If gated Hugging Face models are required, set:

```bat
set HF_TOKEN=your_hf_token
```

Installer behavior and model paths are documented in `docs/INSTALL.md`.

## Notes

- Flux split workflows expect:
  - diffusion model in `models/diffusion_models/`
  - VAE in `models/vae/`
  - text encoders in `models/text_encoders/`
- LoRA loading uses `models/loras/`.
- `DynamicCheckpointLoader` fallback now respects per-model `folder_type` from `model_registry.json`.
- Console logging is prefixed with `Geekatplay Studio | OllamaRouter` to make routing messages easy to spot.
