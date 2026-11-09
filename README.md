# ComfyUI-Cluster

**By Geekatplay Studio.** Ollama-driven model routing for ComfyUI.

Type a prompt (or load a picture). A local Ollama model reads it, picks the best model you have installed, rewrites the prompt for that model, and sets size, steps, CFG, sampler and LoRAs for you. One universal loader then loads the model with the right text encoder and VAE.

## Supported models

Every file name, download link and sampler setting in `model_registry.json` is taken from the official ComfyUI workflow templates.

| Family | Registry type | Good at | VRAM (estimate) |
|---|---|---|---|
| MOSS-Image (SFT, Base) | `moss_image` | text in images, posters, layouts, all-round | 12 GB+ |
| FLUX.3 Image Klein (distilled) | `flux3` | photography, portraits, products | 12 GB+ |
| Krea-2 Turbo | `krea2` | illustration, painterly, stylised | 16 GB+ |
| Z-Image-Turbo, Z-Image Base | `z_image` | fast photorealism, low VRAM | 8 GB+ |
| MiniMax H3, FastVideo FastH3 | `minimax_h3` | video with synchronised sound | 24 GB |

Requires ComfyUI 0.44 or newer. FLUX.3 Klein is a gated download and needs a Hugging Face token.

Any other all-in-one checkpoint file also loads: add an entry with `"type": "checkpoint"` to the registry (see `docs/INSTALL.md`).

## Nodes

All nodes are under **Geekatplay Studio** in the node menu.

| Node | What it does |
|---|---|
| `OllamaPromptPlanner` | Plans from a text prompt. |
| `OllamaVisionStylePlanner` | Looks at an image and plans an image-to-image pass. |
| `OllamaVisionDualPlanner` | Same, with separate "what to look for" and "what to make" prompts. |
| `DynamicCheckpointLoader` | Loads the chosen model plus its text encoder and VAE. Falls back to another installed model of the same family. |
| `DynamicLoraStack` | Applies the chosen LoRAs; skips any that are not installed. |
| `PreviewTextMerge` | Shows several text values in one place. |
| `LiveStatus` | Logs a labelled message to the console. |

How the planner behaves:

- It only offers the language model the models that fit your task, your VRAM and what is installed.
- Sampler settings always come from the registry, never from the language model.
- A model or LoRA name that is not in the registry is rejected, so a made-up name never reaches the loader.
- If Ollama is not running or the model tag is missing, it falls back to tag matching and says why in the `summary` output.
- Changing only the seed does not call Ollama again.

## Workflows

Five ready-to-run workflows are in `workflows/`. Each has numbered groups, notes with setup steps and download links, and a status row showing what the planner and loader did. See `docs/WORKFLOWS.md`.

| File | Purpose |
|---|---|
| `geekatplay_text_to_image_auto.json` | Prompt to image, planner picks the model |
| `geekatplay_text_to_image_moss.json` | Prompt to image, pinned to MOSS-Image |
| `geekatplay_image_to_image_auto.json` | Image plus prompt to image |
| `geekatplay_image_dual_prompt.json` | Image plus vision prompt plus generation prompt |
| `geekatplay_text_to_video_minimax_h3.json` | Prompt to video with sound (MiniMax H3) |

## Quick start

1. Put this folder in `ComfyUI/custom_nodes/` and restart ComfyUI.
2. Install [Ollama](https://ollama.com) and pull a model. `qwen3.5:9b` reads text and images, so it serves every planner:

```bat
ollama pull qwen3.5:9b
```

3. Download models:

```bat
install.bat
```

4. Drag a workflow from `workflows/` into ComfyUI and press Run.

Full details: `docs/INSTALL.md`.

## Maintenance tools

| Command | Purpose |
|---|---|
| `install.bat` | Model installer with VRAM filter, search and progress. |
| `manage_models.bat` | Sync the registry with what is on disk (`sync_registry.py`). |
| `fix_missed_checkpoints.bat` | Check the registry's download links (`sync_source_urls.py`). |
| `python update_manifest.py` | Rebuild `install_manifest.json` from the registry. |
| `python tools/build_workflows.py` | Rebuild and validate the workflows. `--check` validates the files on disk. |

## Changelog

### 2.0
- Models refreshed to MOSS-Image, FLUX.3 Image Klein, Krea-2 Turbo, Z-Image and MiniMax H3; SD1.5, SDXL and other obsolete entries removed.
- Universal loader now handles split models (diffusion model, text encoder, VAE) with the right text encoder type.
- Planners validate every choice against the registry, take sampler settings from it, and fall back to tag matching when Ollama is unavailable.
- Five new workflows with numbered groups, notes and a status row, generated and validated by `tools/build_workflows.py`.
- New installer window with VRAM filter, search, companion-file selection and safe `.part` downloads.
- Nodes now appear under **Geekatplay Studio**. The refiner and ControlNet planner outputs were removed and a `seed` input was added, so workflows saved with the old nodes need relinking.
