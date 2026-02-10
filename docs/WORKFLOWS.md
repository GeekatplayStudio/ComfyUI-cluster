# Workflows

## 1) Text -> Checkpoint + LoRA

File: `workflows/ollama_text_checkpoint_lora.json`

Flow:
- `OllamaPromptPlanner`
- `DynamicCheckpointLoader`
- `DynamicLoraStack`
- `CLIPTextEncode` (positive/negative)
- `KSampler`
- `VAEDecode`
- `SaveImage`

Use when:
- prompt-only generation
- standard SDXL/SD1.5 checkpoints
- automatic LoRA selection from planner output

## 2) Text -> Flux Split + LoRA

File: `workflows/ollama_text_flux_split_lora.json`

Flow differences:
- planner uses `task_hint = flux`
- loader uses `model_type = flux`
- loader receives planner `vae_name` and `clip_name` as overrides

Use when:
- flux diffusion model is separate from VAE and text encoders
- prompt-only generation with optional LoRA stack

## 3) Image + Prompt -> Checkpoint + LoRA

File: `workflows/ollama_image_checkpoint_lora.json`

Flow:
- `LoadImage`
- `OllamaVisionStylePlanner` analyzes image + prompt
- standard checkpoint and LoRA path
- `VAEEncode` for img2img latent
- `KSampler` with planner denoise

Use when:
- image-guided stylization or enhancement with non-flux models

## 4) Image + Prompt -> Flux Split + LoRA

File: `workflows/ollama_image_flux_split_lora.json`

Flow differences:
- vision planner with `task_hint = flux`
- flux split overrides for clip/vae
- img2img latent path through `VAEEncode`

Use when:
- image-guided generation with flux diffusion models and separate clip/vae files

## Debug/Status

Each workflow includes `LiveStatus` nodes to show:
- planner JSON
- selected checkpoint or diffusion model
- loader debug output
- LoRA stack debug output
