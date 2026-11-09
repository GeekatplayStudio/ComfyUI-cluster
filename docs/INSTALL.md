# Install Guide

By Geekatplay Studio.

## Requirements

- ComfyUI 0.44 or newer (MOSS-Image, FLUX.3 and MiniMax H3 need it)
- [Ollama](https://ollama.com) running on `localhost:11434`
- One Ollama model. `qwen3.5:9b` (6.6 GB) handles text and images:

```bat
ollama pull qwen3.5:9b
```

Any other Ollama tag works; type it into `ollama_model` on the planner node. The vision planners need a model that can read images.

## Install the nodes

Copy or clone this folder into `ComfyUI/custom_nodes/` and restart ComfyUI. The nodes appear under **Geekatplay Studio**.

## Download models

```bat
install.bat
```

1. Confirm your ComfyUI `models` folder.
2. Optionally set **My GPU has** to hide models that need more VRAM.
3. Tick the models you want. The text encoder and VAE each model needs are ticked with it.
4. Press **Download selected**.

Files download to a `.part` file and are renamed when complete, so a cancelled download leaves nothing broken. Failures are listed in the log and in `install_errors.log`.

The installer is Windows only. On macOS or Linux, download the files listed in `install_manifest.json` into the matching `models/<subdir>/` folders.

### Gated models

FLUX.3 Image Klein needs a Hugging Face token. Accept the licence on its model page, then:

```bat
set HF_TOKEN=your_hf_token
install.bat
```

## Where files go

| Kind | Folder |
|---|---|
| Diffusion models | `models/diffusion_models/` |
| Text encoders | `models/text_encoders/` |
| VAE | `models/vae/` |
| LoRA | `models/loras/` |
| All-in-one checkpoints | `models/checkpoints/` |

## The registry

`model_registry.json` is the single source of truth. `checkpoints` lists the models the planner may choose; `text_encoders`, `vae` and `loras` list supporting files.

Fields of a checkpoint entry:

| Field | Meaning |
|---|---|
| `name` | File name |
| `type` | Family: `moss_image`, `flux3`, `krea2`, `z_image`, `minimax_h3`, or `checkpoint` for an all-in-one file |
| `folder_type` | `diffusion_models` for split models, `checkpoints` for all-in-one files |
| `required_clip`, `required_vae` | Companion files the loader loads with it |
| `clip_type` | Text encoder type, as in ComfyUI's Load CLIP node |
| `steps`, `cfg`, `sampler_name`, `scheduler` | Sampler settings the planner outputs |
| `negative_prompt` | Default negative prompt, used when CFG is above 1 |
| `min_vram` | Estimated GPU memory in GB, used by the `max_vram` filter |
| `tags` | Words the planner matches against the prompt |
| `enabled` | Set `false` to hide a model from the planner |

### Add your own model

Add an entry and rebuild the manifest:

```json
{
  "name": "my_model.safetensors",
  "type": "checkpoint",
  "folder_type": "checkpoints",
  "steps": 30,
  "cfg": 6.5,
  "sampler_name": "dpmpp_2m",
  "scheduler": "karras",
  "min_vram": 8,
  "tags": ["photo", "portrait"],
  "enabled": true
}
```

```bat
python update_manifest.py
```

## Keeping the registry in sync

```bat
manage_models.bat
```

Scans your model folders, fixes paths and sets `enabled` to match what is installed. Command line: `python sync_registry.py --sync`, `--dry-run` or `--reset`.

To check that every download link still works:

```bat
python sync_source_urls.py --check-live --dry-run
```

## Troubleshooting

| Message | Fix |
|---|---|
| "needs companion files that are not installed" | Run `install.bat`; the message lists the files and links. |
| "this ComfyUI has no text encoder type ..." | Update ComfyUI. |
| Summary says "tag matching" | Start Ollama, or pull the model tag shown in the note. |
| Planner picks a model you did not expect | Set `task_hint` to a family, or set `"enabled": false` on models you do not want. |
