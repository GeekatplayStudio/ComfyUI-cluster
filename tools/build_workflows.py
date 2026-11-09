"""
ComfyUI-Cluster by Geekatplay Studio - workflow builder.

Generates every file in workflows/ from one description, so layout, comments,
branding and wiring stay consistent, and checks the result before writing:

    python tools/build_workflows.py            build and validate
    python tools/build_workflows.py --check    validate the files already on disk

Checks: every link joins matching types, every required input is connected,
widget values match each node's widget list, nodes do not overlap, every node
sits inside its group, and titles and note text fit inside their boxes.

Coordinates are plain LiteGraph units, which is what both the classic renderer
and Nodes 2.0 expect in a saved workflow.
"""

import json
import os
import sys
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT_DIR = os.path.join(REPO, "workflows")

BRAND = "Geekatplay Studio"
PACK_ID = "GeekatplayStudio/ComfyUI-cluster"
FRONTEND_VERSION = "1.58.5"

# Row heights used to size nodes. They follow the taller Nodes 2.0 rows so a
# node never has to grow past its saved size and push into its neighbour.
TITLE_H = 34
SLOT_H = 24
WIDGET_H = 34
PAD = 20

# colour presets from the ComfyUI palette: (title bar, body)
COLORS = {
    "green": ("#232", "#353"),
    "red": ("#322", "#533"),
    "blue": ("#223", "#335"),
    "pale_blue": ("#2a363b", "#3f5159"),
    "purple": ("#323", "#535"),
    "yellow": ("#432", "#653"),
    "cyan": ("#233", "#355"),
    "black": ("#222", "#000"),
}
GROUP_COLORS = {
    "plan": "#a1309b",
    "load": "#3f789e",
    "prompt": "#8A8",
    "sample": "#b58b2a",
    "output": "#8AA",
    "status": "#444",
}

PLANNER_OUTPUTS = [
    ("checkpoint", "STRING"), ("loras", "STRING"), ("lora_strengths", "STRING"), ("model_type", "STRING"),
    ("steps", "INT"), ("cfg", "FLOAT"), ("sampler_name", "COMBO"), ("scheduler", "COMBO"),
    ("width", "INT"), ("height", "INT"), ("seed", "INT"),
    ("positive_prompt", "STRING"), ("negative_prompt", "STRING"), ("plan_json", "STRING"),
    ("vae_name", "STRING"), ("clip_name", "STRING"), ("task", "STRING"), ("denoise", "FLOAT"),
    ("summary", "STRING"),
]
PLANNER_COMMON = [
    ("ollama_model", "STRING", "qwen3.5:9b"),
    ("registry_path", "STRING", "model_registry.json"),
    ("task_hint", "COMBO", "auto"),
    ("user_negative", "TEXT", ""),
    ("aspect_ratio", "COMBO", "1:1"),
    ("base_size", "INT", 1024),
    ("ollama_host", "STRING", "localhost"),
    ("ollama_port", "INT", 11434),
    ("max_vram", "COMBO", 24),
    ("seed", "INT", 0),
    (None, "CONTROL", "randomize"),
]

# Node definitions: sockets = inputs without a widget, widgets = (name, type, default).
# A widget named None is the "control after generate" selector that follows a seed.
# "TEXT" marks a multiline text box. "optional" lists sockets that may stay unconnected.
SPECS = {
    # --- ComfyUI-Cluster nodes ------------------------------------------------
    "OllamaPromptPlanner": {
        "sockets": [], "widgets": [("prompt", "TEXT", "")] + PLANNER_COMMON, "outputs": PLANNER_OUTPUTS, "custom": True,
    },
    "OllamaVisionStylePlanner": {
        "sockets": [("image", "IMAGE")], "widgets": [("prompt", "TEXT", "")] + PLANNER_COMMON,
        "outputs": PLANNER_OUTPUTS, "custom": True,
    },
    "OllamaVisionDualPlanner": {
        "sockets": [("image", "IMAGE")],
        "widgets": [("vision_prompt", "TEXT", ""), ("generation_prompt", "TEXT", "")] + PLANNER_COMMON,
        "outputs": PLANNER_OUTPUTS, "custom": True,
    },
    "DynamicCheckpointLoader": {
        "sockets": [],
        "widgets": [("checkpoint", "STRING", ""), ("model_type", "COMBO", "auto"),
                    ("registry_path", "STRING", "model_registry.json"), ("custom_path", "STRING", ""),
                    ("vae_override", "STRING", ""), ("clip_override", "STRING", "")],
        "outputs": [("MODEL", "MODEL"), ("CLIP", "CLIP"), ("VAE", "VAE"), ("debug_info", "STRING")], "custom": True,
    },
    "DynamicLoraStack": {
        "sockets": [("model", "MODEL"), ("clip", "CLIP")],
        "widgets": [("loras", "STRING", ""), ("lora_strengths", "STRING", ""), ("custom_path", "STRING", "")],
        "outputs": [("MODEL", "MODEL"), ("CLIP", "CLIP"), ("debug_info", "STRING")], "custom": True,
    },
    # --- ComfyUI core nodes ---------------------------------------------------
    "CLIPTextEncode": {
        "sockets": [("clip", "CLIP")], "widgets": [("text", "TEXT", "")], "outputs": [("CONDITIONING", "CONDITIONING")],
    },
    "EmptySD3LatentImage": {
        "sockets": [], "widgets": [("width", "INT", 1024), ("height", "INT", 1024), ("batch_size", "INT", 1)],
        "outputs": [("LATENT", "LATENT")],
    },
    "EmptyFlux2LatentImage": {
        "sockets": [], "widgets": [("width", "INT", 1024), ("height", "INT", 1024), ("batch_size", "INT", 1)],
        "outputs": [("LATENT", "LATENT")],
    },
    "KSampler": {
        "sockets": [("model", "MODEL"), ("positive", "CONDITIONING"), ("negative", "CONDITIONING"), ("latent_image", "LATENT")],
        "widgets": [("seed", "INT", 0), (None, "CONTROL", "fixed"), ("steps", "INT", 20), ("cfg", "FLOAT", 1.0),
                    ("sampler_name", "COMBO", "euler"), ("scheduler", "COMBO", "simple"), ("denoise", "FLOAT", 1.0)],
        "outputs": [("LATENT", "LATENT")],
    },
    "SkipLayerGuidanceDiTSimple": {
        "sockets": [("model", "MODEL")],
        "widgets": [("double_layers", "STRING", ""), ("single_layers", "STRING", "22"),
                    ("start_percent", "FLOAT", 0), ("end_percent", "FLOAT", 0.5)],
        "outputs": [("MODEL", "MODEL")],
    },
    "VAEDecode": {"sockets": [("samples", "LATENT"), ("vae", "VAE")], "widgets": [], "outputs": [("IMAGE", "IMAGE")]},
    "VAEEncode": {"sockets": [("pixels", "IMAGE"), ("vae", "VAE")], "widgets": [], "outputs": [("LATENT", "LATENT")]},
    "LoadImage": {
        "sockets": [], "widgets": [("image", "COMBO", "example.png"), ("upload", "BUTTON", "image")],
        "outputs": [("IMAGE", "IMAGE"), ("MASK", "MASK")],
    },
    "ImageScaleToTotalPixels": {
        "sockets": [("image", "IMAGE")],
        "widgets": [("upscale_method", "COMBO", "lanczos"), ("megapixels", "FLOAT", 1.0), ("resolution_steps", "INT", 16)],
        "outputs": [("IMAGE", "IMAGE")],
    },
    "SaveImage": {
        "sockets": [("images", "IMAGE")], "widgets": [("filename_prefix", "STRING", "ComfyUI")],
        "outputs": [("images", "IMAGE")],
    },
    "PreviewAny": {"sockets": [("source", "*")], "widgets": [], "outputs": [("STRING", "STRING")]},
    "VAELoader": {"sockets": [], "widgets": [("vae_name", "COMBO", "")], "outputs": [("VAE", "VAE")]},
    "MiniMaxH3ImageToVideo": {
        "sockets": [("clip", "CLIP"), ("vae", "VAE"), ("first_frame", "IMAGE"), ("last_frame", "IMAGE")],
        "optional": ["first_frame", "last_frame"],
        "widgets": [("prompt", "TEXT", ""), ("width", "INT", 1344), ("height", "INT", 768), ("length", "INT", 124)],
        "outputs": [("positive", "CONDITIONING"), ("LATENT", "LATENT")],
    },
    "BasicGuider": {
        "sockets": [("model", "MODEL"), ("conditioning", "CONDITIONING")], "widgets": [], "outputs": [("GUIDER", "GUIDER")],
    },
    "BasicScheduler": {
        "sockets": [("model", "MODEL")],
        "widgets": [("scheduler", "COMBO", "simple"), ("steps", "INT", 20), ("denoise", "FLOAT", 1.0)],
        "outputs": [("SIGMAS", "SIGMAS")],
    },
    "KSamplerSelect": {"sockets": [], "widgets": [("sampler_name", "COMBO", "res_multistep")], "outputs": [("SAMPLER", "SAMPLER")]},
    "RandomNoise": {
        "sockets": [], "widgets": [("noise_seed", "INT", 0), (None, "CONTROL", "fixed")], "outputs": [("NOISE", "NOISE")],
    },
    "SamplerCustomAdvanced": {
        "sockets": [("noise", "NOISE"), ("guider", "GUIDER"), ("sampler", "SAMPLER"), ("sigmas", "SIGMAS"), ("latent_image", "LATENT")],
        "widgets": [], "outputs": [("output", "LATENT"), ("denoised_output", "LATENT")],
    },
    "VAEDecodeAudio": {"sockets": [("samples", "LATENT"), ("vae", "VAE")], "widgets": [], "outputs": [("AUDIO", "AUDIO")]},
    "CreateVideo": {
        "sockets": [("images", "IMAGE"), ("audio", "AUDIO")], "optional": ["audio"],
        "widgets": [("fps", "FLOAT", 24), ("bit_depth", "COMBO", "auto"), ("color_space", "COMBO", "sRGB"), ("codec", "COMBO", "none")],
        "outputs": [("VIDEO", "VIDEO")],
    },
    "SaveVideo": {
        "sockets": [("video", "VIDEO")],
        "widgets": [("filename_prefix", "STRING", "video/ComfyUI"), ("format", "COMBO", "auto"),
                    ("format.codec", "COMBO", "auto"), ("codec", "COMBO", "auto")],
        "outputs": [("video", "VIDEO")],
    },
    "MarkdownNote": {"sockets": [], "widgets": [("text", "TEXT", "")], "outputs": [], "virtual": True},
}

# Types a widget input may receive over a link.
WIDGET_LINK_TYPE = {"TEXT": "STRING", "STRING": "STRING", "INT": "INT", "FLOAT": "FLOAT", "COMBO": "COMBO"}


def min_height(node_type, text_heights=None):
    """Smallest height that shows every slot and widget of a node without clipping."""
    spec = SPECS[node_type]
    rows = max(len(spec["sockets"]), len(spec["outputs"]))
    height = rows * SLOT_H + PAD
    texts = list(text_heights or [])
    for _, wtype, _ in spec["widgets"]:
        if wtype == "TEXT":
            height += texts.pop(0) if texts else 110
        else:
            height += WIDGET_H
    return height


class Graph:
    def __init__(self, name):
        self.name = name
        self.nodes = []
        self.links = []
        self.groups = []
        self._node_id = 0
        self._link_id = 0

    # -- nodes ----------------------------------------------------------------
    def node(self, node_type, pos, width, title=None, color=None, values=None, text_heights=None, height=None, extra_height=0):
        spec = SPECS[node_type]
        self._node_id += 1
        h = height or (min_height(node_type, text_heights) + extra_height)
        widget_values = []
        values = dict(values or {})
        for wname, _, default in spec["widgets"]:
            widget_values.append(values.pop(wname, default) if wname else values.pop("_control", default))
        if values:
            raise ValueError(f"{node_type}: unknown widget(s) {sorted(values)}")

        inputs = [{"name": n, "type": t, "link": None} for n, t in spec["sockets"]]
        for opt in spec.get("optional", []):
            next(i for i in inputs if i["name"] == opt)["shape"] = 7
        for wname, wtype, _ in spec["widgets"]:
            if wname and wtype in WIDGET_LINK_TYPE:
                inputs.append({"name": wname, "type": WIDGET_LINK_TYPE[wtype], "widget": {"name": wname}, "link": None})

        node = {
            "id": self._node_id,
            "type": node_type,
            "pos": [int(pos[0]), int(pos[1])],
            "size": [int(width), int(h)],
            "flags": {},
            "order": 0,
            "mode": 0,
            "inputs": inputs,
            "outputs": [{"name": n, "type": t, "links": None} for n, t in spec["outputs"]],
            "properties": {} if spec.get("virtual") else {"Node name for S&R": node_type},
            "widgets_values": widget_values,
        }
        if spec.get("custom"):
            node["properties"]["aux_id"] = PACK_ID
        if title:
            node["title"] = title
        if color:
            node["color"], node["bgcolor"] = COLORS[color]
        self.nodes.append(node)
        return node

    def note(self, pos, size, title, text):
        node = self.node("MarkdownNote", pos, size[0], title=title, color="black", values={"text": text}, height=size[1])
        return node

    # -- links ----------------------------------------------------------------
    def link(self, src, out_name, dst, in_name):
        out_idx = next(i for i, o in enumerate(src["outputs"]) if o["name"] == out_name)
        in_idx = next(i for i, o in enumerate(dst["inputs"]) if o["name"] == in_name)
        out, inp = src["outputs"][out_idx], dst["inputs"][in_idx]
        if inp["link"] is not None:
            raise ValueError(f"{dst['type']}.{in_name} already has a link")
        if inp["type"] != "*" and out["type"] != inp["type"]:
            raise ValueError(f"type mismatch {src['type']}.{out_name} ({out['type']}) -> {dst['type']}.{in_name} ({inp['type']})")
        self._link_id += 1
        out["links"] = (out["links"] or []) + [self._link_id]
        inp["link"] = self._link_id
        self.links.append([self._link_id, src["id"], out_idx, dst["id"], in_idx, out["type"]])

    # -- groups ---------------------------------------------------------------
    def group(self, title, members, color_key):
        x0 = min(n["pos"][0] for n in members) - 20
        y0 = min(n["pos"][1] for n in members) - TITLE_H - 50
        x1 = max(n["pos"][0] + n["size"][0] for n in members) + 20
        y1 = max(n["pos"][1] + n["size"][1] for n in members) + 20
        self.groups.append({
            "id": len(self.groups) + 1,
            "title": title,
            "bounding": [x0, y0, x1 - x0, y1 - y0],
            "color": GROUP_COLORS[color_key],
            "font_size": 24,
            "flags": {},
        })

    # -- output ---------------------------------------------------------------
    def _execution_order(self):
        deps = {n["id"]: set() for n in self.nodes}
        for _, src, _, dst, _, _ in self.links:
            deps[dst].add(src)
        order, placed = {}, set()
        while len(placed) < len(deps):
            ready = sorted(i for i, d in deps.items() if i not in placed and d <= placed)
            if not ready:
                raise ValueError("workflow has a cycle")
            for i in ready:
                order[i] = len(order)
                placed.add(i)
        return order

    def to_json(self):
        order = self._execution_order()
        for n in self.nodes:
            n["order"] = order[n["id"]]
        xs = [n["pos"][0] for n in self.nodes]
        ys = [n["pos"][1] for n in self.nodes]
        return {
            "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"https://github.com/{PACK_ID}/{self.name}")),
            "revision": 0,
            "last_node_id": self._node_id,
            "last_link_id": self._link_id,
            "nodes": self.nodes,
            "links": self.links,
            "groups": self.groups,
            "config": {},
            "extra": {
                "ds": {"scale": 0.5, "offset": [-min(xs) + 60, -min(ys) + 140]},
                "frontendVersion": FRONTEND_VERSION,
                "workflowRendererVersion": "Vue-corrected",
                "author": BRAND,
            },
            "version": 0.4,
        }


# ---------------------------------------------------------------------------
# Notes shared by the workflows
# ---------------------------------------------------------------------------
HF = "https://huggingface.co/"
FILES = {
    "moss": [
        ("diffusion_models", "moss_image_sft_int8_convrot.safetensors", HF + "Comfy-Org/MOSS-Image/resolve/main/diffusion_models/moss_image_sft_int8_convrot.safetensors", "5.4 GB"),
        ("text_encoders", "qwen3.5_9b_w4a8_gptq.safetensors", HF + "Comfy-Org/MOSS-Image/resolve/main/text_encoders/qwen3.5_9b_w4a8_gptq.safetensors", "7.3 GB"),
        ("vae", "flux2-vae_small.safetensors", HF + "Comfy-Org/MOSS-Image/resolve/main/vae/flux2-vae_small.safetensors", "0.25 GB"),
    ],
    "h3": [
        ("diffusion_models", "minimax_h3_fl2va_pruned_int8_convrot.safetensors", HF + "Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors", "21 GB"),
        ("text_encoders", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", HF + "Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "15.7 GB"),
        ("vae", "minimax_h3_video_vae_int8_convrot.safetensors", HF + "Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_int8_convrot.safetensors", "2.8 GB"),
        ("vae", "minimax_h3_audio_vae_fp32.safetensors", HF + "Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors", "0.6 GB"),
    ],
}


def file_list(key):
    lines = []
    for folder, name, url, size in FILES[key]:
        lines.append(f"- `models/{folder}/`\n  [{name}]({url}) ({size})")
    return "\n".join(lines)


def about_note(title, what, steps):
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))
    return (
        f"# {title}\n\n"
        f"**{BRAND}** · ComfyUI-Cluster\n\n"
        f"{what}\n\n"
        f"## How to use\n\n{numbered}\n\n"
        "## Reading the graph\n\n"
        "Groups run left to right and are numbered in the order they execute. "
        "The **Status** row at the bottom shows what the planner decided and what the loader did.\n\n"
        "## If something fails\n\n"
        "- **Model not installed**: the loader error lists the exact files and download links. Run `install.bat`.\n"
        "- **Planner says tag matching**: Ollama was not reachable or the model tag is not pulled. "
        "The workflow still runs; the summary shows the reason.\n"
    )


AUTO_MODELS_NOTE = (
    "# Models\n\n"
    "The planner chooses from the models **you have installed** that are listed in `model_registry.json`:\n\n"
    "| Family | Good at | VRAM |\n"
    "|---|---|---|\n"
    "| MOSS-Image | text in images, posters, layouts, all-round | 12 GB+ |\n"
    "| FLUX.3 Klein | photography, portraits, products | 12 GB+ |\n"
    "| Krea-2 Turbo | illustration, painterly, stylised | 16 GB+ |\n"
    "| Z-Image-Turbo | fast photorealism, low VRAM | 8 GB+ |\n\n"
    "VRAM figures are estimates.\n\n"
    "## Install\n\n"
    "Run `install.bat` in the ComfyUI-Cluster folder and tick the models you want. "
    "Each model's text encoder and VAE are selected with it.\n\n"
    "FLUX.3 Klein is a gated download: accept its licence on Hugging Face and set `HF_TOKEN` first.\n\n"
    "## Ollama\n\n"
    "```\nollama pull qwen3.5:9b\n```\n\n"
    "`qwen3.5:9b` reads both text and images, so it serves every planner. "
    "Any other Ollama tag works: type it into `ollama_model`.\n\n"
    "## Pin a family\n\n"
    "Set `task_hint` on the planner to `moss_image`, `flux3`, `krea2` or `z_image` to stop it choosing another family."
)


def status_row(g, x, y, planner, loader, lora=None, extra=None):
    """Bottom row of text previews. Returns the nodes for grouping."""
    members = []
    items = [("Plan summary", planner, "summary", 460), ("Prompt sent to the model", planner, "positive_prompt", 460),
             ("Loader log", loader, "debug_info", 520)]
    if lora is not None:
        items.append(("LoRA log", lora, "debug_info", 380))
    for title, src, out, width in items + list(extra or []):
        node = g.node("PreviewAny", (x, y), width, title=title, height=300)
        g.link(src, out, node, "source")
        members.append(node)
        x += width + 30
    return members


# ---------------------------------------------------------------------------
# Workflow definitions
# ---------------------------------------------------------------------------
X_NOTES = -1140
Y = 0
STATUS_GAP = 130


def image_pipeline(g, planner, x, latent_source, pinned_moss=False):
    """Load -> LoRA -> encode -> sample -> decode -> save. Shared by the image workflows."""
    loader = g.node("DynamicCheckpointLoader", (x, Y), 400, color="pale_blue")
    lora = g.node("DynamicLoraStack", (x, Y + loader["size"][1] + 80), 400, color="pale_blue")
    g.link(planner, "checkpoint", loader, "checkpoint")
    g.link(loader, "MODEL", lora, "model")
    g.link(loader, "CLIP", lora, "clip")
    g.link(planner, "loras", lora, "loras")
    g.link(planner, "lora_strengths", lora, "lora_strengths")
    load_members = [loader, lora]
    model_src, model_out = lora, "MODEL"
    if pinned_moss:
        slg = g.node("SkipLayerGuidanceDiTSimple", (x, lora["pos"][1] + lora["size"][1] + 80), 400,
                     title="Skip Layer Guidance (MOSS layer 22)", color="pale_blue")
        g.link(lora, "MODEL", slg, "model")
        load_members.append(slg)
        model_src, model_out = slg, "MODEL"
    g.group("2 · Load model and LoRAs", load_members, "load")

    x += 400 + 80
    pos = g.node("CLIPTextEncode", (x, Y), 420, title="Positive prompt", color="green", text_heights=[170])
    neg = g.node("CLIPTextEncode", (x, Y + pos["size"][1] + 80), 420, title="Negative prompt", color="red", text_heights=[130])
    for enc, out in ((pos, "positive_prompt"), (neg, "negative_prompt")):
        g.link(lora, "CLIP", enc, "clip")
        g.link(planner, out, enc, "text")
    g.group("3 · Encode prompts", [pos, neg], "prompt")

    x += 420 + 80
    sample_members = []
    if latent_source is None:
        latent_type = "EmptyFlux2LatentImage" if pinned_moss else "EmptySD3LatentImage"
        latent = g.node(latent_type, (x, Y), 340, title="Empty latent (size from planner)")
        g.link(planner, "width", latent, "width")
        g.link(planner, "height", latent, "height")
        latent_out = "LATENT"
    else:
        latent = g.node("VAEEncode", (x, Y), 340, title="Encode input image")
        g.link(latent_source, "IMAGE", latent, "pixels")
        g.link(loader, "VAE", latent, "vae")
        latent_out = "LATENT"
    sampler = g.node("KSampler", (x, Y + latent["size"][1] + 80), 340, title="KSampler (settings from planner)")
    g.link(model_src, model_out, sampler, "model")
    g.link(pos, "CONDITIONING", sampler, "positive")
    g.link(neg, "CONDITIONING", sampler, "negative")
    g.link(latent, latent_out, sampler, "latent_image")
    for name in ("seed", "steps", "cfg", "sampler_name", "scheduler", "denoise"):
        g.link(planner, name, sampler, name)
    decode = g.node("VAEDecode", (x, sampler["pos"][1] + sampler["size"][1] + 80), 340)
    g.link(sampler, "LATENT", decode, "samples")
    g.link(loader, "VAE", decode, "vae")
    sample_members += [latent, sampler, decode]
    g.group("4 · Sample and decode", sample_members, "sample")

    x += 340 + 80
    save = g.node("SaveImage", (x, Y), 560, values={"filename_prefix": f"Geekatplay/{g.name}"}, height=640)
    g.link(decode, "IMAGE", save, "images")
    g.group("5 · Result", [save], "output")
    return loader, lora, [loader, lora, pos, neg, latent, sampler, decode, save] + load_members


def finish(g, about, models, top_nodes, status_nodes_fn):
    bottom = max(n["pos"][1] + n["size"][1] for n in top_nodes)
    status = status_nodes_fn(bottom + STATUS_GAP + TITLE_H + 50)
    g.group("Status · what the planner and loader did", status, "status")
    # Both notes get the height the longer one needs, so neither scrolls or clips.
    first = g.note((X_NOTES, Y), (520, 100), "About this workflow", about)
    second = g.note((X_NOTES + 540, Y), (520, 100), "Models and setup", models)
    note_h = int(max(_text_fits(first)[0], _text_fits(second)[0])) + 70
    first["size"][1] = second["size"][1] = note_h


def build_text_to_image_auto():
    g = Graph("text_to_image_auto")
    planner = g.node("OllamaPromptPlanner", (0, Y), 460, color="purple", text_heights=[190, 90], values={
        "prompt": "A weathered lighthouse keeper reading a letter by a window, storm clouds over the sea, "
                  "warm lamp light, 35mm film photograph",
        "user_negative": "blurry, low quality, watermark",
        "aspect_ratio": "3:2",
    })
    g.group("1 · Describe and plan", [planner], "plan")
    loader, lora, top = image_pipeline(g, planner, 460 + 80, None)
    finish(g, about_note(
        "Text to Image · Auto Router",
        "Type a prompt. A local Ollama model reads it, picks the best installed image model and LoRAs, "
        "rewrites the prompt for that model, and sets size, steps, CFG and sampler for you.",
        ["Start Ollama.", "Write your prompt in **Ollama Prompt Planner**.", "Pick `aspect_ratio` and `max_vram` for your GPU.",
         "Press **Run**. Leave `seed` on *randomize* for a new image each run."],
    ), AUTO_MODELS_NOTE, [planner] + top, lambda y: status_row(g, 0, y, planner, loader, lora))
    return g


def build_text_to_image_moss():
    g = Graph("text_to_image_moss")
    planner = g.node("OllamaPromptPlanner", (0, Y), 460, color="purple", text_heights=[190, 90], values={
        "prompt": "Event poster for a jazz night: a saxophonist in silhouette under a single spotlight, "
                  "headline text \"BLUE HOUR\" in tall condensed capitals, smaller line \"Live every Friday\"",
        "task_hint": "moss_image",
        "user_negative": "",
        "aspect_ratio": "4:5",
    })
    g.group("1 · Describe and plan", [planner], "plan")
    loader, lora, top = image_pipeline(g, planner, 460 + 80, None, pinned_moss=True)
    models = (
        "# Models\n\n"
        "This workflow is pinned to **MOSS-Image** with `task_hint = moss_image`. "
        "MOSS-Image is strong at readable text, posters and layouts in English and Chinese.\n\n"
        "## Files\n\n" + file_list("moss") + "\n\n"
        "`install.bat` downloads all three. Licence: Apache 2.0.\n\n"
        "## Settings used\n\n"
        "| Setting | Value |\n|---|---|\n"
        "| Steps / CFG | 25 / 2.5 |\n| Sampler | er_sde, simple |\n| Skip layer | 22, first half of sampling |\n"
        "| Size | about 1 megapixel, sides in steps of 16 |\n\n"
        "These follow the official ComfyUI MOSS-Image template. For results closer to the model authors' "
        "reference, set `steps` to 50 and `cfg` to 3 in the registry entry.\n\n"
        "## Tips\n\n"
        "- Put text you want rendered inside quotes.\n"
        "- A negative prompt is used here because CFG is above 1. Leave `user_negative` empty to get the recommended default.\n\n"
        "## Ollama\n\n```\nollama pull qwen3.5:9b\n```"
    )
    finish(g, about_note(
        "Text to Image · MOSS-Image",
        "The same planner-driven pipeline, pinned to MOSS-Image and wired the way the official template is: "
        "FLUX.2 latent, skip layer guidance on layer 22, er_sde sampler.",
        ["Start Ollama.", "Write your prompt. Put any text that must appear in the image in quotes.",
         "Press **Run**."],
    ), models, [planner] + top, lambda y: status_row(g, 0, y, planner, loader, lora))
    return g


def _image_input(g):
    load = g.node("LoadImage", (0, Y), 400, title="Load input image", height=420)
    scale = g.node("ImageScaleToTotalPixels", (0, Y + load["size"][1] + 80), 400, title="Resize to about 1 megapixel")
    g.link(load, "IMAGE", scale, "image")
    g.group("1 · Input image", [load, scale], "output")
    return load, scale


IMG2IMG_MODELS_NOTE = AUTO_MODELS_NOTE + (
    "\n\n## Denoise\n\n"
    "The planner sets `denoise` from your request: about 0.3-0.5 for small fixes, 0.6-0.9 for a new style. "
    "To override it, disconnect `denoise` on the KSampler and type a value."
)


def build_image_to_image_auto():
    g = Graph("image_to_image_auto")
    load, scale = _image_input(g)
    planner = g.node("OllamaVisionStylePlanner", (480, Y), 460, color="purple", text_heights=[170, 90], values={
        "prompt": "Keep the composition and subject. Re-render as a moody cinematic photograph with rich detail and soft rim light.",
        "task_hint": "img2img",
        "user_negative": "blurry, oversmoothed, artifacts",
    })
    g.link(scale, "IMAGE", planner, "image")
    g.group("2 · Look and plan", [planner], "plan")
    loader, lora, top = image_pipeline(g, planner, 480 + 460 + 80, scale)
    _renumber(g, start=3)
    finish(g, about_note(
        "Image to Image · Auto Router",
        "Load a picture and say what to do with it. A local Ollama vision model studies the image, "
        "picks the model and LoRAs, and decides how strongly to change the picture.",
        ["Start Ollama with a vision-capable model.", "Choose a picture in **Load input image**.",
         "Describe the change in **Ollama Vision Style Planner**.", "Press **Run**."],
    ), IMG2IMG_MODELS_NOTE, [load, scale, planner] + top, lambda y: status_row(g, 0, y, planner, loader, lora))
    return g


def build_image_dual_prompt():
    g = Graph("image_dual_prompt")
    load, scale = _image_input(g)
    planner = g.node("OllamaVisionDualPlanner", (480, Y), 460, color="purple", text_heights=[130, 150, 90], values={
        "vision_prompt": "Describe the style, composition, lighting, and key elements of this image.",
        "generation_prompt": "Turn the scene into a neon-lit cyberpunk street at night, high contrast, rain on the ground.",
        "task_hint": "img2img",
        "user_negative": "blurry, oversmoothed, artifacts",
    })
    g.link(scale, "IMAGE", planner, "image")
    g.group("2 · Look and plan", [planner], "plan")
    loader, lora, top = image_pipeline(g, planner, 480 + 460 + 80, scale)
    _renumber(g, start=3)
    finish(g, about_note(
        "Image to Image · Dual Prompt",
        "Two prompts instead of one. The **vision prompt** tells the Ollama vision model what to look for "
        "in your picture; the **generation prompt** says what to make from it.",
        ["Start Ollama with a vision-capable model.", "Choose a picture in **Load input image**.",
         "Edit both prompts in **Ollama Vision Dual Planner**.", "Press **Run**."],
    ), IMG2IMG_MODELS_NOTE, [load, scale, planner] + top, lambda y: status_row(g, 0, y, planner, loader, lora))
    return g


def _renumber(g, start):
    """Image workflows have an extra first group, so shift the shared pipeline's numbers."""
    n = start
    for grp in g.groups:
        head, sep, tail = grp["title"].partition(" · ")
        if sep and head.isdigit() and int(head) >= 2 and grp["title"] not in ("2 · Look and plan",):
            grp["title"] = f"{n} · {tail}"
            n += 1


def build_text_to_video_minimax_h3():
    g = Graph("text_to_video_minimax_h3")
    planner = g.node("OllamaPromptPlanner", (0, Y), 460, color="purple", text_heights=[190, 90], values={
        "prompt": "A red fox trots through fresh snow at sunrise, stops and looks at the camera. "
                  "Quiet wind, soft crunch of footsteps, a distant bird call.",
        "task_hint": "video",
        "aspect_ratio": "16:9",
        "base_size": 768,
    })
    g.group("1 · Describe and plan", [planner], "plan")

    x = 540
    loader = g.node("DynamicCheckpointLoader", (x, Y), 400, color="pale_blue")
    audio_vae = g.node("VAELoader", (x, Y + loader["size"][1] + 80), 400, title="Load audio VAE", color="pale_blue",
                       values={"vae_name": "minimax_h3_audio_vae_fp32.safetensors"})
    audio_vae["properties"]["models"] = [{
        "name": "minimax_h3_audio_vae_fp32.safetensors",
        "url": FILES["h3"][3][2],
        "directory": "vae",
    }]
    g.link(planner, "checkpoint", loader, "checkpoint")
    g.group("2 · Load model", [loader, audio_vae], "load")

    x += 480
    cond = g.node("MiniMaxH3ImageToVideo", (x, Y), 420, title="MiniMax H3 prompt and video latent", color="green",
                  text_heights=[190], values={"length": 124})
    g.link(loader, "CLIP", cond, "clip")
    g.link(loader, "VAE", cond, "vae")
    g.link(planner, "positive_prompt", cond, "prompt")
    g.link(planner, "width", cond, "width")
    g.link(planner, "height", cond, "height")
    g.group("3 · Prompt and video length", [cond], "prompt")

    x += 500
    noise = g.node("RandomNoise", (x, Y), 340)
    select = g.node("KSamplerSelect", (x, noise["pos"][1] + noise["size"][1] + 80), 340)
    sched = g.node("BasicScheduler", (x, select["pos"][1] + select["size"][1] + 80), 340)
    guider = g.node("BasicGuider", (x, sched["pos"][1] + sched["size"][1] + 80), 340)
    sampler = g.node("SamplerCustomAdvanced", (x, guider["pos"][1] + guider["size"][1] + 80), 340)
    g.link(planner, "seed", noise, "noise_seed")
    g.link(planner, "sampler_name", select, "sampler_name")
    g.link(loader, "MODEL", sched, "model")
    g.link(planner, "scheduler", sched, "scheduler")
    g.link(planner, "steps", sched, "steps")
    g.link(loader, "MODEL", guider, "model")
    g.link(cond, "positive", guider, "conditioning")
    g.link(noise, "NOISE", sampler, "noise")
    g.link(guider, "GUIDER", sampler, "guider")
    g.link(select, "SAMPLER", sampler, "sampler")
    g.link(sched, "SIGMAS", sampler, "sigmas")
    g.link(cond, "LATENT", sampler, "latent_image")
    g.group("4 · Sample", [noise, select, sched, guider, sampler], "sample")

    x += 420
    decode = g.node("VAEDecode", (x, Y), 340, title="Decode video frames")
    decode_audio = g.node("VAEDecodeAudio", (x, decode["pos"][1] + decode["size"][1] + 80), 340, title="Decode audio")
    video = g.node("CreateVideo", (x, decode_audio["pos"][1] + decode_audio["size"][1] + 80), 340, title="Combine frames and audio (24 fps)")
    save = g.node("SaveVideo", (x, video["pos"][1] + video["size"][1] + 80), 340,
                  values={"filename_prefix": f"video/Geekatplay/{g.name}"}, extra_height=260)
    g.link(sampler, "output", decode, "samples")
    g.link(loader, "VAE", decode, "vae")
    g.link(sampler, "output", decode_audio, "samples")
    g.link(audio_vae, "VAE", decode_audio, "vae")
    g.link(decode, "IMAGE", video, "images")
    g.link(decode_audio, "AUDIO", video, "audio")
    g.link(video, "VIDEO", save, "video")
    g.group("5 · Decode and save", [decode, decode_audio, video, save], "output")

    models = (
        "# Models\n\n"
        "**MiniMax H3** makes video and matching stereo sound in one pass: voices, effects and music.\n\n"
        "## Files\n\n" + file_list("h3") + "\n\n"
        "`install.bat` downloads all four. About 40 GB in total; plan for a 24 GB GPU (estimate).\n\n"
        "## Settings used\n\n"
        "| Setting | Value |\n|---|---|\n"
        "| Steps | 20 |\n| Sampler | res_multistep, simple |\n| Size | 768 px short edge, up to 1344 x 768 |\n"
        "| Length | 124 frames = about 5 s at 24 fps |\n\n"
        "These follow the official ComfyUI MiniMax H3 text-to-video template.\n\n"
        "## Length\n\n"
        "`length` on the H3 node is in frames. Valid values are 5, 22, 39, ... in steps of 17. "
        "The model was trained for about 124 to 362 frames (5 to 15 s).\n\n"
        "## Faster drafts\n\n"
        "The registry also lists **FastVideo FastH3 8-step**. To use it, disable the base H3 entry in "
        "`model_registry.json` (or install only FastH3) and add a *ModelSamplingMiniMaxH3* node with "
        "video shift 10 after the loader.\n\n"
        "## Ollama\n\n```\nollama pull qwen3.5:9b\n```"
    )
    top = [planner, loader, audio_vae, cond, noise, select, sched, guider, sampler, decode, decode_audio, video, save]
    finish(g, about_note(
        "Text to Video · MiniMax H3",
        "Type what should happen and what it should sound like. The planner turns it into a shot-by-shot "
        "description with audio, and MiniMax H3 renders video and sound together.",
        ["Start Ollama.", "Describe the action **and the sound** in **Ollama Prompt Planner**.",
         "Set `length` on the H3 node (124 frames is about 5 seconds).", "Press **Run**."],
    ), models, top, lambda y: status_row(g, 0, y, planner, loader))
    return g


WORKFLOWS = {
    "geekatplay_text_to_image_auto.json": build_text_to_image_auto,
    "geekatplay_text_to_image_moss.json": build_text_to_image_moss,
    "geekatplay_image_to_image_auto.json": build_image_to_image_auto,
    "geekatplay_image_dual_prompt.json": build_image_dual_prompt,
    "geekatplay_text_to_video_minimax_h3.json": build_text_to_video_minimax_h3,
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _text_fits(node):
    """Rough check that a note's text fits its box: 7.4 px per character, 22 px per line."""
    text = node["widgets_values"][0]
    width = node["size"][0] - 44
    height = 0
    in_code = False
    for line in text.split("\n"):
        if line.startswith("```"):
            in_code = not in_code
            height += 12
            continue
        if not line.strip():
            height += 10
            continue
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            char_w, line_h = (15, 44) if level == 1 else (11, 36)
        elif line.startswith("|"):
            if set(line) <= set("|-: "):
                continue
            cells = [c for c in line.strip("|").split("|")]
            col_w = width / len(cells)
            rows = max(max(1, -(-len(c.strip()) * 7.4 // max(col_w - 16, 1))) for c in cells)
            height += rows * 22 + 12
            continue
        else:
            char_w, line_h = 7.4, 22
        plain = line
        while "](" in plain:  # count link text only, not the URL
            a = plain.index("](")
            b = plain.index(")", a)
            plain = plain[:a] + plain[b + 1:]
        height += max(1, -(-len(plain) * char_w // width)) * line_h
    return height, node["size"][1] - 30


def validate(wf, name):
    errors = []
    nodes = {n["id"]: n for n in wf["nodes"]}
    links = {l[0]: l for l in wf["links"]}

    for n in wf["nodes"]:
        spec = SPECS.get(n["type"])
        if spec is None:
            errors.append(f"unknown node type {n['type']}")
            continue
        if len(n["widgets_values"]) != len(spec["widgets"]):
            errors.append(f"{n['type']} #{n['id']}: {len(n['widgets_values'])} widget values, expected {len(spec['widgets'])}")
        if n["size"][1] < min_height(n["type"], [60] * 4) - 1:
            errors.append(f"{n['type']} #{n['id']}: height {n['size'][1]} clips its widgets")
        title = n.get("title") or n["type"]
        if len(title) * 8.2 + 50 > n["size"][0]:
            errors.append(f"{n['type']} #{n['id']}: title '{title}' is wider than the node")
        for inp in n["inputs"]:
            required = "widget" not in inp and inp["name"] not in spec.get("optional", [])
            if inp["link"] is None and required:
                errors.append(f"{n['type']} #{n['id']}: input '{inp['name']}' is not connected")
            if inp["link"] is not None:
                l = links.get(inp["link"])
                if l is None or l[3] != n["id"] or n["inputs"][l[4]] is not inp:
                    errors.append(f"{n['type']} #{n['id']}: input '{inp['name']}' has a broken link")
        for idx, out in enumerate(n["outputs"]):
            for lid in out["links"] or []:
                l = links.get(lid)
                if l is None or l[1] != n["id"] or l[2] != idx:
                    errors.append(f"{n['type']} #{n['id']}: output '{out['name']}' has a broken link")
        if n["type"] == "MarkdownNote":
            need, have = _text_fits(n)
            if need > have:
                errors.append(f"note '{n.get('title')}': text needs about {int(need)} px, box has {have}")

    for lid, src, so, dst, di, typ in wf["links"]:
        a, b = nodes.get(src), nodes.get(dst)
        if a is None or b is None:
            errors.append(f"link {lid}: missing node")
            continue
        out_t, in_t = a["outputs"][so]["type"], b["inputs"][di]["type"]
        if out_t != typ or (in_t != "*" and in_t != typ):
            errors.append(f"link {lid}: {a['type']}.{a['outputs'][so]['name']} ({out_t}) -> {b['type']}.{b['inputs'][di]['name']} ({in_t})")
        if lid not in (a["outputs"][so]["links"] or []) or b["inputs"][di]["link"] != lid:
            errors.append(f"link {lid}: not registered on both ends")

    def box(n):
        return (n["pos"][0], n["pos"][1] - TITLE_H, n["pos"][0] + n["size"][0], n["pos"][1] + n["size"][1])

    def overlap(p, q, gap=0):
        return p[0] < q[2] + gap and q[0] < p[2] + gap and p[1] < q[3] + gap and q[1] < p[3] + gap

    items = wf["nodes"]
    for i, p in enumerate(items):
        for q in items[i + 1:]:
            if overlap(box(p), box(q), gap=10):
                errors.append(f"nodes overlap: {p.get('title') or p['type']} / {q.get('title') or q['type']}")

    group_boxes = []
    for grp in wf["groups"]:
        x, y, w, h = grp["bounding"]
        group_boxes.append((grp, (x, y, x + w, y + h)))
        if len(grp["title"]) * 13.5 + 30 > w:
            errors.append(f"group title '{grp['title']}' is wider than the group")
    for i, (ga, ba) in enumerate(group_boxes):
        for gb, bb in group_boxes[i + 1:]:
            if overlap(ba, bb):
                errors.append(f"groups overlap: {ga['title']} / {gb['title']}")
    for n in items:
        b = box(n)
        inside = [g for g, gb in group_boxes if gb[0] <= b[0] and gb[1] + 40 <= b[1] and b[2] <= gb[2] and b[3] <= gb[3]]
        touching = [g for g, gb in group_boxes if overlap(b, gb)]
        if n["type"] == "MarkdownNote":
            if touching:
                errors.append(f"note '{n.get('title')}' overlaps group '{touching[0]['title']}'")
        elif len(inside) != 1 or len(touching) != 1:
            errors.append(f"{n.get('title') or n['type']} #{n['id']}: must sit inside exactly one group")

    numbered = [g["title"].split(" · ")[0] for g in wf["groups"] if g["title"][0].isdigit()]
    if numbered != [str(i) for i in range(1, len(numbered) + 1)]:
        errors.append(f"group numbers are not 1..n in order: {numbered}")

    blob = json.dumps(wf)
    if BRAND not in blob:
        errors.append("branding missing")
    return errors


def main():
    check_only = "--check" in sys.argv
    os.makedirs(OUT_DIR, exist_ok=True)
    failed = False
    for filename, builder in WORKFLOWS.items():
        path = os.path.join(OUT_DIR, filename)
        if check_only:
            with open(path, "r", encoding="utf-8") as f:
                wf = json.load(f)
        else:
            wf = builder().to_json()
        errors = validate(wf, filename)
        status = "ok" if not errors else "FAILED"
        print(f"{status:7} {filename}: {len(wf['nodes'])} nodes, {len(wf['links'])} links, {len(wf['groups'])} groups")
        for e in errors:
            print(f"        - {e}")
        failed = failed or bool(errors)
        if not check_only and not errors:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(wf, f, indent=2, ensure_ascii=False)
                f.write("\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
