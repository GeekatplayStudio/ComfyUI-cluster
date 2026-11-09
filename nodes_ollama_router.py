"""
ComfyUI-Cluster by Geekatplay Studio.

Ollama-driven routing, loading and status nodes. A local Ollama model reads the
prompt (and optionally an image), picks the best installed model from
``model_registry.json`` and hands every sampler setting to the rest of the graph.

The module is organised top to bottom as:
    1. model families   - per-family defaults (text encoder type, sizes, sampler)
    2. registry helpers - reading, filtering and scoring registry entries
    3. Ollama client    - one cached JSON chat call, with or without images
    4. planning         - shared plan/validate/finalise pipeline for all planners
    5. nodes            - planners, universal loader, LoRA stack, status helpers
"""

import base64
import hashlib
import io
import json
import math
import os
import re
import urllib.error
import urllib.request

import numpy as np
from PIL import Image

import folder_paths
import comfy.sd
import comfy.utils
import comfy.samplers


BRAND_NAME = "Geekatplay Studio"
LOG_PREFIX = f"[{BRAND_NAME} | Cluster]"
CATEGORY_ROOT = BRAND_NAME

DEFAULT_REGISTRY_FILE = "model_registry.json"
# qwen3.5:9b reads images too, so one Ollama model serves the text and vision planners.
DEFAULT_TEXT_MODEL = "qwen3.5:9b"
DEFAULT_VISION_MODEL = "qwen3.5:9b"
OLLAMA_TIMEOUT_SECONDS = 180
MAX_SEED = 0xFFFFFFFFFFFFFFFF


def _log(message: str, leading_newline: bool = False) -> None:
    """Console logger with the Geekatplay Studio prefix so messages are easy to spot."""
    text = f"{LOG_PREFIX} {message}"
    print(f"\n{text}" if leading_newline else text)


# ---------------------------------------------------------------------------
# 1. Model families
# ---------------------------------------------------------------------------
# One entry per model family the router understands. Registry entries point at a
# family through their "type" field and may override any of these defaults.
#
#   clip_type   comfy.sd.CLIPType name used when loading the text encoder
#   media       "image" or "video"; image workflows never pick video models
#   multiple    width/height are snapped to this many pixels
#   sizing      "area"       -> base_size is the side of a square of equal area
#               "short_edge" -> base_size is the short edge (video models)
FAMILIES = {
    "moss_image": {
        "label": "MOSS-Image",
        "clip_type": "moss_image",
        "media": "image",
        "multiple": 16,
        "sizing": "area",
        "default_size": 1024,
        "min_size": 512,
        "max_size": 2048,
        "steps": 25,
        "cfg": 2.5,
        "sampler_name": "er_sde",
        "scheduler": "simple",
    },
    "flux3": {
        "label": "FLUX.3 Klein",
        "clip_type": "flux3",
        "media": "image",
        "multiple": 16,
        "sizing": "area",
        "default_size": 1024,
        "min_size": 512,
        "max_size": 2048,
        "steps": 16,
        "cfg": 1.0,
        "sampler_name": "euler",
        "scheduler": "simple",
    },
    "krea2": {
        "label": "Krea-2",
        "clip_type": "krea2",
        "media": "image",
        "multiple": 16,
        "sizing": "area",
        "default_size": 1024,
        "min_size": 512,
        "max_size": 2048,
        "steps": 8,
        "cfg": 1.0,
        "sampler_name": "euler",
        "scheduler": "simple",
    },
    "z_image": {
        "label": "Z-Image",
        "clip_type": "lumina2",
        "media": "image",
        "multiple": 16,
        "sizing": "area",
        "default_size": 1024,
        "min_size": 512,
        "max_size": 2048,
        "steps": 8,
        "cfg": 1.0,
        "sampler_name": "res_multistep",
        "scheduler": "simple",
    },
    "minimax_h3": {
        "label": "MiniMax H3",
        "clip_type": "minimax",
        "media": "video",
        "multiple": 32,
        "sizing": "short_edge",
        "default_size": 768,
        "min_size": 352,
        "max_size": 1344,
        "steps": 20,
        "cfg": 1.0,
        "sampler_name": "res_multistep",
        "scheduler": "simple",
    },
    # Any all-in-one checkpoint file (model + text encoder + VAE in one file).
    "checkpoint": {
        "label": "All-in-one checkpoint",
        "clip_type": "stable_diffusion",
        "media": "image",
        "multiple": 64,
        "sizing": "area",
        "default_size": 1024,
        "min_size": 512,
        "max_size": 2048,
        "steps": 30,
        "cfg": 6.5,
        "sampler_name": "dpmpp_2m",
        "scheduler": "karras",
    },
}

FAMILY_NAMES = list(FAMILIES.keys())
TASK_HINTS = ["auto", "text2img", "img2img", "video"] + [f for f in FAMILY_NAMES if f != "checkpoint"]
LOADER_MODEL_TYPES = ["auto"] + FAMILY_NAMES
VRAM_OPTIONS = [48, 32, 24, 16, 12, 8]
SPLIT_FOLDERS = ("diffusion_models", "unet")

ASPECT_RATIOS = {
    "1:1": (1, 1),
    "3:2": (3, 2),
    "2:3": (2, 3),
    "4:3": (4, 3),
    "3:4": (3, 4),
    "16:9": (16, 9),
    "9:16": (9, 16),
    "21:9": (21, 9),
    "9:21": (9, 21),
    "2:1": (2, 1),
    "1:2": (1, 2),
    "5:3": (5, 3),
    "3:5": (3, 5),
    "4:5": (4, 5),
    "5:4": (5, 4),
}


def _family(model_type: str) -> dict:
    """Return the family defaults for a registry type, falling back to a generic checkpoint."""
    return FAMILIES.get(str(model_type or "").lower(), FAMILIES["checkpoint"])


def _entry_media(entry: dict) -> str:
    return str(entry.get("media") or _family(entry.get("type")).get("media", "image"))


# ---------------------------------------------------------------------------
# 2. Registry helpers
# ---------------------------------------------------------------------------
_EMPTY_REGISTRY = {"version": 2, "checkpoints": [], "loras": [], "text_encoders": [], "vae": []}

# Where ComfyUI may keep each kind of file (covers extra_model_paths.yaml aliases).
_FOLDER_SEARCH = {
    "checkpoints": ["checkpoints", "diffusion_models", "unet"],
    "diffusion_models": ["diffusion_models", "unet", "checkpoints"],
    "unet": ["diffusion_models", "unet", "checkpoints"],
    "clip": ["text_encoders", "clip"],
    "text_encoders": ["text_encoders", "clip"],
    "vae": ["vae"],
    "loras": ["loras"],
}


def _read_registry(registry_path: str) -> dict:
    """Load the model registry JSON, returning an empty registry if it is missing or invalid."""
    path = registry_path or DEFAULT_REGISTRY_FILE
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(__file__), path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        _log(f"Registry at {path} is not a JSON object; using an empty registry.")
    except FileNotFoundError:
        _log(f"Registry not found at {path}; using an empty registry.")
    except Exception as exc:
        _log(f"Registry at {path} could not be read ({exc}); using an empty registry.")
    return dict(_EMPTY_REGISTRY)


def _search_folders(folder_type: str) -> list[str]:
    return _FOLDER_SEARCH.get(folder_type, [folder_type])


def _quick_find(folder_type: str, filename: str) -> tuple[str | None, str | None]:
    """Fast lookup through ComfyUI's own path mapping. Returns (path, folder it was found in)."""
    if not filename:
        return None, None
    for ft in _search_folders(folder_type):
        try:
            path = folder_paths.get_full_path(ft, filename)
        except Exception:
            path = None
        if path:
            return path, ft
    return None, None


def _is_installed(entry: dict, default_folder: str = "checkpoints") -> bool:
    path, _ = _quick_find(entry.get("folder_type") or default_folder, entry.get("name", ""))
    return path is not None


def _find_entry(items: list, name: str) -> dict | None:
    for item in items or []:
        if item.get("name") == name:
            return item
    return None


def _lora_fits(lora: dict, model_type: str) -> bool:
    """A LoRA only works on the family it was trained for."""
    return str(lora.get("type", "")).lower() == str(model_type or "").lower()


def _score_tags(prompt: str, tags: list[str]) -> int:
    """
    Score how well a registry entry's tags match a prompt.
    Whole-word hits count most, substring hits less, and a clear style clash
    (anime prompt vs photo-only model, or the reverse) is penalised.
    """
    prompt = (prompt or "").lower()
    padded = " " + re.sub(r"[^a-z0-9]", " ", prompt) + " "
    words = [w for w in padded.split() if len(w) > 3]

    score = 0
    matched = 0
    for tag in tags or []:
        t = str(tag).lower().replace("-", " ").strip()
        if not t:
            continue
        if f" {t} " in padded:
            score += 5
            matched += 1
        elif t in padded:
            score += 2
            matched += 1
        elif any(w in t for w in words):
            score += 1

    joined = " ".join(str(t).lower() for t in tags or [])
    wants_anime = any(x in prompt for x in ("anime", "cartoon", "illustration", "manga", "2d"))
    wants_photo = any(x in prompt for x in ("photo", "realis", "dslr", "raw", "cinematic"))
    has_anime = "anime" in joined or "cartoon" in joined or "illustration" in joined
    has_photo = "photo" in joined or "realis" in joined
    if wants_anime and has_photo and not has_anime:
        score -= 10
    if wants_photo and has_anime and not has_photo:
        score -= 10

    if matched > 1:
        score += matched * 2
    return score


def _select_candidates(registry: dict, task_hint: str, max_vram: int) -> tuple[list[dict], list[str]]:
    """
    Narrow the registry to checkpoints the planner is allowed to choose from.
    Returns (candidates, notes). Filters are relaxed one at a time rather than
    returning nothing, and every relaxation is reported in the notes.
    """
    notes = []
    enabled = [c for c in registry.get("checkpoints", []) if c.get("enabled", True) is not False and c.get("name")]
    if not enabled:
        return [], ["Registry has no enabled checkpoints."]

    hint = str(task_hint or "auto").lower()
    if hint in FAMILIES and hint != "checkpoint":
        pool = [c for c in enabled if str(c.get("type", "")).lower() == hint]
        if not pool:
            notes.append(f"No enabled '{hint}' model in the registry; using any image model instead.")
            pool = [c for c in enabled if _entry_media(c) == "image"] or enabled
    else:
        media = "video" if hint == "video" else "image"
        pool = [c for c in enabled if _entry_media(c) == media]
        if not pool:
            notes.append(f"No enabled {media} model in the registry; using every enabled model.")
            pool = enabled

    fits = [c for c in pool if float(c.get("min_vram", 0) or 0) <= float(max_vram)]
    if not fits:
        notes.append(f"No model fits in {max_vram} GB VRAM; ignoring the VRAM limit.")
        fits = pool

    installed = [c for c in fits if _is_installed(c)]
    if installed:
        fits = installed
    else:
        notes.append("None of the matching models are installed yet; run the installer to download one.")
    return fits, notes


def _best_by_tags(prompt: str, candidates: list[dict]) -> dict | None:
    best, best_score = None, None
    for entry in candidates:
        score = _score_tags(prompt, entry.get("tags", []))
        if entry.get("recommended"):
            score += 3
        if best_score is None or score > best_score:
            best, best_score = entry, score
    return best


def _compact_for_llm(candidates: list[dict], loras: list[dict]) -> dict:
    """Trim registry entries to the few fields the language model needs."""
    def slim(item):
        out = {
            "name": item.get("name", ""),
            "type": item.get("type", ""),
            "tags": item.get("tags", [])[:10],
        }
        if item.get("title"):
            out["title"] = item["title"]
        if item.get("recommended"):
            out["recommended"] = True
        return out

    return {"checkpoints": [slim(c) for c in candidates[:40]], "loras": [slim(l) for l in loras[:40]]}


# ---------------------------------------------------------------------------
# 3. Ollama client
# ---------------------------------------------------------------------------
_OLLAMA_CACHE: dict[str, str] = {}
_OLLAMA_CACHE_LIMIT = 32


def _encode_image_base64(image) -> str:
    """Convert a ComfyUI IMAGE tensor (B,H,W,C float 0..1) into a base64 PNG, long side capped at 1024."""
    if image is None:
        return ""
    try:
        img = image
        if hasattr(img, "cpu"):
            img = img.cpu().numpy()
        if isinstance(img, np.ndarray) and img.ndim == 4:
            img = img[0]
        if not isinstance(img, np.ndarray):
            return ""
        pil = Image.fromarray(np.clip(img * 255.0, 0, 255).astype(np.uint8))
        if max(pil.size) > 1024:
            pil.thumbnail((1024, 1024))
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as exc:
        _log(f"Could not encode the input image for Ollama: {exc}")
        return ""


def _ollama_chat(model: str, system_prompt: str, user_prompt: str, images: list[str] | None = None,
                 host: str = "localhost", port: int = 11434, timeout: int = OLLAMA_TIMEOUT_SECONDS) -> str:
    """
    Ask a local Ollama model for a JSON answer. Identical requests are answered
    from a small in-memory cache, so re-rolling the seed does not call Ollama again.
    """
    user_message = {"role": "user", "content": user_prompt}
    if images:
        user_message["images"] = images
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt}, user_message],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.2},
    }
    data = json.dumps(payload).encode("utf-8")
    cache_key = hashlib.sha1(f"{host}:{port}".encode("utf-8") + data).hexdigest()
    if cache_key in _OLLAMA_CACHE:
        return _OLLAMA_CACHE[cache_key]

    req = urllib.request.Request(
        f"http://{host}:{port}/api/chat", data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="ignore")

    content = raw
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and isinstance(parsed.get("message"), dict):
            content = parsed["message"].get("content", "")
    except Exception:
        pass

    if len(_OLLAMA_CACHE) >= _OLLAMA_CACHE_LIMIT:
        _OLLAMA_CACHE.pop(next(iter(_OLLAMA_CACHE)))
    _OLLAMA_CACHE[cache_key] = content
    return content


def _parse_plan(raw: str) -> dict | None:
    """Extract the JSON plan from model output, tolerating extra text around it."""
    if not raw:
        return None
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                return data
        except Exception:
            return None
    return None


def _describe_ollama_error(exc: Exception, model: str, host: str, port: int) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code == 404:
            return f"Ollama has no model '{model}'. Run: ollama pull {model}"
        return f"Ollama returned HTTP {exc.code}."
    if isinstance(exc, urllib.error.URLError):
        return f"Ollama is not reachable at {host}:{port} ({exc.reason})."
    if isinstance(exc, TimeoutError):
        return f"Ollama did not answer within {OLLAMA_TIMEOUT_SECONDS}s."
    return f"Ollama call failed: {exc}"


# ---------------------------------------------------------------------------
# 4. Planning
# ---------------------------------------------------------------------------
_PLAN_KEYS = (
    "Return ONLY one JSON object with these keys: "
    "task (text2img|img2img), "
    "checkpoint (string), "
    "loras (array of names, may be empty), "
    "lora_strengths (array of floats 0.3-1.0, one per lora), "
    "positive_prompt (string), "
    "negative_prompt (string), "
    "denoise (float 0-1). "
    "Rules: choose checkpoint and loras ONLY from the names listed in registry, copied exactly. "
    "Only use a lora whose type equals the chosen checkpoint's type, and only when it clearly helps. "
    "Prefer a checkpoint marked recommended when several fit equally well. "
    "Write positive_prompt as one detailed natural-language description in full sentences "
    "(subject, setting, lighting, camera, style); keep any text the user put in quotes exactly as written. "
    "Keep negative_prompt short."
)

_DENOISE_RULES = (
    " For img2img, set denoise by how much should change: "
    "0.3-0.5 for small fixes that keep the picture, 0.6-0.9 for a new style or a variation. "
    "Use 1.0 only when the input image should be ignored."
)

_VIDEO_RULES = (
    " The checkpoint is a video model with sound: describe the shots, camera moves and motion over time, "
    "and the audio (dialogue, sound effects, music) in the same positive_prompt."
)


def _compute_resolution(aspect_ratio: str, base_size: int, model_type: str) -> tuple[int, int]:
    """
    Turn an aspect ratio and a base size into width/height for a model family.
    Image families keep the pixel area of a base_size x base_size square;
    video families treat base_size as the short edge. Sides are snapped to the
    family's multiple and kept inside its supported range.
    """
    spec = _family(model_type)
    mult = int(spec["multiple"])
    lo, hi = int(spec["min_size"]), int(spec["max_size"])
    num, den = ASPECT_RATIOS.get(aspect_ratio, (1, 1))
    ratio = num / den

    base = int(base_size) if base_size and int(base_size) > 0 else int(spec["default_size"])

    if spec.get("sizing") == "short_edge":
        short = max(lo, min(base, int(spec["default_size"])))
        long_side = min(hi, short * max(ratio, 1 / ratio))
        w, h = (long_side, short) if ratio >= 1 else (short, long_side)
    else:
        base = max(lo, min(base, hi))
        w = base * math.sqrt(ratio)
        h = base / math.sqrt(ratio)
        scale = min(1.0, hi / max(w, h))
        w, h = w * scale, h * scale

    def snap(x):
        return int(max(mult, min(hi, round(x / mult) * mult)))

    return snap(w), snap(h)


def _clean_names(value) -> list[str]:
    if isinstance(value, str):
        value = value.split(",")
    if not isinstance(value, (list, tuple)):
        return []
    return [str(v).strip() for v in value if str(v).strip()]


def _clamp(value, lo: float, hi: float, default: float) -> float:
    try:
        return max(lo, min(hi, float(value)))
    except (TypeError, ValueError):
        return default


def _plan_summary(plan: dict) -> str:
    """Short multi-line summary for a text preview node."""
    lines = [
        f"Model     {plan.get('checkpoint') or '(none)'}",
        f"Family    {plan.get('model_label', plan.get('model_type', ''))}",
        f"Task      {plan.get('task', '')}  |  denoise {plan.get('denoise')}",
        f"Size      {plan.get('width')} x {plan.get('height')}",
        f"Sampling  {plan.get('steps')} steps  |  cfg {plan.get('cfg')}  |  "
        f"{plan.get('sampler_name')} / {plan.get('scheduler')}",
        f"LoRAs     {', '.join(plan.get('loras') or []) or 'none'}",
        f"Planner   {plan.get('planner_source', '')}",
    ]
    for note in plan.get("planner_notes") or []:
        lines.append(f"Note      {note}")
    return "\n".join(lines)


def _run_planner(*, role: str, user_fields: dict, scoring_prompt: str, fallback_positive: str,
                 ollama_model: str, registry_path: str, task_hint: str, user_negative: str,
                 aspect_ratio: str, base_size: int, seed: int, max_vram: int,
                 ollama_host: str, ollama_port: int, image=None,
                 default_task: str = "text2img", default_denoise: float = 1.0) -> tuple:
    """
    Shared pipeline behind every planner node:
      1. narrow the registry to models that fit the task, VRAM and what is installed
      2. ask Ollama for a plan (falls back to tag scoring when Ollama is unavailable)
      3. validate the plan against the registry so a made-up name can never reach the loader
      4. fill in sampler settings, size and negative prompt from the registry
    """
    registry = _read_registry(registry_path)
    candidates, notes = _select_candidates(registry, task_hint, max_vram)
    candidate_types = {str(c.get("type", "")).lower() for c in candidates}
    is_video = bool(candidates) and all(_entry_media(c) == "video" for c in candidates)
    loras = [
        l for l in registry.get("loras", [])
        if l.get("enabled", True) is not False and str(l.get("type", "")).lower() in candidate_types
    ]

    plan = None
    source = "ollama"
    if candidates:
        system_prompt = role + " " + _PLAN_KEYS
        if image is not None:
            system_prompt += _DENOISE_RULES
        if is_video:
            system_prompt += _VIDEO_RULES
        payload = dict(user_fields)
        payload["task_hint"] = task_hint
        payload["user_negative"] = user_negative
        payload["registry"] = _compact_for_llm(candidates, loras)
        try:
            images = None
            if image is not None:
                encoded = _encode_image_base64(image)
                images = [encoded] if encoded else None
            raw = _ollama_chat(
                ollama_model, system_prompt, json.dumps(payload, ensure_ascii=True),
                images=images, host=ollama_host, port=ollama_port,
            )
            plan = _parse_plan(raw)
            if plan is None:
                notes.append("Ollama answered without valid JSON; used tag matching instead.")
        except Exception as exc:
            notes.append(_describe_ollama_error(exc, ollama_model, ollama_host, ollama_port))
            notes.append("Used tag matching instead.")

    if plan is None:
        plan = {}
        source = "tag matching"

    # --- checkpoint: must be one of the allowed candidates -------------------
    target = _find_entry(candidates, str(plan.get("checkpoint", "")))
    if target is None and candidates:
        if plan.get("checkpoint"):
            notes.append(f"Planner asked for unknown model '{plan.get('checkpoint')}'; picked the best tag match.")
        target = _best_by_tags(scoring_prompt, candidates)
    target = target or {}
    model_type = str(target.get("type") or "checkpoint").lower()
    spec = _family(model_type)

    # --- LoRAs: registry entries of the same family only ----------------------
    wanted = _clean_names(plan.get("loras"))
    raw_strengths = plan.get("lora_strengths")
    if isinstance(raw_strengths, str):
        raw_strengths = [s for s in raw_strengths.split(",") if s.strip()]
    if not isinstance(raw_strengths, (list, tuple)):
        raw_strengths = []
    if source != "ollama":
        wanted = [l.get("name") for l in loras if _lora_fits(l, model_type) and _score_tags(scoring_prompt, l.get("tags", [])) >= 5]
        raw_strengths = []
    chosen_loras, chosen_strengths = [], []
    for idx, name in enumerate(wanted):
        entry = _find_entry(loras, name)
        if entry is None or not _lora_fits(entry, model_type) or name in chosen_loras:
            continue
        default_strength = _clamp(entry.get("strength", 0.8), 0.0, 2.0, 0.8)
        strength = _clamp(raw_strengths[idx], 0.0, 2.0, default_strength) if idx < len(raw_strengths) else default_strength
        chosen_loras.append(name)
        chosen_strengths.append(round(strength, 2))
        if len(chosen_loras) == 3:
            break

    # --- sampler settings: always from the registry, never from the LLM -------
    steps = int(target.get("steps") or spec["steps"])
    cfg = float(target.get("cfg") if target.get("cfg") is not None else spec["cfg"])
    sampler_name = str(target.get("sampler_name") or spec["sampler_name"])
    scheduler = str(target.get("scheduler") or spec["scheduler"])
    if sampler_name not in comfy.samplers.KSampler.SAMPLERS:
        notes.append(f"Sampler '{sampler_name}' is not available in this ComfyUI; using euler.")
        sampler_name = "euler"
    if scheduler not in comfy.samplers.KSampler.SCHEDULERS:
        notes.append(f"Scheduler '{scheduler}' is not available in this ComfyUI; using simple.")
        scheduler = "simple"

    width, height = _compute_resolution(aspect_ratio, base_size, model_type)

    # --- prompts ---------------------------------------------------------------
    positive = str(plan.get("positive_prompt") or "").strip() or fallback_positive.strip()
    negative = ""
    if cfg > 1.0:
        parts = [str(plan.get("negative_prompt") or "").strip(), (user_negative or "").strip()]
        negative = ", ".join(p for p in parts if p) or str(target.get("negative_prompt") or "")
    # cfg <= 1 means a guidance-distilled model: the negative branch is never evaluated.

    task = str(plan.get("task") or default_task).lower()
    if task not in ("text2img", "img2img"):
        task = default_task
    if str(task_hint).lower() in ("text2img", "img2img"):
        task = str(task_hint).lower()
    if is_video:
        task = "text2video"
    denoise = 1.0 if task != "img2img" else _clamp(plan.get("denoise"), 0.05, 1.0, default_denoise)

    final = {
        "checkpoint": target.get("name", ""),
        "model_type": model_type,
        "model_label": target.get("title") or spec["label"],
        "task": task,
        "loras": chosen_loras,
        "lora_strengths": chosen_strengths,
        "positive_prompt": positive,
        "negative_prompt": negative,
        "steps": steps,
        "cfg": cfg,
        "sampler_name": sampler_name,
        "scheduler": scheduler,
        "width": width,
        "height": height,
        "denoise": round(denoise, 2),
        "seed": int(seed),
        "vae_name": str(target.get("required_vae") or ""),
        "clip_name": str(target.get("required_clip") or ""),
        "planner_source": f"{source} ({ollama_model})" if source == "ollama" else source,
        "planner_notes": notes,
    }
    for note in notes:
        _log(f"Planner: {note}")
    _log(f"Planner picked {final['checkpoint'] or '(nothing)'} [{model_type}] via {final['planner_source']}")

    return (
        final["checkpoint"],
        ",".join(chosen_loras),
        ",".join(str(s) for s in chosen_strengths),
        model_type,
        steps,
        cfg,
        sampler_name,
        scheduler,
        width,
        height,
        int(seed),
        positive,
        negative,
        json.dumps(final, ensure_ascii=False, indent=2),
        final["vae_name"],
        final["clip_name"],
        task,
        final["denoise"],
        _plan_summary(final),
    )


_PLANNER_RETURN_TYPES = (
    "STRING",  # checkpoint
    "STRING",  # loras
    "STRING",  # lora_strengths
    "STRING",  # model_type
    "INT",     # steps
    "FLOAT",   # cfg
    comfy.samplers.KSampler.SAMPLERS,    # sampler_name
    comfy.samplers.KSampler.SCHEDULERS,  # scheduler
    "INT",     # width
    "INT",     # height
    "INT",     # seed
    "STRING",  # positive_prompt
    "STRING",  # negative_prompt
    "STRING",  # plan_json
    "STRING",  # vae_name
    "STRING",  # clip_name
    "STRING",  # task
    "FLOAT",   # denoise
    "STRING",  # summary
)
_PLANNER_RETURN_NAMES = (
    "checkpoint",
    "loras",
    "lora_strengths",
    "model_type",
    "steps",
    "cfg",
    "sampler_name",
    "scheduler",
    "width",
    "height",
    "seed",
    "positive_prompt",
    "negative_prompt",
    "plan_json",
    "vae_name",
    "clip_name",
    "task",
    "denoise",
    "summary",
)
_PLANNER_OUTPUT_TOOLTIPS = (
    "File name of the chosen model. Connect to the Dynamic Checkpoint Loader.",
    "Comma-separated LoRA file names for the Dynamic LoRA Stack.",
    "Comma-separated strengths, one per LoRA.",
    "Model family of the chosen model (moss_image, flux3, krea2, z_image, minimax_h3, checkpoint).",
    "Sampling steps from the registry.",
    "CFG scale from the registry.",
    "Sampler from the registry.",
    "Scheduler from the registry.",
    "Width snapped to what the model family supports.",
    "Height snapped to what the model family supports.",
    "The seed set on this node, passed through.",
    "Prompt rewritten for the chosen model.",
    "Negative prompt. Empty for guidance-distilled models (cfg 1), which ignore it.",
    "The full plan as formatted JSON.",
    "VAE file the chosen model needs.",
    "Text encoder file the chosen model needs.",
    "text2img, img2img or text2video.",
    "Denoise strength: 1.0 for text-to-image, lower to keep more of an input image.",
    "Short readable summary of the plan for a Preview as Text node.",
)


def _planner_common_inputs(default_model: str) -> dict:
    """Inputs shared by all planner nodes, in widget order."""
    return {
        "ollama_model": ("STRING", {
            "default": default_model,
            "tooltip": "Ollama model tag to plan with. It must already be pulled (ollama pull <tag>).",
        }),
        "registry_path": ("STRING", {
            "default": DEFAULT_REGISTRY_FILE,
            "tooltip": "Model registry JSON. A relative path is resolved inside the ComfyUI-Cluster folder.",
        }),
        "task_hint": (TASK_HINTS, {
            "default": "auto",
            "tooltip": "auto picks any image model. Choose a family to pin it, or video for video models.",
        }),
        "user_negative": ("STRING", {
            "multiline": True, "default": "",
            "tooltip": "Added to the negative prompt. Ignored by guidance-distilled models (cfg 1).",
        }),
        "aspect_ratio": (list(ASPECT_RATIOS.keys()), {"default": "1:1"}),
        "base_size": ("INT", {
            "default": 1024, "min": 256, "max": 2048, "step": 64,
            "tooltip": "Image models: the picture has about base_size x base_size pixels at any aspect ratio. "
                       "Video models: the short edge.",
        }),
        "ollama_host": ("STRING", {"default": "localhost"}),
        "ollama_port": ("INT", {"default": 11434, "min": 1, "max": 65535}),
        "max_vram": (VRAM_OPTIONS, {
            "default": 24,
            "tooltip": "Your GPU memory in GB. Models that need more are not offered to the planner.",
        }),
        "seed": ("INT", {
            "default": 0, "min": 0, "max": MAX_SEED, "control_after_generate": True,
            "tooltip": "Passed through to the seed output. Changing it does not call Ollama again.",
        }),
    }


class OllamaPromptPlanner:
    """Pick a model, LoRAs and sampler settings for a text prompt using a local Ollama model."""

    DESCRIPTION = (
        "Geekatplay Studio. Reads your prompt, picks the best installed model and LoRAs from the "
        "registry with a local Ollama model, and outputs every setting the sampler needs."
    )

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "What you want to see."}),
        }
        required.update(_planner_common_inputs(DEFAULT_TEXT_MODEL))
        return {"required": required}

    RETURN_TYPES = _PLANNER_RETURN_TYPES
    RETURN_NAMES = _PLANNER_RETURN_NAMES
    OUTPUT_TOOLTIPS = _PLANNER_OUTPUT_TOOLTIPS
    FUNCTION = "plan"
    CATEGORY = f"{CATEGORY_ROOT}/Planner"

    def plan(self, prompt, ollama_model, registry_path, task_hint, user_negative, aspect_ratio, base_size,
             ollama_host="localhost", ollama_port=11434, max_vram=24, seed=0):
        return _run_planner(
            role="You are a routing planner for an image and video generation pipeline.",
            user_fields={"prompt": prompt},
            scoring_prompt=prompt,
            fallback_positive=prompt,
            ollama_model=ollama_model, registry_path=registry_path, task_hint=task_hint,
            user_negative=user_negative, aspect_ratio=aspect_ratio, base_size=base_size, seed=seed,
            max_vram=max_vram, ollama_host=ollama_host, ollama_port=ollama_port,
            default_task="text2img", default_denoise=1.0,
        )


class OllamaVisionStylePlanner:
    """Look at an image with a local Ollama vision model and plan an image-to-image pass."""

    DESCRIPTION = (
        "Geekatplay Studio. A local Ollama vision model studies the input image together with your "
        "prompt, then picks the model, LoRAs, denoise strength and sampler settings."
    )

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "image": ("IMAGE", {"tooltip": "Image the vision model looks at."}),
            "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "What to do with the image."}),
        }
        required.update(_planner_common_inputs(DEFAULT_VISION_MODEL))
        return {"required": required}

    RETURN_TYPES = _PLANNER_RETURN_TYPES
    RETURN_NAMES = _PLANNER_RETURN_NAMES
    OUTPUT_TOOLTIPS = _PLANNER_OUTPUT_TOOLTIPS
    FUNCTION = "plan"
    CATEGORY = f"{CATEGORY_ROOT}/Planner"

    def plan(self, image, prompt, ollama_model, registry_path, task_hint, user_negative, aspect_ratio, base_size,
             ollama_host="localhost", ollama_port=11434, max_vram=24, seed=0):
        return _run_planner(
            role=(
                "You are a routing planner for an image generation pipeline. Study the attached image "
                "(subject, composition, lighting, style) and plan how to re-render it as the prompt asks."
            ),
            user_fields={"prompt": prompt},
            scoring_prompt=prompt,
            fallback_positive=prompt,
            ollama_model=ollama_model, registry_path=registry_path, task_hint=task_hint,
            user_negative=user_negative, aspect_ratio=aspect_ratio, base_size=base_size, seed=seed,
            max_vram=max_vram, ollama_host=ollama_host, ollama_port=ollama_port, image=image,
            default_task="img2img", default_denoise=0.7,
        )


class OllamaVisionDualPlanner:
    """Vision planner with two prompts: one tells the model what to look for, one what to create."""

    DESCRIPTION = (
        "Geekatplay Studio. Two prompts: the vision prompt tells the Ollama vision model what to "
        "look for in the image, the generation prompt says what to create from it."
    )

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "image": ("IMAGE", {"tooltip": "Image the vision model looks at."}),
            "vision_prompt": ("STRING", {
                "multiline": True,
                "default": "Describe the style, composition, lighting, and key elements of this image.",
                "tooltip": "What the vision model should pay attention to in the image.",
            }),
            "generation_prompt": ("STRING", {
                "multiline": True, "default": "",
                "tooltip": "What to create, using what was seen in the image.",
            }),
        }
        required.update(_planner_common_inputs(DEFAULT_VISION_MODEL))
        return {"required": required}

    RETURN_TYPES = _PLANNER_RETURN_TYPES
    RETURN_NAMES = _PLANNER_RETURN_NAMES
    OUTPUT_TOOLTIPS = _PLANNER_OUTPUT_TOOLTIPS
    FUNCTION = "plan"
    CATEGORY = f"{CATEGORY_ROOT}/Planner"

    def plan(self, image, vision_prompt, generation_prompt, ollama_model, registry_path, task_hint, user_negative,
             aspect_ratio, base_size, ollama_host="localhost", ollama_port=11434, max_vram=24, seed=0):
        return _run_planner(
            role=(
                "You are an art director planning an image generation pipeline. First study the attached "
                "image as vision_instruction asks. Then plan the generation that generation_request asks "
                "for, carrying over what you saw in the image."
            ),
            user_fields={"vision_instruction": vision_prompt, "generation_request": generation_prompt},
            scoring_prompt=generation_prompt,
            fallback_positive=generation_prompt,
            ollama_model=ollama_model, registry_path=registry_path, task_hint=task_hint,
            user_negative=user_negative, aspect_ratio=aspect_ratio, base_size=base_size, seed=seed,
            max_vram=max_vram, ollama_host=ollama_host, ollama_port=ollama_port, image=image,
            default_task="img2img", default_denoise=0.7,
        )


# ---------------------------------------------------------------------------
# 5. Loader nodes
# ---------------------------------------------------------------------------
def _find_file(folder_type: str, filename: str, custom_path: str = "") -> tuple[str | None, str | None]:
    """
    Locate a model file. Order: ComfyUI path mapping, case-insensitive walk of the
    configured folders, then the optional custom folder. Returns (path, folder type).
    """
    if not filename:
        return None, None
    path, found_in = _quick_find(folder_type, filename)
    if path:
        return path, found_in

    wanted = filename.lower()
    for ft in _search_folders(folder_type):
        try:
            roots = folder_paths.get_folder_paths(ft)
        except Exception:
            roots = []
        for root in roots:
            if not root or not os.path.isdir(root):
                continue
            for current, _, files in os.walk(root):
                for f in files:
                    if f.lower() == wanted:
                        return os.path.join(current, f), ft

    if custom_path and os.path.isdir(custom_path):
        direct = os.path.join(custom_path, filename)
        if os.path.isfile(direct):
            return direct, folder_type
        for ft in _search_folders(folder_type):
            sub = os.path.join(custom_path, ft, filename)
            if os.path.isfile(sub):
                return sub, ft
        for current, _, files in os.walk(custom_path):
            for f in files:
                if f.lower() == wanted:
                    parent = os.path.basename(current).lower()
                    return os.path.join(current, f), (parent if parent in _FOLDER_SEARCH else folder_type)
    return None, None


def _download_hint(registry: dict, section: str, filename: str) -> str:
    entry = _find_entry(registry.get(section, []), filename)
    if entry and entry.get("url"):
        return f"\n    {filename}\n      -> {entry['url']}"
    return f"\n    {filename}"


class DynamicCheckpointLoader:
    """
    Universal loader. Loads either an all-in-one checkpoint or a split model
    (diffusion model + text encoder + VAE as separate files), using the registry to
    find the companion files and the right text encoder type. If the requested
    model is missing it falls back to another installed model of the same family.
    """

    DESCRIPTION = (
        "Geekatplay Studio. Loads the model the planner picked. Works with all-in-one checkpoints and "
        "with split models (MOSS-Image, FLUX.3, Krea-2, Z-Image, MiniMax H3): the text encoder and VAE "
        "named in the registry are loaded automatically."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "checkpoint": ("STRING", {
                    "default": "",
                    "tooltip": "Model file name. Usually connected from a planner's checkpoint output.",
                }),
                "model_type": (LOADER_MODEL_TYPES, {
                    "default": "auto",
                    "tooltip": "auto reads the family from the registry. Set it only to force a family.",
                }),
                "registry_path": ("STRING", {"default": DEFAULT_REGISTRY_FILE}),
                "custom_path": ("STRING", {
                    "default": "",
                    "tooltip": "Optional extra folder to search for model files, for example an external drive.",
                }),
            },
            "optional": {
                "vae_override": ("STRING", {
                    "default": "",
                    "tooltip": "VAE file to use when the registry does not name one.",
                }),
                "clip_override": ("STRING", {
                    "default": "",
                    "tooltip": "Text encoder file(s), comma-separated, to use when the registry does not name any.",
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE", "debug_info")
    FUNCTION = "load"
    CATEGORY = f"{CATEGORY_ROOT}/Loader"

    def load(self, checkpoint, model_type, registry_path, custom_path="", vae_override="", clip_override=""):
        registry = _read_registry(registry_path)
        checkpoints = registry.get("checkpoints", [])
        log = ["--- Request ---", f"Model: '{checkpoint}'", f"Type: {model_type}"]
        if custom_path:
            log.append(f"Custom path: {custom_path}")

        info = _find_entry(checkpoints, checkpoint)
        resolved_type = model_type if model_type != "auto" else str((info or {}).get("type") or "checkpoint")
        preferred_folder = (info or {}).get("folder_type") or "checkpoints"

        # --- 1. find the file, or a same-family fallback -----------------------
        log.append("--- Resolve ---")
        path, found_in = _find_file(preferred_folder, checkpoint, custom_path)
        if path:
            log.append(f"Found: {path}")
        else:
            log.append(f"Not installed: '{checkpoint}'")
            _log(f"Missing model: {checkpoint}", leading_newline=True)
            fallback = self._find_fallback(checkpoints, info, resolved_type, custom_path, log)
            if fallback is None:
                wanted = _download_hint(registry, "checkpoints", checkpoint) if checkpoint else ""
                raise FileNotFoundError(
                    f"{BRAND_NAME} loader: model '{checkpoint}' is not installed and no other "
                    f"'{resolved_type}' model was found to use instead.\n"
                    f"Run install.bat (or download it into ComfyUI/models/{preferred_folder}):{wanted}\n\n"
                    + "\n".join(log)
                )
            info, path, found_in = fallback
            if model_type == "auto":
                resolved_type = str(info.get("type") or "checkpoint")
            log.append(f"Fallback: using '{info.get('name')}' ({resolved_type}) at {path}")
            _log(f"Fallback to {info.get('name')}")

        info = info or {}
        is_split = found_in in SPLIT_FOLDERS or info.get("folder_type") in SPLIT_FOLDERS
        log.append(f"Family: {_family(resolved_type)['label']} ({resolved_type})")
        log.append(f"Layout: {'split files' if is_split else 'all-in-one checkpoint'}")

        # --- 2. load the main model --------------------------------------------
        log.append("--- Load ---")
        model, clip, vae = None, None, None
        if is_split:
            model = comfy.sd.load_diffusion_model(path)
            log.append("Diffusion model loaded.")
        else:
            model, clip, vae = comfy.sd.load_checkpoint_guess_config(
                path,
                output_vae=True,
                output_clip=True,
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
            )[:3]
            log.append("Checkpoint loaded.")

        # --- 3. companion files: the registry wins over the node inputs --------
        vae_name = str(info.get("required_vae") or vae_override or "").split(",")[0].strip()
        clip_names = _clean_names(info.get("required_clip") or clip_override)
        skip_words = ("default", "native", "none")
        missing = []

        if vae_name and vae_name.lower() not in skip_words:
            vae_path, _ = _find_file("vae", vae_name, custom_path)
            if vae_path:
                sd, metadata = comfy.utils.load_torch_file(vae_path, return_metadata=True)
                vae = comfy.sd.VAE(sd=sd, metadata=metadata)
                vae.throw_exception_if_invalid()
                log.append(f"VAE: {vae_path}")
            else:
                log.append(f"VAE not installed: {vae_name}")
                missing.append(("vae", vae_name))

        clip_names = [c for c in clip_names if c.lower() not in skip_words]
        if clip_names:
            clip_paths = []
            for name in clip_names:
                clip_path, _ = _find_file("text_encoders", name, custom_path)
                if clip_path:
                    clip_paths.append(clip_path)
                else:
                    log.append(f"Text encoder not installed: {name}")
                    missing.append(("text_encoders", name))
            if clip_paths and len(clip_paths) == len(clip_names):
                type_name = str(info.get("clip_type") or _family(resolved_type)["clip_type"])
                clip_type = getattr(comfy.sd.CLIPType, type_name.upper(), None)
                if clip_type is None:
                    raise RuntimeError(
                        f"{BRAND_NAME} loader: this ComfyUI has no text encoder type '{type_name}', which "
                        f"{info.get('name', checkpoint)} needs. Update ComfyUI to the latest version."
                    )
                clip = comfy.sd.load_clip(
                    ckpt_paths=clip_paths,
                    embedding_directory=folder_paths.get_folder_paths("embeddings"),
                    clip_type=clip_type,
                )
                log.append(f"Text encoder: {clip_paths} (type {type_name})")

        if missing:
            wanted = "".join(_download_hint(registry, section, name) for section, name in missing)
            raise FileNotFoundError(
                f"{BRAND_NAME} loader: '{info.get('name', checkpoint)}' needs companion files that are not "
                f"installed. Run install.bat or download them into ComfyUI/models/<folder>:{wanted}"
            )
        if clip is None or vae is None:
            parts = [n for n, v in (("text encoder", clip), ("VAE", vae)) if v is None]
            raise RuntimeError(
                f"{BRAND_NAME} loader: '{info.get('name', checkpoint)}' is a split model but the registry "
                f"names no {' or '.join(parts)} for it. Add required_clip / required_vae to its registry "
                f"entry, or fill in clip_override / vae_override on this node."
            )

        return (model, clip, vae, "\n".join(log))

    def _find_fallback(self, checkpoints, info, resolved_type, custom_path, log):
        """Find another installed model: same group first, then same family, then (auto only) anything of the same media."""
        missing_name = (info or {}).get("name")
        group = (info or {}).get("group")
        media = _entry_media(info) if info else _family(resolved_type)["media"]

        def usable(entry):
            return entry.get("enabled", True) is not False and entry.get("name") and entry.get("name") != missing_name

        tiers = [
            ("same group", lambda e: group and e.get("group") == group and e.get("type") == resolved_type),
            ("same family", lambda e: e.get("type") == resolved_type),
        ]
        if info is None or resolved_type == "checkpoint":
            tiers.append(("any installed model", lambda e: _entry_media(e) == media))

        for label, accept in tiers:
            for entry in checkpoints:
                if not usable(entry) or not accept(entry):
                    continue
                path, found_in = _find_file(entry.get("folder_type") or "checkpoints", entry["name"], custom_path)
                if path:
                    log.append(f"Fallback search ({label}): found '{entry['name']}'")
                    return entry, path, found_in
            log.append(f"Fallback search ({label}): nothing installed")
        return None


class DynamicLoraStack:
    """Apply a comma-separated list of LoRAs to a model/clip pair, skipping any that are not installed."""

    DESCRIPTION = (
        "Geekatplay Studio. Applies the LoRAs the planner chose. A LoRA that is not installed is "
        "skipped with a note in debug_info instead of stopping the workflow."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "loras": ("STRING", {"default": "", "tooltip": "Comma-separated LoRA file names."}),
                "lora_strengths": ("STRING", {"default": "", "tooltip": "Comma-separated strengths, one per LoRA."}),
                "custom_path": ("STRING", {"default": "", "tooltip": "Optional extra folder to search for LoRA files."}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "debug_info")
    FUNCTION = "apply"
    CATEGORY = f"{CATEGORY_ROOT}/Loader"

    def apply(self, model, clip, loras, lora_strengths, custom_path=""):
        names = _clean_names(loras)
        strengths = []
        for s in _clean_names(lora_strengths):
            strengths.append(_clamp(s, -4.0, 4.0, 0.8))

        log = ["--- LoRA stack ---"]
        if not names:
            log.append("No LoRAs requested.")
            return (model, clip, "\n".join(log))

        for idx, name in enumerate(names):
            strength = strengths[idx] if idx < len(strengths) else 0.8
            path, _ = _find_file("loras", name, custom_path)
            if not path:
                log.append(f"Skipped (not installed): {name}")
                _log(f"LoRA not installed, skipped: {name}")
                continue
            try:
                lora, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
                try:
                    model, clip = comfy.sd.load_lora_for_models(model, clip, lora, strength, strength, lora_metadata=metadata)
                except TypeError:
                    # ComfyUI builds without the lora_metadata argument
                    model, clip = comfy.sd.load_lora_for_models(model, clip, lora, strength, strength)
                log.append(f"Applied at {strength}: {name}")
            except Exception as exc:
                log.append(f"Failed to apply {name}: {exc}")
                _log(f"LoRA failed to load: {name}: {exc}")

        return (model, clip, "\n".join(log))


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------
class PreviewTextMerge:
    """Join up to eight text inputs into one preview, optionally passing them through."""

    DESCRIPTION = "Geekatplay Studio. Shows several text values in one place instead of many preview nodes."

    @classmethod
    def INPUT_TYPES(cls):
        optional = {f"text{i}": ("STRING", {"forceInput": True, "default": ""}) for i in range(1, 9)}
        return {
            "required": {
                "count": ("INT", {"default": 4, "min": 1, "max": 8, "tooltip": "How many of the text inputs to show."}),
                "separator": ("STRING", {"default": "\\n", "tooltip": "Placed between texts in the combined output."}),
                "emit_outputs": ("BOOLEAN", {"default": False, "tooltip": "Also pass the texts through to the outputs."}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("STRING",) * 9
    RETURN_NAMES = ("combined",) + tuple(f"text{i}" for i in range(1, 9))
    FUNCTION = "merge"
    CATEGORY = f"{CATEGORY_ROOT}/Status"
    OUTPUT_NODE = True

    def merge(self, count, separator, emit_outputs, **kwargs):
        n = max(1, min(8, int(count)))
        texts = ["" if kwargs.get(f"text{i}") is None else str(kwargs.get(f"text{i}")) for i in range(1, 9)]
        active = texts[:n]
        sep = separator.replace("\\n", "\n").replace("\\t", "\t")
        combined = sep.join(active)
        preview = "\n".join(f"[{i}] {t}" for i, t in enumerate(active, 1))
        result = (combined, *texts) if emit_outputs else tuple("" for _ in range(9))
        return {"ui": {"text": [preview]}, "result": result}


class LiveStatus:
    """Write a labelled status line to the ComfyUI console and pass it on as text."""

    DESCRIPTION = "Geekatplay Studio. Logs a labelled message to the console and outputs it as text."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stage": ("STRING", {"default": "stage", "tooltip": "Label put in front of the message."}),
                "message": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "emit"
    CATEGORY = f"{CATEGORY_ROOT}/Status"
    OUTPUT_NODE = True

    def emit(self, stage, message):
        text = f"[{stage}] {message}"
        _log(text)
        return {"ui": {"text": [text]}, "result": (text,)}
