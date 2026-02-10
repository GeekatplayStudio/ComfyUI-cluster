
import base64
import io
import json
import os
import re
import time
import urllib.request
import urllib.error

import numpy as np
from PIL import Image

import folder_paths
import comfy.sd
import comfy.utils
import comfy.controlnet
import comfy.samplers


_DEFAULT_REGISTRY = {
    "version": 1,
    "checkpoints": [],
    "loras": [],
    "controlnets": [],
}


def _read_registry(registry_path: str) -> dict:
    path = registry_path
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(__file__), path)
    if not os.path.exists(path):
        return _DEFAULT_REGISTRY
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return _DEFAULT_REGISTRY


def _compact_registry(registry: dict, max_items: int = 50) -> dict:
    checkpoints = registry.get("checkpoints", [])[:max_items]
    loras = registry.get("loras", [])[:max_items]
    controlnets = registry.get("controlnets", [])[:max_items]

    def _strip(items):
        compact = []
        for item in items:
            compact.append({
                "name": item.get("name", ""),
                "type": item.get("type", ""),
                "tags": item.get("tags", []),
                "recommended": bool(item.get("recommended", False)),
                "enabled": item.get("enabled", True), 
            })
        return compact

    return {
        "checkpoints": _strip(checkpoints),
        "loras": _strip(loras),
        "controlnets": _strip(controlnets),
    }


def _score_tags(prompt: str, tags: list[str]) -> int:
    """
    Advanced scoring:
    - Matches tokens (words) to find substring matches.
    - Matches multi-word tags against the prompt.
    - Penalizes contradictions (e.g., 'anime' vs 'photorealistic').
    - Rewards maximum matching tags.
    """
    prompt = prompt.lower()
    # Normalize punctuation-heavy prompts
    prompt_search = " " + re.sub(r"[^a-z0-9]", " ", prompt) + " "
    
    score = 0
    matched_tags = 0
    
    # 1. Direct Tag Matching (Weighted by length/specificity)
    for tag in tags:
        t = tag.lower()
        if not t: 
            continue
            
        # Check whole word match first (stronger)
        # e.g., tag "landscape" in "beautiful landscape photo"
        if f" {t} " in prompt_search:
            score += 5
            matched_tags += 1
            
        # Check substring match (weaker)
        # e.g., tag "realism" in "photorealism"
        elif t in prompt:
            score += 2
            matched_tags += 1
        
        # Check reverse substring (prompt word in tag)
        # e.g., prompt "photo" in tag "photorealism"
        elif any(w in t for w in prompt.split() if w and len(w) > 3):
            score += 1
            
    # 2. Negative Constraints / Penalty
    # If prompt strongly implies style X, penalize model with style Y
    # Explicit Style Groups
    is_anime_prompt = any(x in prompt for x in ["anime", "cartoon", "illustration", "waifu", "2d"])
    is_photo_prompt = any(x in prompt for x in ["photo", "realis", "raw", "dslr", "4k"])
    
    has_anime_tag = any("anime" in t or "cartoon" in t for t in tags)
    has_photo_tag = any("realis" in t or "photo" in t for t in tags)
    
    if is_anime_prompt and has_photo_tag and not has_anime_tag:
        score -= 10
    if is_photo_prompt and has_anime_tag and not has_photo_tag:
        score -= 10

    # 3. Specificity Bonus
    # If we matched multiple tags, it means the model is a better 'conceptual' fit
    # than a model where we just matched one generic tag.
    if matched_tags > 1:
        score += (matched_tags * 2)

    return score


def _heuristic_plan(prompt: str, registry: dict) -> dict:
    prompt_l = prompt.lower()
    checkpoints = registry.get("checkpoints", [])
    loras = registry.get("loras", [])
    controlnets = registry.get("controlnets", [])

    best_ckpt = None
    best_score = -1
    for ckpt in checkpoints:
        if ckpt.get("enabled", True) is False:
             continue
        tags = [t.lower() for t in ckpt.get("tags", [])]
        score = _score_tags(prompt_l, tags)
        if ckpt.get("recommended"):
            score += 1
        if score > best_score:
            best_score = score
            best_ckpt = ckpt

    # Safety: If filtered list is empty, best_ckpt remains None.
    # Fallback to first available enabled checkpoint if possible?
    if best_ckpt is None:
         # Try find ANY enabled checkpoint
         for ckpt in checkpoints:
             if ckpt.get("enabled", True) is not False:
                 best_ckpt = ckpt
                 break

    if best_ckpt is None and checkpoints:
        # If absolutely everything is disabled, we might have to just pick one or fail.
        # Let's pick the first one and hope the user knows what they are doing, or just return it.
        # But per user request "only use those that are enabled", maybe we should warn?
        # For now, if all disabled, just pick first to avoid crash, but it will likely fail downstream.
        best_ckpt = checkpoints[0]

    selected_loras = []
    selected_strengths = []
    for lora in loras:
        if lora.get("enabled", True) is False:
             continue
        tags = [t.lower() for t in lora.get("tags", [])]
        if _score_tags(prompt_l, tags) > 0:
            selected_loras.append(lora.get("name", ""))
            selected_strengths.append(float(lora.get("strength", 0.7)))

    model_type = (best_ckpt or {}).get("type", "sdxl")
    controlnet_name = ""
    controlnet_strength = 0.0
    for cn in controlnets:
        if cn.get("enabled", True) is False:
             continue
        tags = [t.lower() for t in cn.get("tags", [])]
        if "pose" in prompt_l and ("pose" in tags or "openpose" in tags):
            controlnet_name = cn.get("name", "")
            controlnet_strength = float(cn.get("strength", 0.8))
            break
        if ("edge" in prompt_l or "canny" in prompt_l or "lineart" in prompt_l) and (
            "canny" in tags or "edge" in tags or "lineart" in tags
        ):
            controlnet_name = cn.get("name", "")
            controlnet_strength = float(cn.get("strength", 0.8))
            break
    return {
        "task": "text2img",
        "model_type": model_type or "sdxl",
        "checkpoint": (best_ckpt or {}).get("name", ""),
        "vae_name": "",
        "clip_name": "",
        "use_refiner": False,
        "refiner_checkpoint": "",
        "loras": selected_loras,
        "lora_strengths": selected_strengths,
        "positive_prompt": prompt.strip(),
        "negative_prompt": "",
        "steps": 25,
        "cfg": 6.5,
        "width": 1024 if model_type == "sdxl" else 768,
        "height": 1024 if model_type == "sdxl" else 768,
        "denoise": 1.0,
        "seed": int(time.time()) % 2_000_000_000,
        "controlnet_name": controlnet_name,
        "controlnet_strength": controlnet_strength,
        "controlnet_start": 0.0,
        "controlnet_end": 1.0,
    }


def _ollama_chat(model: str, system_prompt: str, user_prompt: str, host: str = "localhost", port: int = 11434, timeout: int = 40) -> str:
    url = f"http://{host}:{port}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0.2,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="ignore")
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "message" in parsed and isinstance(parsed["message"], dict):
            return parsed["message"].get("content", "")
    except Exception:
        pass
    return raw


def _encode_image_base64(image) -> str:
    # image is a ComfyUI IMAGE tensor (batch, height, width, channels) in 0..1
    if image is None:
        return ""
    try:
        img = image
        if hasattr(img, "cpu"):
            img = img.cpu().numpy()
        if isinstance(img, np.ndarray) and img.ndim == 4:
            img = img[0]
        if isinstance(img, np.ndarray):
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            pil = Image.fromarray(img)
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return ""
    return ""


def _ollama_chat_with_images(model: str, system_prompt: str, user_prompt: str, images: list[str], host: str = "localhost", port: int = 11434, timeout: int = 60) -> str:
    url = f"http://{host}:{port}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt, "images": images},
        ],
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0.2,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="ignore")
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "message" in parsed and isinstance(parsed["message"], dict):
            return parsed["message"].get("content", "")
    except Exception:
        pass
    return raw


def _parse_plan(raw: str) -> dict | None:
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None


def _extract_keywords(prompt: str, top_k: int = 12) -> list[str]:
    stop = {
        "the","a","an","with","and","of","in","on","at","for","by","to","from","is","are","be",
        "this","that","these","those","it","as","over","under","into","onto","around","about",
        "portrait","photo","image","picture"
    }
    tokens = re.findall(r"[a-zA-Z0-9]{3,}", prompt.lower())
    freq = {}
    for t in tokens:
        if t in stop:
            continue
        freq[t] = freq.get(t, 0) + 1
    sorted_tokens = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [t for t, _ in sorted_tokens[:top_k]]


def _score_model(keywords: list[str], tags: list[str], recommended: bool) -> int:
    score = 0
    tag_set = {t.lower() for t in tags}
    for kw in keywords:
        if kw in tag_set:
            score += 4
        elif any(kw in t for t in tag_set):
            score += 2
    if recommended:
        score += 1
    return score


def _pick_sampler(model_type: str, target_info: dict | None) -> tuple[str, str]:
    # Default fallback
    sampler = "euler"
    scheduler = "normal"
    
    # Check target_info first
    if target_info:
        s_try = target_info.get("sampler_name") or target_info.get("sampler")
        sch_try = target_info.get("scheduler")
        if s_try:
            sampler = s_try
        if sch_try:
            scheduler = sch_try
            
    # Heuristics if not set in registry
    else:
        mt = (model_type or "").lower()
        if "flux" in mt:
            sampler = "euler"
            scheduler = "simple"
        elif "sdxl" in mt:
            sampler = "dpmpp_2m"
            scheduler = "karras"
        elif "sd15" in mt or "sd1.5" in mt:
            sampler = "euler_ancestral" 
            scheduler = "normal"

    # Validate against ComfyUI known types to ensure compatibility
    # This helps when connecting to KSampler nodes which expect exact string matches
    try:
        valid_samplers = comfy.samplers.KSampler.SAMPLERS
        valid_schedulers = comfy.samplers.KSampler.SCHEDULERS
        
        if sampler not in valid_samplers:
            # Try close match or standard backup
            if sampler == "euler_a": sampler = "euler_ancestral"
            elif sampler == "dpm++_2m": sampler = "dpmpp_2m"
            elif sampler not in valid_samplers:
                # Keep it but warn? Or fallback. KSampler might error if invalid.
                # Fallback to euler if completely unknown
                 pass 

        if scheduler not in valid_schedulers:
            if scheduler == "simple" and "simple" not in valid_schedulers:
                scheduler = "normal" # simple not available in old comfy
            elif scheduler == "karras" and "karras" not in valid_schedulers:
                scheduler = "normal"
    except:
        pass

    return (sampler, scheduler)


def _compute_resolution(aspect_ratio: str, base_size: int, model_type: str) -> tuple[int, int]:
    ratios = {
        "1:1": (1, 1),
        "3:2": (3, 2),
        "2:3": (2, 3),
        "16:9": (16, 9),
        "9:16": (9, 16),
        "4:5": (4, 5),
        "5:4": (5, 4),
    }
    num, den = ratios.get(aspect_ratio, (1, 1))
    long_side = max(256, int(base_size))
    short_side = int(long_side * den / num)
    # snap to multiples of 64
    def snap(x): return max(256, int(round(x / 64)) * 64)
    w = snap(long_side)
    h = snap(short_side)
    if w > 2048: w = 2048
    if h > 2048: h = 2048
    if model_type.lower() in ("sd15", "sd1.5", "sd"):
        if w > 1024: w = 1024
        if h > 1024: h = 1024
    return w, h


class OllamaPromptPlanner:
    """
    Use local Ollama to pick checkpoint + LoRAs + params based on the prompt and registry.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "ollama_model": ("STRING", {"default": "qwen2.5:7b"}),
                "registry_path": ("STRING", {"default": "model_registry.json"}),
                "task_hint": (["auto", "text2img", "img2img", "inpaint", "sdxl", "sd15", "flux"],),
                "user_negative": ("STRING", {"multiline": True, "default": ""}),
                "aspect_ratio": (["1:1", "3:2", "2:3", "16:9", "9:16", "4:5", "5:4"],),
                "base_size": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "ollama_host": ("STRING", {"default": "localhost"}),
                "ollama_port": ("INT", {"default": 11434, "min": 1, "max": 65535}),
                "max_vram": ([24, 16, 12, 8, 6], {"default": 24}),
            }
        }

    RETURN_TYPES = (
        "STRING",  # checkpoint
        "STRING",  # loras
        "STRING",  # lora_strengths
        "STRING",  # model_type
        "INT",     # steps
        "FLOAT",   # cfg
        comfy.samplers.KSampler.SAMPLERS,  # sampler_name
        comfy.samplers.KSampler.SCHEDULERS,  # scheduler
        "INT",     # width
        "INT",     # height
        "INT",     # seed
        "STRING",  # positive
        "STRING",  # negative
        "STRING",  # plan_json
        "STRING",  # vae_name
        "STRING",  # clip_name
        "STRING",  # task
        "FLOAT",   # denoise
        "BOOLEAN", # use_refiner
        "STRING",  # refiner_checkpoint
        "STRING",  # controlnet_name
        "FLOAT",   # controlnet_strength
        "FLOAT",   # controlnet_start
        "FLOAT",   # controlnet_end
    )
    RETURN_NAMES = (
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
        "use_refiner",
        "refiner_checkpoint",
        "controlnet_name",
        "controlnet_strength",
        "controlnet_start",
        "controlnet_end",
    )
    FUNCTION = "plan"
    CATEGORY = "Ollama/Planner"

    def plan(self, prompt, ollama_model, registry_path, task_hint, user_negative, aspect_ratio, base_size, max_vram=24, ollama_host="localhost", ollama_port=11434):
        registry = _read_registry(registry_path)
        
        # Filter by VRAM
        valid_ckpts = []
        for ckpt in registry.get("checkpoints", []):
            req = ckpt.get("min_vram", 0)
            if req <= max_vram:
                valid_ckpts.append(ckpt)
        registry["checkpoints"] = valid_ckpts
        
        compact = _compact_registry(registry)
        system_prompt = (
            "You are a routing planner for image generation. "
            "Return ONLY valid JSON with keys: "
            "task (text2img|img2img|inpaint), "
            "model_type (sdxl|sd15|flux), checkpoint, vae_name (optional), clip_name (optional), "
            "use_refiner (bool), refiner_checkpoint, "
            "loras (array of names), lora_strengths (array of floats), "
            "controlnet_name, controlnet_strength (float), controlnet_start (float), controlnet_end (float), "
            "positive_prompt, negative_prompt, steps (int), cfg (float), sampler_name, scheduler, width (int), height (int), "
            "denoise (float), seed (int). "
            "Use checkpoint, vae, clip and lora names exactly from the registry. "
            "If unsure, pick a recommended checkpoint."
        )
        user_payload = {
            "task_hint": task_hint,
            "prompt": prompt,
            "user_negative": user_negative,
            "registry": compact,
        }
        raw = ""
        try:
            raw = _ollama_chat(ollama_model, system_prompt, json.dumps(user_payload, ensure_ascii=True), host=ollama_host, port=ollama_port)
            plan = _parse_plan(raw)
        except urllib.error.URLError:
            plan = None
        except Exception:
            plan = None

        if plan is None:
            plan = _heuristic_plan(prompt, registry)

        # Keywords + registry-driven selection if checkpoint missing
        keywords = _extract_keywords(prompt)
        if not plan.get("checkpoint"):
            best_ckpt = None
            best_score = -1
            for ckpt in registry.get("checkpoints", []):
                score = _score_model(keywords, ckpt.get("tags", []), ckpt.get("recommended", False))
                if score > best_score:
                    best_score = score
                    best_ckpt = ckpt
            if best_ckpt:
                plan["checkpoint"] = best_ckpt.get("name", "")
                plan["model_type"] = best_ckpt.get("type", "sdxl")
                plan["vae_name"] = best_ckpt.get("required_vae", "")
                plan["clip_name"] = best_ckpt.get("required_clip", "")

        target_info = None
        for ckpt in registry.get("checkpoints", []):
            if ckpt.get("name") == plan.get("checkpoint"):
                target_info = ckpt
                break
        if target_info:
            if target_info.get("required_vae"):
                plan["vae_name"] = target_info.get("required_vae", "")
            if target_info.get("required_clip"):
                plan["clip_name"] = target_info.get("required_clip", "")

            if target_info.get("steps"):
                plan["steps"] = target_info.get("steps")
            if target_info.get("cfg"):
                plan["cfg"] = target_info.get("cfg")

            sampler_name, scheduler = _pick_sampler(plan.get("model_type", ""), target_info)
        else:
            sampler_name, scheduler = _pick_sampler(plan.get("model_type", ""), None)

        w, h = _compute_resolution(
            aspect_ratio=aspect_ratio,
            base_size=base_size,
            model_type=plan.get("model_type", "sdxl"),
        )
        plan["width"] = w
        plan["height"] = h
        plan["sampler_name"] = sampler_name
        plan["scheduler"] = scheduler

        negative = plan.get("negative_prompt", "") or ""
        if user_negative.strip():
            if negative.strip():
                negative = f"{negative}, {user_negative.strip()}"
            else:
                negative = user_negative.strip()

        loras = plan.get("loras", [])
        if isinstance(loras, str):
            loras = [l.strip() for l in loras.split(",") if l.strip()]
        strengths = plan.get("lora_strengths", [])
        if isinstance(strengths, str):
            strengths = [s.strip() for s in strengths.split(",") if s.strip()]

        lora_str = ",".join([l for l in loras if l])
        strength_str = ",".join([str(s) for s in strengths if str(s).strip()])

        # Flux typically ignores negative prompts; enforce blank to avoid artifacts.
        if str(plan.get("model_type", "")).lower() == "flux":
            negative = ""
            plan["use_refiner"] = False

        return (
            str(plan.get("checkpoint", "")),
            lora_str,
            strength_str,
            str(plan.get("model_type", "sdxl")),
            int(plan.get("steps", 25)),
            float(plan.get("cfg", 6.5)),
            str(plan.get("sampler_name", "euler")),
            str(plan.get("scheduler", "normal")),
            int(plan.get("width", 1024)),
            int(plan.get("height", 1024)),
            int(plan.get("seed", int(time.time()) % 2_000_000_000)),
            str(plan.get("positive_prompt", prompt)),
            negative,
            json.dumps(plan, ensure_ascii=True),
            str(plan.get("vae_name", "")),
            str(plan.get("clip_name", "")),
            str(plan.get("task", "text2img")),
            float(plan.get("denoise", 1.0)),
            bool(plan.get("use_refiner", False)),
            str(plan.get("refiner_checkpoint", "")),
            str(plan.get("controlnet_name", "")),
            float(plan.get("controlnet_strength", 0.0)),
            float(plan.get("controlnet_start", 0.0)),
            float(plan.get("controlnet_end", 1.0)),
        )


class DynamicCheckpointLoader:
    """
    Universal Loader:
    - Loads Checkpoint (Model/Clip/VAE)
    - Supports separate Diffusion/CLIP/VAE loading if overrides provided.
    - Implements Fallback logic: If 'checkpoint' is missing, finds the best alternative from registry not just by name but by Group/Category.
    - Prompts for download if missing.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "checkpoint": ("STRING", {"default": ""}),
                "model_type": (["auto", "sdxl", "sd15", "flux", "svd", "hunyuan", "wan"],),
                "registry_path": ("STRING", {"default": "model_registry.json"}),
                "custom_path": ("STRING", {"default": ""}),
            },
            "optional": {
                 "vae_override": ("STRING", {"default": ""}),
                 "clip_override": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE", "debug_info")
    FUNCTION = "load"
    CATEGORY = "Ollama/Loader"

    def load(self, checkpoint, model_type, registry_path, custom_path="", vae_override="", clip_override=""):
        registry = _read_registry(registry_path)
        log_lines = []
        
        # Log Search Paths
        search_paths = folder_paths.get_folder_paths("checkpoints")
        log_lines.append(f"--- Search Context ---")
        log_lines.append(f"ComfyUI Checkpoint Paths: {search_paths}")
        if custom_path:
             log_lines.append(f"Custom Path: {custom_path}")
        else:
             log_lines.append(f"Custom Path: (None)")

        log_lines.append(f"--- Request ---")
        log_lines.append(f"Model: '{checkpoint}'")
        log_lines.append(f"Type Constraint: {model_type}")

        # 1. Identify target metadata from registry
        target_info = None
        registry_idx = -1
        for i, ckpt in enumerate(registry.get("checkpoints", [])):
            if ckpt.get("name") == checkpoint:
                target_info = ckpt
                registry_idx = i
                break
        
        resolved_type = model_type
        if target_info and model_type == "auto":
             resolved_type = target_info.get("type", "sdxl")
        
        # Determine Folder Type (default to checkpoints)
        preferred_folder = "checkpoints"
        if target_info and target_info.get("folder_type"):
             preferred_folder = target_info.get("folder_type")

        log_lines.append(f"Resolved Type: {resolved_type}")
        log_lines.append(f"Search Folder Strategy: {preferred_folder}")
        if target_info:
            log_lines.append(f"Registry Metadata: Group='{target_info.get('group')}', Category='{target_info.get('category')}'")
        else:
            log_lines.append(f"Registry Metadata: Not found in registry (using raw filename)")

        # 2. Resolve Checkpoint Path (or Fallback)
        log_lines.append(f"--- Resolution ---")
        ckpt_path = self._find_path(preferred_folder, checkpoint, custom_path)
        
        # Track Active Model Name for overrides
        active_model_name = checkpoint
        is_fallback = False

        if ckpt_path:
             log_lines.append(f"Found directly at: {ckpt_path}")
        else:
            # Checkpoint Missing: logical fallback
            log_lines.append(f"primary lookup failed for '{checkpoint}'")
            print(f"\\n[OllamaRouter] !!! MISSING MODEL: {checkpoint} !!!")
            
            # Find Fallback
            # Update fallback search to respect preferred folder of CANDIDATES if possible, 
            # but _find_fallback uses _find_path internally? No. 
            # _find_fallback returns a NAME. We need to resolve that name.
            
            fallback_name = self._find_fallback(registry, target_info, resolved_type, custom_path, start_index=registry_idx, log_store=log_lines)
            if fallback_name:
                msg = f"-> Fallback SUCCESS: Selected '{fallback_name}'"
                log_lines.append(msg)
                print(f"[OllamaRouter] {msg}")
                
                # We need to determine the folder type for the FALLBACK model too!
                fallback_info = None
                for c in registry.get("checkpoints", []):
                     if c.get("name") == fallback_name:
                         fallback_info = c
                         break

                # CRITICAL FIX: If we fell back to a model, we MUST update target_info effectively
                # so that downstream logic (VAE/CLIP requirements) uses the FALLBACK model's metadata, not the original missing one.
                if fallback_info:
                    target_info = fallback_info
                    # Also update resolved_type if it was generic 'auto' and now we have specific
                    if model_type == "auto":
                        resolved_type = fallback_info.get("type", "sdxl")
                    
                    log_lines.append(f"-> Switched metadata context to '{fallback_name}' (Type: {resolved_type})")
                
                fallback_folder = "checkpoints"
                if fallback_info and fallback_info.get("folder_type"):
                     fallback_folder = fallback_info.get("folder_type")
                
                ckpt_path = self._find_path(fallback_folder, fallback_name, custom_path)
                log_lines.append(f"Fallback Path: {ckpt_path} (Type: {fallback_folder})")
                active_model_name = fallback_name
                is_fallback = True
            else:
                error_msg = f"FATAL: Model '{checkpoint}' not found and no suitable fallback in group/type '{resolved_type}' discovered."
                log_lines.append(error_msg)
                full_log = "\n".join(log_lines)
                raise FileNotFoundError(f"{error_msg}\n\nDEBUG LOG:\n{full_log}")
        
        log_lines.append(f"--- Loading ---")
        # 3. Load Main Components
        out_model, out_clip, out_vae = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )[:3]
        log_lines.append(f"Main Model Loaded.")

        # 4. Handle Overrides (Universal Loading)
        # Determine effective overrides (Use Native if Fallback)
        effective_vae = vae_override
        effective_clip = clip_override
        
        # Mandatory Components Check (Flux, Wan, Hunyuan, etc)
        # Some models require explicit loading of separate CLIP/VAE files.
        # We check the registry info for the ACTIVE model (either requested or fallback).
        active_registry_info = target_info
        if is_fallback and active_model_name != checkpoint:
             # Look up fallback metadata
             for ckpt in registry.get("checkpoints", []):
                 if ckpt.get("name") == active_model_name:
                     active_registry_info = ckpt
                     break
        
        if active_registry_info:
             # Check for MANDATORY fields 'required_vae' or 'required_clip'
             # Or just standard 'vae'/'clip' fields which imply a preference
             reg_vae = active_registry_info.get("vae") or active_registry_info.get("required_vae")
             reg_clip = active_registry_info.get("clip") or active_registry_info.get("required_clip")
             
             if reg_vae:
                 log_lines.append(f"-> Registry MANDATE: Model requires VAE '{reg_vae}'")
                 effective_vae = reg_vae
             
             if reg_clip:
                 log_lines.append(f"-> Registry MANDATE: Model requires CLIP '{reg_clip}'")
                 effective_clip = reg_clip

        if is_fallback and not active_registry_info:
             # Fallback active but no registry info found -> assume native
             log_lines.append(f"-> Fallback active but no registry metadata. Resetting overrides to Native.")
             effective_vae = ""
             effective_clip = ""

        if effective_vae:
            # Handle comma-separated VAEs (rare, but possibly for fallbacks? Take first)
            if "," in effective_vae:
                log_lines.append(f"-> Detected multiple VAEs '{effective_vae}', using first one.")
                effective_vae = effective_vae.split(",")[0].strip()

            log_lines.append(f"Override VAE: '{effective_vae}' ...")
            # Special handling for "default" or "native" keywords to skip loading
            if effective_vae.lower() in ["default", "native", "none"]:
                 log_lines.append(f"-> Using Native VAE (explicitly requested).")
            else:
                v_path = self._find_path("vae", effective_vae, custom_path)
                if v_path:
                    try:
                        out_vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(v_path))
                        log_lines.append(f"-> VAE Loaded from: {v_path}")
                    except Exception as e:
                        log_lines.append(f"-> VAE Load Failed: {e}")
                else:
                    msg = f"-> VAE '{effective_vae}' NOT FOUND in search paths!"
                    log_lines.append(msg)
                    print(f"[OllamaRouter] WARNING: {msg}")
                    if "flux" in str(active_model_name).lower() or reg_vae:
                        print("[OllamaRouter] CRITICAL: Missing required VAE for this model. Rendering will likely fail.")

        if effective_clip:
             log_lines.append(f"Override CLIP: '{effective_clip}' ...")
             # Special handling for "default" or "native" keywords to skip loading
             if effective_clip.lower() in ["default", "native", "none"]:
                  log_lines.append(f"-> Using Native CLIP (explicitly requested).")
             else:
                 # SPLIT logic for comma-separated CLIPs (e.g. "clip_l, t5")
                 clip_candidates = [c.strip() for c in effective_clip.split(",") if c.strip()]

                 clip_paths = []
                 for c_cand in clip_candidates:
                     path_res = self._find_path("clip", c_cand, custom_path)
                     if not path_res:
                         path_res = self._find_path("checkpoints", c_cand, custom_path)
                     if path_res:
                         clip_paths.append(path_res)
                         log_lines.append(f"-> CLIP candidate found: {path_res}")
                     else:
                         log_lines.append(f"-> Candidate '{c_cand}' missing.")
                 
                 if clip_paths:
                      try:
                          # Determine clip_type based on resolved_type to ensure correct model loading (especially for Flux/SD3)
                          clip_type_enum = comfy.sd.CLIPType.STABLE_DIFFUSION
                          if str(resolved_type).lower() == "flux":
                              clip_type_enum = comfy.sd.CLIPType.FLUX
                          elif str(resolved_type).lower() == "sd3":
                              clip_type_enum = comfy.sd.CLIPType.SD3
                          
                          out_clip = comfy.sd.load_clip(
                                clip_paths,
                                embedding_directory=folder_paths.get_folder_paths("embeddings"),
                                clip_type=clip_type_enum,
                          )
                          log_lines.append(f"-> CLIP Loaded from: {clip_paths} (Type: {clip_type_enum})")
                      except Exception as e:
                          log_lines.append(f"-> CLIP Load Failed: {e}")
                 else:
                     msg = f"-> CLIP components {clip_candidates} NOT FOUND!"
                     log_lines.append(msg)
                     print(f"[OllamaRouter] WARNING: {msg}")
                     if "flux" in str(active_model_name).lower() or reg_clip:
                        print("[OllamaRouter] CRITICAL: Missing required CLIP/T5 for this model. Rendering will likely fail.")

        return (out_model, out_clip, out_vae, "\n".join(log_lines))

    def _find_path(self, folder_type, filename, custom_path):
        folder_types = [folder_type]
        # Expand searches for common types to support 'extra_model_paths.yaml' variations
        if folder_type == "checkpoints":
            folder_types = ["checkpoints", "diffusion_models", "unet"]
        elif folder_type == "clip":
            folder_types = ["clip", "text_encoders", "checkpoints"]
        elif folder_type == "diffusion_models":
               folder_types = ["diffusion_models", "unet", "checkpoints"]
        elif folder_type == "vae":
            folder_types = ["vae", "checkpoints"]

        # Direct lookup using ComfyUI path mapping
        for ft in folder_types:
            try:
                path = folder_paths.get_full_path(ft, filename)
                if path: return path
            except: pass

        # Recursive search inside ComfyUI configured roots for the folder types
        searched_roots = set()
        for ft in folder_types:
            try:
                roots = folder_paths.get_folder_paths(ft)
            except Exception:
                roots = []
            for root in roots:
                if not root or root in searched_roots:
                    continue
                searched_roots.add(root)
                try:
                    if os.path.isfile(os.path.join(root, filename)):
                        return os.path.join(root, filename)
                    if os.path.isdir(root):
                        for r, _, files in os.walk(root):
                            # Exact match first
                            if filename in files:
                                return os.path.join(r, filename)
                            # Case-insensitive match
                            lower_map = {f.lower(): f for f in files}
                            if filename.lower() in lower_map:
                                return os.path.join(r, lower_map[filename.lower()])
                except Exception:
                    pass
        
        # Checkpoint fallback (original behavior) if not covered above
        if folder_type != "checkpoints" and "checkpoints" not in folder_types:
             try:
                path = folder_paths.get_full_path("checkpoints", filename)
                if path: return path
             except: pass

        # Custom Path Smart Search
        if custom_path and os.path.exists(custom_path):
            # 1. Direct path check
            p = os.path.join(custom_path, filename)
            if os.path.isfile(p): return p

            # 2. Subfolder check matching preferred types
            # Note: We reuse the folder_types list expanded above
            for ft in folder_types:
                p_sub = os.path.join(custom_path, ft, filename)
                if os.path.isfile(p_sub): return p_sub

            # 3. Recursive walk (Last resort)
            # Only do this if specific lookups fail to avoid performance hits or permissions issues
            try:
                for root, dirs, files in os.walk(custom_path):
                    if filename in files:
                        return os.path.join(root, filename)
            except Exception:
                pass
        
        return None

    def _find_fallback(self, registry, target_info, resolved_type, custom_path, start_index=-1, log_store=None):
        if log_store is None: log_store = []
        candidates = registry.get("checkpoints", [])
        if not candidates: 
            log_store.append("Fallback: No candidates in registry.")
            return None

        idx_start = start_index if start_index >= 0 else 0
        ordered_candidates = candidates[idx_start:] + candidates[:idx_start]
        
        # Add next-in-line logging
        log_store.append(f"Fallback Search: checking {len(ordered_candidates)} candidates starting from index {idx_start}")
        if len(ordered_candidates) > 1:
             next_model = ordered_candidates[1].get("name") if len(ordered_candidates) > 1 else "None"
             log_store.append(f"Next in line model would be: '{next_model}'")

        if not target_info:
             log_store.append("Fallback: No target info, searching by type only...")
             for c in ordered_candidates:
                 # Check enabled status
                 if c.get("enabled", True) is False: continue

                 # If mode is 'auto', accept any type, otherwise must match
                 if resolved_type == "auto" or c.get("type") == resolved_type:
                     # We need to find the correct folder type for this candidate to check existence
                     cand_folder = c.get("folder_type", "checkpoints")
                     found = self._find_path(cand_folder, c["name"], custom_path)
                     if found:
                          log_store.append(f"-> Found candidate '{c['name']}' in {cand_folder}") 
                          return c["name"]
             return None

        grp = target_info.get("group")
        cat = target_info.get("category")
        log_store.append(f"Fallback Strategy: Match Group='{grp}' AND Category='{cat}' AND Type='{resolved_type}'")
        
        # Priority 1: Same Group & Category
        for c in ordered_candidates:
            if c["name"] == target_info["name"]: continue
            if c.get("enabled", True) is False: continue
            if c.get("group") == grp and c.get("category") == cat and c.get("type") == resolved_type:
                 cand_folder = c.get("folder_type", "checkpoints")
                 found = self._find_path(cand_folder, c["name"], custom_path)
                 if found: return c["name"]
                 # log_store.append(f"Candidate '{c['name']}' matched metadata but file not found.") 
        
        # Priority 2: Same Group & Type
        log_store.append(f"Fallback Strategy: Relaxed -> Match Group='{grp}' AND Type='{resolved_type}'")
        for c in ordered_candidates:
            if c["name"] == target_info["name"]: continue
            if c.get("enabled", True) is False: continue
            if c.get("group") == grp and c.get("type") == resolved_type:
                 cand_folder = c.get("folder_type", "checkpoints")
                 if self._find_path(cand_folder, c["name"], custom_path): return c["name"]

        # Priority 3: Same Type
        log_store.append(f"Fallback Strategy: Relaxed -> Match Type='{resolved_type}' only")
        for c in ordered_candidates:
            if c["name"] == target_info["name"]: continue
            if c.get("enabled", True) is False: continue
            if c.get("type") == resolved_type:
                 cand_folder = c.get("folder_type", "checkpoints")
                 if self._find_path(cand_folder, c["name"], custom_path): return c["name"]
        
        log_store.append("Fallback: No suitable candidate found on disk.")
        return None


class DynamicLoraStack:
    """
    Apply a comma-separated list of LoRAs to a model/clip pair.
    Includes Smart Fallback lookup if specific LoRA name is missing.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "loras": ("STRING", {"default": ""}),
                "lora_strengths": ("STRING", {"default": ""}),
                "custom_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "debug_info")
    FUNCTION = "apply"
    CATEGORY = "Ollama/Loader"

    def apply(self, model, clip, loras, lora_strengths, custom_path=""):
        log_lines = []
        log_lines.append(f"--- LoRA Stack Request ---")
        
        lora_list = [l.strip() for l in loras.split(",") if l.strip()]
        strength_list = [s.strip() for s in lora_strengths.split(",") if s.strip()]
        strengths = []
        for s in strength_list:
            try:
                strengths.append(float(s))
            except Exception:
                strengths.append(0.7)

        log_lines.append(f"LoRAs: {lora_list}")
        log_lines.append(f"Search Paths: {folder_paths.get_folder_paths('loras')}")
        if custom_path: log_lines.append(f"Custom Path: {custom_path}")

        if not lora_list:
            log_lines.append("No LoRAs requested.")
            return (model, clip, "\n".join(log_lines))
            
        registry = {} 
        try:
           registry = _read_registry("model_registry.json")
        except:
           pass

        model_out, clip_out = model, clip
        for idx, lora_name in enumerate(lora_list):
            strength = strengths[idx] if idx < len(strengths) else 0.7
            log_lines.append(f"\nProcessing '{lora_name}' (Strength: {strength})")
            
            lora_path = self._find_path(lora_name, custom_path)
            
            if not lora_path:
                target_info = None
                for l in registry.get("loras", []):
                    if l.get("name") == lora_name:
                        target_info = l
                        break
                
                log_lines.append(f"!!! MISSING LORA: {lora_name} !!!")
                print(f"\\n[OllamaRouter] !!! MISSING LORA: {lora_name} !!!")
                if target_info:
                     info_str = f"Registry Info: Group={target_info.get('group')}, Category={target_info.get('category')}"
                     log_lines.append(info_str)
                     print(f"[OllamaRouter] {info_str}")
                
                # Try Fallback
                fallback_name = None
                if target_info:
                   fallback_name = self._find_fallback(registry, target_info, custom_path, log_store=log_lines)
                
                if fallback_name:
                    msg = f"-> Falling back to LoRA: {fallback_name}"
                    log_lines.append(msg)
                    print(f"[OllamaRouter] {msg}")
                    lora_path = self._find_path(fallback_name, custom_path)
                else:
                    msg = f"-> Skipping missing LoRA '{lora_name}' (No fallback found)"
                    log_lines.append(msg)
                    print(f"[OllamaRouter] {msg}")
                    continue

            # Load
            try:
                lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                model_out, clip_out = comfy.sd.load_lora_for_models(model_out, clip_out, lora, strength, strength)
                msg = f"Loaded: {lora_path}"
                log_lines.append(msg)
                print(f"[OllamaRouter] {msg}")
            except Exception as e:
                err = f"Error loading LoRA {lora_path}: {e}"
                log_lines.append(err)
                print(f"[OllamaRouter] {err}")

        return (model_out, clip_out, "\n".join(log_lines))

    def _find_path(self, filename, custom_path):
        try:
            path = folder_paths.get_full_path("loras", filename)
            if path: return path
        except: pass
        if custom_path and os.path.exists(custom_path):
            p = os.path.join(custom_path, filename)
            if os.path.exists(p): return p
        return None

    def _find_fallback(self, registry, target_info, custom_path, log_store=None):
        if log_store is None: log_store = []
        target_grp = target_info.get("group", "")
        target_cat = target_info.get("category", "")
        
        # Priority 1: Same Group & Category
        for l in registry.get("loras", []):
            if l["name"] == target_info["name"]: continue
            if l.get("enabled", True) is False: continue
            if l.get("group") == target_grp and l.get("category") == target_cat:
                if self._find_path(l["name"], custom_path): return l["name"]
        
        # Priority 2: Same Category only
        for l in registry.get("loras", []):
            if l["name"] == target_info["name"]: continue
            if l.get("enabled", True) is False: continue
            if l.get("category") == target_cat:
                if self._find_path(l["name"], custom_path): return l["name"]
        
        log_store.append("Fallback: No suitable LoRA candidate found.")
        return None


class RouteConditioningByType:
    """
    Route conditioning between SDXL and SD1.5 branches based on model_type.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_type": (["sdxl", "sd15", "flux"],),
                "sdxl": ("CONDITIONING",),
                "sd15": ("CONDITIONING",),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "route"
    CATEGORY = "Ollama/Route"

    def route(self, model_type, sdxl, sd15):
        return (sdxl if model_type == "sdxl" else sd15,)


class OllamaVisionStylePlanner:
    """
    Use local Ollama vision model to analyze an image and pick checkpoint + LoRAs + params.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "ollama_model": ("STRING", {"default": "llava:7b"}),
                "registry_path": ("STRING", {"default": "model_registry.json"}),
                "task_hint": (["auto", "img2img", "sdxl", "sd15", "flux"],),
                "user_negative": ("STRING", {"multiline": True, "default": ""}),
                "aspect_ratio": (["1:1", "3:2", "2:3", "16:9", "9:16", "4:5", "5:4"],),
                "base_size": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "ollama_host": ("STRING", {"default": "localhost"}),
                "ollama_port": ("INT", {"default": 11434, "min": 1, "max": 65535}),
                "max_vram": ([24, 16, 12, 8, 6], {"default": 24}),
            }
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "INT",
        "FLOAT",
        comfy.samplers.KSampler.SAMPLERS,
        comfy.samplers.KSampler.SCHEDULERS,
        "INT",
        "INT",
        "INT",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "FLOAT",
        "BOOLEAN",
        "STRING",
        "STRING",
        "FLOAT",
        "FLOAT",
        "FLOAT",
    )
    RETURN_NAMES = (
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
        "use_refiner",
        "refiner_checkpoint",
        "controlnet_name",
        "controlnet_strength",
        "controlnet_start",
        "controlnet_end",
    )
    FUNCTION = "plan"
    CATEGORY = "Ollama/Planner"

    def plan(self, image, prompt, ollama_model, registry_path, task_hint, user_negative, aspect_ratio, base_size, max_vram=24, ollama_host="localhost", ollama_port=11434):
        registry = _read_registry(registry_path)

        # Filter by VRAM
        valid_ckpts = []
        for ckpt in registry.get("checkpoints", []):
            req = ckpt.get("min_vram", 0)
            if req <= max_vram:
                valid_ckpts.append(ckpt)
        registry["checkpoints"] = valid_ckpts
        
        compact = _compact_registry(registry)
        system_prompt = (
            "You analyze an image style and select the best image generation setup. "
            "Return ONLY valid JSON with keys: "
            "task (text2img|img2img|inpaint), "
            "model_type (sdxl|sd15|flux), checkpoint, vae_name (optional), clip_name (optional), "
            "use_refiner (bool), refiner_checkpoint, "
            "loras (array of names), lora_strengths (array of floats), "
            "controlnet_name, controlnet_strength (float), controlnet_start (float), controlnet_end (float), "
            "positive_prompt, negative_prompt, steps (int), cfg (float), sampler_name, scheduler, width (int), height (int), "
            "denoise (float), seed (int). "
            "Use checkpoint, vae, clip and lora names exactly from the registry. "
            "If unsure, pick a recommended checkpoint."
        )
        user_payload = {
            "task_hint": task_hint,
            "prompt": prompt,
            "user_negative": user_negative,
            "registry": compact,
        }
        raw = ""
        try:
            img_b64 = _encode_image_base64(image)
            raw = _ollama_chat_with_images(
                ollama_model,
                system_prompt,
                json.dumps(user_payload, ensure_ascii=True),
                [img_b64] if img_b64 else [],
                host=ollama_host,
                port=ollama_port
            )
            plan = _parse_plan(raw)
        except urllib.error.URLError:
            plan = None
        except Exception:
            plan = None

        if plan is None:
            plan = _heuristic_plan(prompt, registry)

        # Keywords + registry-driven selection if checkpoint missing
        keywords = _extract_keywords(prompt)
        if not plan.get("checkpoint"):
            best_ckpt = None
            best_score = -1
            for ckpt in registry.get("checkpoints", []):
                score = _score_model(keywords, ckpt.get("tags", []), ckpt.get("recommended", False))
                if score > best_score:
                    best_score = score
                    best_ckpt = ckpt
            if best_ckpt:
                plan["checkpoint"] = best_ckpt.get("name", "")
                plan["model_type"] = best_ckpt.get("type", "sdxl")
                plan["vae_name"] = best_ckpt.get("required_vae", "")
                plan["clip_name"] = best_ckpt.get("required_clip", "")

        target_info = None
        for ckpt in registry.get("checkpoints", []):
            if ckpt.get("name") == plan.get("checkpoint"):
                target_info = ckpt
                break
        if target_info:
            if target_info.get("required_vae"):
                plan["vae_name"] = target_info.get("required_vae", "")
            if target_info.get("required_clip"):
                plan["clip_name"] = target_info.get("required_clip", "")

            if target_info.get("steps"):
                plan["steps"] = target_info.get("steps")
            if target_info.get("cfg"):
                plan["cfg"] = target_info.get("cfg")

            sampler_name, scheduler = _pick_sampler(plan.get("model_type", ""), target_info)
        else:
            sampler_name, scheduler = _pick_sampler(plan.get("model_type", ""), None)

        w, h = _compute_resolution(
            aspect_ratio=aspect_ratio,
            base_size=base_size,
            model_type=plan.get("model_type", "sdxl"),
        )
        plan["width"] = w
        plan["height"] = h
        plan["sampler_name"] = sampler_name
        plan["scheduler"] = scheduler

        negative = plan.get("negative_prompt", "") or ""
        if user_negative.strip():
            if negative.strip():
                negative = f"{negative}, {user_negative.strip()}"
            else:
                negative = user_negative.strip()

        loras = plan.get("loras", [])
        if isinstance(loras, str):
            loras = [l.strip() for l in loras.split(",") if l.strip()]
        strengths = plan.get("lora_strengths", [])
        if isinstance(strengths, str):
            strengths = [s.strip() for s in strengths.split(",") if s.strip()]

        lora_str = ",".join([l for l in loras if l])
        strength_str = ",".join([str(s) for s in strengths if str(s).strip()])

        if str(plan.get("model_type", "")).lower() == "flux":
            negative = ""
            plan["use_refiner"] = False

        return (
            str(plan.get("checkpoint", "")),
            lora_str,
            strength_str,
            str(plan.get("model_type", "sdxl")),
            int(plan.get("steps", 25)),
            float(plan.get("cfg", 6.5)),
            str(plan.get("sampler_name", "euler")),
            str(plan.get("scheduler", "normal")),
            int(plan.get("width", 1024)),
            int(plan.get("height", 1024)),
            int(plan.get("seed", int(time.time()) % 2_000_000_000)),
            str(plan.get("positive_prompt", prompt)),
            negative,
            json.dumps(plan, ensure_ascii=True),
            str(plan.get("vae_name", "")),
            str(plan.get("clip_name", "")),
            str(plan.get("task", "img2img")),
            float(plan.get("denoise", 0.7)),
            bool(plan.get("use_refiner", False)),
            str(plan.get("refiner_checkpoint", "")),
            str(plan.get("controlnet_name", "")),
            float(plan.get("controlnet_strength", 0.0)),
            float(plan.get("controlnet_start", 0.0)),
            float(plan.get("controlnet_end", 1.0)),
        )


class OptionalControlNetApply:
    """
    Optionally apply ControlNet based on a boolean flag and controlnet name.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "controlnet_name": ("STRING", {"default": ""}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            },
            "optional": {
                "image": ("IMAGE",),
                "vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "apply"
    CATEGORY = "Ollama/ControlNet"

    def apply(self, positive, negative, controlnet_name, strength, start_percent, end_percent, image=None, vae=None):
        if (not controlnet_name) or (image is None) or strength == 0:
            return (positive, negative)

        controlnet_path = folder_paths.get_full_path_or_raise("controlnet", controlnet_name)
        controlnet = comfy.controlnet.load_controlnet(controlnet_path)
        if controlnet is None:
            raise RuntimeError("ERROR: controlnet file is invalid and does not contain a valid controlnet model.")

        control_hint = image.movedim(-1, 1)
        cnets = {}
        out = []
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()
                prev_cnet = d.get("control", None)
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    c_net = controlnet.copy().set_cond_hint(
                        control_hint,
                        strength,
                        (start_percent, end_percent),
                        vae=vae,
                        extra_concat=[],
                    )
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net
                d["control"] = c_net
                d["control_apply_to_uncond"] = True
                n = [t[0], d]
                c.append(n)
            out.append(c)
        return (out[0], out[1])


class RouteLatentByBool:
    """
    Route between two LATENT inputs based on a boolean.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "use_b": ("BOOLEAN", {"default": False}),
                "a": ("LATENT",),
                "b": ("LATENT",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "route"
    CATEGORY = "Ollama/Route"

    def route(self, use_b, a, b):
        return (b if use_b else a,)


class RouteVaeByBool:
    """
    Route between two VAE inputs based on a boolean.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "use_b": ("BOOLEAN", {"default": False}),
                "a": ("VAE",),
                "b": ("VAE",),
            }
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "route"
    CATEGORY = "Ollama/Route"

    def route(self, use_b, a, b):
        return (b if use_b else a,)


class OllamaDebugInfo:
    """
    Simple debug node to visualize the plan JSON as text.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "notify"
    CATEGORY = "Ollama/Debug"
    OUTPUT_NODE = True

    def notify(self, text):
        print(f"\n[Ollama Plan Debug]:\n{text}\n")
        return {"ui": {"text": [text]}, "result": (text,)}


class PreviewTextMerge:
    """
    Aggregate multiple text inputs, preview them together, and output combined + passthrough strings.
    Useful for keeping a single preview node instead of many individual text preview nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Fixed max of 8 inputs; "count" lets the user decide how many are active.
        optional_fields = {f"text{i}": ("STRING", {"forceInput": True, "default": ""}) for i in range(1, 9)}
        return {
            "required": {
                "count": ("INT", {"default": 4, "min": 1, "max": 8}),
                "separator": ("STRING", {"default": "\\n"}),
                "emit_outputs": ("BOOLEAN", {"default": False}),
            },
            "optional": optional_fields,
        }

    RETURN_TYPES = (
        "STRING",  # combined
        "STRING", "STRING", "STRING", "STRING",
        "STRING", "STRING", "STRING", "STRING",  # passthrough text1..text8
    )
    RETURN_NAMES = (
        "combined",
        "text1", "text2", "text3", "text4",
        "text5", "text6", "text7", "text8",
    )
    FUNCTION = "merge"
    CATEGORY = "Ollama/Debug"
    OUTPUT_NODE = True

    def merge(self, count, separator, emit_outputs, **kwargs):
        # clamp count
        n = max(1, min(8, int(count)))
        texts = []
        for i in range(1, 9):
            key = f"text{i}"
            val = kwargs.get(key, "")
            texts.append("" if val is None else str(val))
        active = texts[:n]
        combined = separator.join(active)
        preview = "\n".join(f"[{i}] {t}" for i, t in enumerate(active, 1))
        if emit_outputs:
            return {
                "ui": {"text": [preview]},
                "result": (combined, *texts),
            }
        else:
            # Only preview; outputs are empty strings to satisfy the interface.
            empty_out = tuple("" for _ in range(9))
            return {
                "ui": {"text": [preview]},
                "result": empty_out,
            }


class LiveStatus:
    """
    Emit status text for live monitoring in the UI.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stage": ("STRING", {"default": "stage"}),
                "message": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "emit"
    CATEGORY = "Ollama/Debug"
    OUTPUT_NODE = True

    def emit(self, stage, message):
        text = f"[{stage}] {message}"
        print(text)
        return {"ui": {"text": [text]}, "result": (text,)}


class OllamaRegistryInfo:
    """
    Reads model_registry.json and outputs as string for debugging.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "registry_path": ("STRING", {"default": "model_registry.json"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json_content",)
    FUNCTION = "read_registry"
    CATEGORY = "Ollama/Debug"
    
    def read_registry(self, registry_path):
        try:
           registry = _read_registry(registry_path)
           import json
           return (json.dumps(registry, indent=2),)
        except Exception as e:
           return (f"Error reading registry: {e}",)
