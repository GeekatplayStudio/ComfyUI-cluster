
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


_DEFAULT_REGISTRY = {
    "version": 1,
    "checkpoints": [],
    "loras": [],
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

    def _strip(items):
        compact = []
        for item in items:
            compact.append({
                "name": item.get("name", ""),
                "type": item.get("type", ""),
                "tags": item.get("tags", []),
                "recommended": bool(item.get("recommended", False)),
            })
        return compact

    return {"checkpoints": _strip(checkpoints), "loras": _strip(loras)}


def _score_tags(prompt: str, tags: list[str]) -> int:
    score = 0
    for tag in tags:
        if tag and tag in prompt:
            score += 3
    return score


def _heuristic_plan(prompt: str, registry: dict) -> dict:
    prompt_l = prompt.lower()
    checkpoints = registry.get("checkpoints", [])
    loras = registry.get("loras", [])

    best_ckpt = None
    best_score = -1
    for ckpt in checkpoints:
        tags = [t.lower() for t in ckpt.get("tags", [])]
        score = _score_tags(prompt_l, tags)
        if ckpt.get("recommended"):
            score += 1
        if score > best_score:
            best_score = score
            best_ckpt = ckpt

    if best_ckpt is None and checkpoints:
        best_ckpt = checkpoints[0]

    selected_loras = []
    selected_strengths = []
    for lora in loras:
        tags = [t.lower() for t in lora.get("tags", [])]
        if _score_tags(prompt_l, tags) > 0:
            selected_loras.append(lora.get("name", ""))
            selected_strengths.append(float(lora.get("strength", 0.7)))

    model_type = (best_ckpt or {}).get("type", "sdxl")
    return {
        "model_type": model_type or "sdxl",
        "checkpoint": (best_ckpt or {}).get("name", ""),
        "loras": selected_loras,
        "lora_strengths": selected_strengths,
        "positive_prompt": prompt.strip(),
        "negative_prompt": "",
        "steps": 25,
        "cfg": 6.5,
        "width": 1024 if model_type == "sdxl" else 768,
        "height": 1024 if model_type == "sdxl" else 768,
        "seed": int(time.time()) % 2_000_000_000,
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
                "ollama_host": ("STRING", {"default": "localhost"}),
                "ollama_port": ("INT", {"default": 11434, "min": 1, "max": 65535}),
            }
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "INT",
        "FLOAT",
        "INT",
        "INT",
        "INT",
        "STRING",
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "checkpoint",
        "vae_name",
        "clip_name",
        "loras",
        "lora_strengths",
        "model_type",
        "steps",
        "cfg",
        "width",
        "height",
        "seed",
        "positive_prompt",
        "negative_prompt",
        "plan_json",
    )
    FUNCTION = "plan"
    CATEGORY = "Ollama/Planner"

    def plan(self, prompt, ollama_model, registry_path, task_hint, user_negative, ollama_host="localhost", ollama_port=11434):
        registry = _read_registry(registry_path)
        compact = _compact_registry(registry)
        system_prompt = (
            "You are a routing planner for image generation. "
            "Return ONLY valid JSON with keys: "
            "model_type (sdxl|sd15|flux), checkpoint, vae_name (optional), clip_name (optional), "
            "loras (array of names), lora_strengths (array of floats), positive_prompt, negative_prompt, "
            "steps (int), cfg (float), width (int), height (int), seed (int). "
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

        return (
            str(plan.get("checkpoint", "")),
            str(plan.get("vae_name", "")),
            str(plan.get("clip_name", "")),
            lora_str,
            strength_str,
            str(plan.get("model_type", "sdxl")),
            int(plan.get("steps", 25)),
            float(plan.get("cfg", 6.5)),
            int(plan.get("width", 1024)),
            int(plan.get("height", 1024)),
            int(plan.get("seed", int(time.time()) % 2_000_000_000)),
            str(plan.get("positive_prompt", prompt)),
            negative,
            json.dumps(plan, ensure_ascii=True),
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

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load"
    CATEGORY = "Ollama/Loader"

    def load(self, checkpoint, model_type, registry_path, custom_path="", vae_override="", clip_override=""):
        registry = _read_registry(registry_path)
        
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

        # 2. Resolve Checkpoint Path (or Fallback)
        ckpt_path = self._find_path("checkpoints", checkpoint, custom_path)
        
        if not ckpt_path:
            # Checkpoint Missing: logical fallback
            print(f"\\n[OllamaRouter] !!! MISSING MODEL: {checkpoint} !!!")
            if target_info:
                print(f"[OllamaRouter] DOWNLOAD URL: {target_info.get('url', 'N/A')}")
                print(f"[OllamaRouter] GROUP: {target_info.get('group', 'N/A')}")
            
            # Find Fallback
            fallback_ckpt = self._find_fallback(registry, target_info, resolved_type, custom_path)
            if fallback_ckpt:
                print(f"[OllamaRouter] -> Falling back to available model: {fallback_ckpt}")
                ckpt_path = self._find_path("checkpoints", fallback_ckpt, custom_path)
            else:
                raise FileNotFoundError(f"Model '{checkpoint}' not found and no suitable fallback in group/type '{resolved_type}' discovered.")

        # 3. Load Main Components
        # If it's a standard checkpoint (safetensors with bundled model/clip/vae)
        # Note: folder_paths.get_full_path_or_raise would have raised if we used it directly, but we used _find_path
        
        out_model, out_clip, out_vae = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )[:3]

        # 4. Handle Overrides (Universal Loading)
        if vae_override:
            print(f"[OllamaRouter] Attempting VAE Override: {vae_override}")
            v_path = self._find_path("vae", vae_override, custom_path)
            if v_path:
                try:
                    out_vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(v_path))
                    print(f"[OllamaRouter] VAE loaded: {v_path}")
                except Exception as e:
                    print(f"[OllamaRouter] Failed to load VAE: {e}")
            else:
                print(f"[OllamaRouter] Warning: VAE Override '{vae_override}' not found.")

        if clip_override:
             print(f"[OllamaRouter] Attempting CLIP Override: {clip_override}")
             c_path = self._find_path("checkpoints", clip_override, custom_path)
             if c_path:
                  try:
                      _, out_clip, _ = comfy.sd.load_checkpoint_guess_config(
                            c_path, 
                            output_vae=False, 
                            output_clip=True, 
                            embedding_directory=folder_paths.get_folder_paths("embeddings")
                      )[:3]
                      print(f"[OllamaRouter] CLIP loaded: {c_path}")
                  except Exception as e:
                      print(f"[OllamaRouter] Failed to load CLIP: {e}")
             else:
                 print(f"[OllamaRouter] Warning: CLIP Override '{clip_override}' not found.")

        return (out_model, out_clip, out_vae)

    def _find_path(self, folder_type, filename, custom_path):
        try:
            path = folder_paths.get_full_path(folder_type, filename)
            if path: return path
        except: pass
        
        if folder_type != "checkpoints":
             try:
                path = folder_paths.get_full_path("checkpoints", filename)
                if path: return path
             except: pass

        if custom_path and os.path.exists(custom_path):
            p = os.path.join(custom_path, filename)
            if os.path.exists(p): return p
            for root, dirs, files in os.walk(custom_path):
                if filename in files:
                    return os.path.join(root, filename)
        return None

    def _find_fallback(self, registry, target_info, resolved_type, custom_path):
        candidates = registry.get("checkpoints", [])
        if not target_info:
             for c in candidates:
                 if c.get("type") == resolved_type:
                     if self._find_path("checkpoints", c["name"], custom_path): return c["name"]
             return None

        grp = target_info.get("group")
        cat = target_info.get("category")
        
        # Priority 1: Same Group & Category
        for c in candidates:
            if c["name"] == target_info["name"]: continue
            if c.get("group") == grp and c.get("category") == cat and c.get("type") == resolved_type:
                 if self._find_path("checkpoints", c["name"], custom_path): return c["name"]
        
        # Priority 2: Same Group & Type
        for c in candidates:
            if c["name"] == target_info["name"]: continue
            if c.get("group") == grp and c.get("type") == resolved_type:
                 if self._find_path("checkpoints", c["name"], custom_path): return c["name"]

        # Priority 3: Same Type
        for c in candidates:
            if c["name"] == target_info["name"]: continue
            if c.get("type") == resolved_type:
                 if self._find_path("checkpoints", c["name"], custom_path): return c["name"]
        
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

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "apply"
    CATEGORY = "Ollama/Loader"

    def apply(self, model, clip, loras, lora_strengths, custom_path=""):
        lora_list = [l.strip() for l in loras.split(",") if l.strip()]
        strength_list = [s.strip() for s in lora_strengths.split(",") if s.strip()]
        strengths = []
        for s in strength_list:
            try:
                strengths.append(float(s))
            except Exception:
                strengths.append(0.7)

        if not lora_list:
            return (model, clip)
            
        registry = {} 
        try:
           registry = _read_registry("model_registry.json")
        except:
           pass

        model_out, clip_out = model, clip
        for idx, lora_name in enumerate(lora_list):
            strength = strengths[idx] if idx < len(strengths) else 0.7
            
            lora_path = self._find_path(lora_name, custom_path)
            
            if not lora_path:
                target_info = None
                for l in registry.get("loras", []):
                    if l.get("name") == lora_name:
                        target_info = l
                        break
                
                print(f"\\n[OllamaRouter] !!! MISSING LORA: {lora_name} !!!")
                if target_info:
                     print(f"[OllamaRouter] DOWNLOAD URL: {target_info.get('url', 'N/A')}")
                
                # Try Fallback
                fallback_name = None
                if target_info:
                   fallback_name = self._find_fallback(registry, target_info, custom_path)
                
                if fallback_name:
                    print(f"[OllamaRouter] -> Falling back to LoRA: {fallback_name}")
                    lora_path = self._find_path(fallback_name, custom_path)
                else:
                    print(f"[OllamaRouter] -> Skipping missing LoRA.")
                    continue

            # Load
            try:
                lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                model_out, clip_out = comfy.sd.load_lora_for_models(model_out, clip_out, lora, strength, strength)
                print(f"[OllamaRouter] LoRA Loaded: {os.path.basename(lora_path)}")
            except Exception as e:
                print(f"[OllamaRouter] Error loading LoRA {lora_path}: {e}")

        return (model_out, clip_out)

    def _find_path(self, filename, custom_path):
        try:
            path = folder_paths.get_full_path("loras", filename)
            if path: return path
        except: pass
        if custom_path and os.path.exists(custom_path):
            p = os.path.join(custom_path, filename)
            if os.path.exists(p): return p
        return None

    def _find_fallback(self, registry, target_info, custom_path):
        target_grp = target_info.get("group", "")
        target_cat = target_info.get("category", "")
        
        # Priority 1: Same Group & Category
        for l in registry.get("loras", []):
            if l["name"] == target_info["name"]: continue
            if l.get("group") == target_grp and l.get("category") == target_cat:
                if self._find_path(l["name"], custom_path): return l["name"]
        
        # Priority 2: Same Category only
        for l in registry.get("loras", []):
            if l["name"] == target_info["name"]: continue
            if l.get("category") == target_cat:
                if self._find_path(l["name"], custom_path): return l["name"]
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
                "ollama_host": ("STRING", {"default": "localhost"}),
                "ollama_port": ("INT", {"default": 11434, "min": 1, "max": 65535}),
            }
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "INT",
        "FLOAT",
        "INT",
        "INT",
        "INT",
        "STRING",
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "checkpoint",
        "vae_name",
        "clip_name",
        "loras",
        "lora_strengths",
        "model_type",
        "steps",
        "cfg",
        "width",
        "height",
        "seed",
        "positive_prompt",
        "negative_prompt",
        "plan_json",
    )
    FUNCTION = "plan"
    CATEGORY = "Ollama/Planner"

    def plan(self, image, prompt, ollama_model, registry_path, task_hint, user_negative, ollama_host="localhost", ollama_port=11434):
        registry = _read_registry(registry_path)
        compact = _compact_registry(registry)
        system_prompt = (
            "You analyze an image style and select the best image generation setup. "
            "Return ONLY valid JSON with keys: "
            "model_type (sdxl|sd15|flux), checkpoint, vae_name (optional), clip_name (optional), "
            "loras (array of names), lora_strengths (array of floats), positive_prompt, negative_prompt, "
            "steps (int), cfg (float), width (int), height (int), seed (int). "
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

        return (
            str(plan.get("checkpoint", "")),
            str(plan.get("vae_name", "")),
            str(plan.get("clip_name", "")),
            lora_str,
            strength_str,
            str(plan.get("model_type", "sdxl")),
            int(plan.get("steps", 25)),
            float(plan.get("cfg", 6.5)),
            int(plan.get("width", 1024)),
            int(plan.get("height", 1024)),
            int(plan.get("seed", int(time.time()) % 2_000_000_000)),
            str(plan.get("positive_prompt", prompt)),
            negative,
            json.dumps(plan, ensure_ascii=True),
        )


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
