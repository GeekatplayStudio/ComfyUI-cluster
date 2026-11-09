"""
ComfyUI-Cluster by Geekatplay Studio.

Rebuilds install_manifest.json from model_registry.json so the installer always
lists exactly what the registry knows about: every model plus the text encoder
and VAE files each one needs.

    python update_manifest.py
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REGISTRY_FILE = os.path.join(HERE, "model_registry.json")
MANIFEST_FILE = os.path.join(HERE, "install_manifest.json")

# registry section -> (default models subfolder, label shown in the installer)
SECTIONS = [
    ("checkpoints", "checkpoints", "Model"),
    ("text_encoders", "text_encoders", "Text encoder"),
    ("vae", "vae", "VAE"),
    ("loras", "loras", "LoRA"),
]


# sync_registry.py may store legacy folder names; the installer writes to the current ones.
SUBDIR_ALIAS = {"clip": "text_encoders", "unet": "diffusion_models"}


def _names(value):
    return [v.strip() for v in str(value or "").split(",") if v.strip()]


def build_manifest(registry):
    used_by = {}
    for ckpt in registry.get("checkpoints", []):
        label = ckpt.get("title") or ckpt.get("name", "")
        companions = _names(ckpt.get("required_clip")) + _names(ckpt.get("required_vae")) + _names(ckpt.get("required_audio_vae"))
        for name in companions:
            used_by.setdefault(name, []).append(label)

    downloads = []
    seen = set()
    for section, default_subdir, kind in SECTIONS:
        for item in registry.get(section, []):
            name = item.get("name")
            if not name or name in seen:
                continue
            seen.add(name)
            if section == "checkpoints":
                needs = _names(item.get("required_clip")) + _names(item.get("required_vae")) + _names(item.get("required_audio_vae"))
                description = item.get("title") or name
            else:
                needs = []
                users = used_by.get(name, [])
                description = item.get("title") or (f"Needed by: {', '.join(users)}" if users else kind)
            downloads.append({
                "url": item.get("url", ""),
                "subdir": SUBDIR_ALIAS.get(item.get("folder_type"), item.get("folder_type") or default_subdir),
                "filename": name,
                "kind": kind,
                "group": item.get("group") or "Shared files",
                "description": description,
                "size_gb": item.get("size_gb", 0),
                "min_vram": item.get("min_vram", 0),
                "gated": bool(item.get("gated", False)),
                "needs": needs,
            })
    return {"generated_from": "model_registry.json", "downloads": downloads}


def main():
    if not os.path.exists(REGISTRY_FILE):
        print(f"Registry not found: {REGISTRY_FILE}")
        return 1
    with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
        registry = json.load(f)
    manifest = build_manifest(registry)
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"Manifest rebuilt: {len(manifest['downloads'])} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
