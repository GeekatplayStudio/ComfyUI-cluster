import json
import os
import sys

# Script to sync model_registry.json content into install_manifest.json
# This ensures the installer sees all models defined in the registry.

REGISTRY_FILE = "model_registry.json"
MANIFEST_FILE = "install_manifest.json"

def main():
    if not os.path.exists(REGISTRY_FILE):
        print(f"Registry not found: {REGISTRY_FILE}")
        return

    with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
        registry = json.load(f)

    manifest_data = {"downloads": []}
    if os.path.exists(MANIFEST_FILE):
        try:
            with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
        except Exception:
            pass

    existing_filenames = {item["filename"]: item for item in manifest_data.get("downloads", [])}
    
    # helper to construct download url from homepage if possible
    def guess_download_url(homepage_url, filename):
        if not homepage_url:
            return ""
        if "huggingface.co" in homepage_url and "/resolve/" not in homepage_url:
            # Clean base url
            base = homepage_url.split("?")[0].rstrip("/")
            if base.endswith("/blob/main"):
                 base = base.replace("/blob/main", "/resolve/main")
            elif "/tree/" in base:
                 base = base.replace("/tree/", "/resolve/")
            else:
                 # assume repo root
                 base = f"{base}/resolve/main"
            
            return f"{base}/{filename}"
        return homepage_url

    new_count = 0

    # 1. Process Checkpoints
    for ckpt in registry.get("checkpoints", []):
        fname = ckpt.get("name")
        if fname in existing_filenames:
            continue
        
        folder = ckpt.get("folder_type", "checkpoints")
        desc = f"[{ckpt.get('type', 'sd')}] {ckpt.get('category', 'General')}"
        if ckpt.get("tags"):
            desc += " - " + ", ".join(ckpt.get("tags")[:3])
            
        url = guess_download_url(ckpt.get("url"), fname)
        
        entry = {
            "url": url,
            "subdir": folder,
            "filename": fname,
            "description": desc,
            "nsfw": False # Default, registry doesn't store this yet
        }
        manifest_data["downloads"].append(entry)
        existing_filenames[fname] = entry
        new_count += 1

    # 2. Process LoRAs
    for item in registry.get("loras", []):
        fname = item.get("name")
        if fname in existing_filenames:
            continue
            
        desc = f"[LoRA] {item.get('category', 'General')}"
        
        url = guess_download_url(item.get("url"), fname)

        entry = {
            "url": url,
            "subdir": "loras",
            "filename": fname,
            "description": desc,
            "nsfw": False
        }
        manifest_data["downloads"].append(entry)
        existing_filenames[fname] = entry
        new_count += 1

    # 3. Process ControlNets
    for item in registry.get("controlnets", []):
        fname = item.get("name")
        if fname in existing_filenames:
            continue
            
        desc = f"[ControlNet] {item.get('category', 'General')}"
        
        url = guess_download_url(item.get("url"), fname)

        entry = {
            "url": url,
            "subdir": "controlnet",
            "filename": fname,
            "description": desc,
            "nsfw": False
        }
        manifest_data["downloads"].append(entry)
        existing_filenames[fname] = entry
        new_count += 1

    if new_count > 0:
        print(f"Adding {new_count} new models from registry to manifest...")
        with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)
        print("Manifest updated.")
    else:
        print("Manifest is up to date.")

if __name__ == "__main__":
    main()
