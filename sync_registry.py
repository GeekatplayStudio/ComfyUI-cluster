import json
import os
import sys

# Add ComfyUI root to path to import folder_paths
# Assumes this script is in custom_nodes/ComfyUI-Cluster/
current_dir = os.path.dirname(os.path.abspath(__file__))
comfy_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(comfy_root)

try:
    import folder_paths
except ImportError:
    print("Error: Could not import folder_paths. Make sure this script is running inside the ComfyUI environment or folder structure.")
    print(f"Constructed ComfyUI root path: {comfy_root}")
    sys.exit(1)

REGISTRY_FILE = "model_registry.json"

def load_registry(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_registry(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def check_exists(folder_type, filename):
    # Map registry folder types to ComfyUI folder types
    # folder_paths.get_full_path takes (type, filename)
    
    # Handle custom folder overrides from our registry
    # In registry checkoints usually have "folder_type": "diffusion_models" or "checkpoints"
    
    # Try the specific type first
    path = folder_paths.get_full_path(folder_type, filename)
    if path:
        return True
    
    # Fallbacks if strict type fails (files might be in 'checkpoints' even if type is 'diffusion_models')
    if folder_type == "diffusion_models":
        path = folder_paths.get_full_path("checkpoints", filename)
        if path: return True
        path = folder_paths.get_full_path("unet", filename) # Rare but possible
        if path: return True

    return False

def sync_models():
    registry_path = os.path.join(current_dir, REGISTRY_FILE)
    registry = load_registry(registry_path)
    
    if not registry:
        print(f"Registry not found at {registry_path}")
        return

    print(f"Scanning models for registry: {registry_path}")
    changes = 0

    # 1. Processing Checkpoints
    for ckpt in registry.get("checkpoints", []):
        name = ckpt.get("name")
        # Default to 'checkpoints' if not specified
        ftype = ckpt.get("folder_type", "checkpoints")
        
        exists = check_exists(ftype, name)
        
        old_status = ckpt.get("enabled", None)
        ckpt["enabled"] = exists
        
        if old_status != exists:
            status_str = "ENABLED" if exists else "DISABLED"
            print(f"[{status_str}] Checkpoint: {name}")
            changes += 1

    # 2. Processing LoRAs
    for l in registry.get("loras", []):
        name = l.get("name")
        exists = check_exists("loras", name)
        
        old_status = l.get("enabled", None)
        l["enabled"] = exists
        
        if old_status != exists:
            status_str = "ENABLED" if exists else "DISABLED"
            print(f"[{status_str}] LoRA: {name}")
            changes += 1

    # 3. Processing ControlNets
    for cn in registry.get("controlnets", []):
        name = cn.get("name")
        exists = check_exists("controlnet", name)
        
        old_status = cn.get("enabled", None)
        cn["enabled"] = exists
        
        if old_status != exists:
            status_str = "ENABLED" if exists else "DISABLED"
            print(f"[{status_str}] ControlNet: {name}")
            changes += 1

    if changes > 0:
        print(f"\nUpdating registry with {changes} status changes...")
        save_registry(registry_path, registry)
        print("Done.")
    else:
        print("\nAll model statuses are up to date.")

def reset_registry():
    registry_path = os.path.join(current_dir, REGISTRY_FILE)
    registry = load_registry(registry_path)
    
    if not registry:
        print(f"Registry not found at {registry_path}")
        return

    print("Resetting all models to ENABLED...")
    count = 0

    for category in ["checkpoints", "loras", "controlnets", "vae", "text_encoders", "diffusion_models", "clip", "embeddings"]:
        if category in registry:
            for item in registry[category]:
                item["enabled"] = True
                count += 1
    
    save_registry(registry_path, registry)
    print(f"Done. {count} items enabled.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--sync":
            sync_models()
        elif sys.argv[1] == "--reset":
            reset_registry()
        else:
            print("Usage: sync_registry.py [--sync | --reset]")
    else:
        print("\n--- Model Registry Manager ---")
        print("1. Sync with Disk (Enable installed, Disable missing)")
        print("2. Reset Registry (Enable ALL models)")
        print("3. Exit")
        choice = input("Select option (1-3): ").strip()
        
        if choice == "1":
            sync_models()
        elif choice == "2":
            reset_registry()

