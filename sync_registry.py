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
CONFIG_FILE = "cluster_config.json"

def load_registry(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_registry(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load_config():
    config_path = os.path.join(current_dir, CONFIG_FILE)
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_config(data):
    config_path = os.path.join(current_dir, CONFIG_FILE)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def get_user_root():
    config = load_config()
    root = config.get("models_root")
    
    # Verify validity of stored root
    if root and os.path.isdir(root):
        print(f"Using configured models root: {root}")
        return root
    
    print("\n--- Configuration Required ---")
    print("Could not automatically locate your models using ComfyUI defaults.")
    print("Please enter the path to your 'models' directory (e.g. D:\\ComfyUI\\models or I:\\models).")
    
    while True:
        try:
            root = input("Models Root Path: ").strip().strip('"') # Remove quotes if user copy-pasted
        except EOFError:
            print("\nInput canceled.")
            sys.exit(1)
            
        if os.path.isdir(root):
            config["models_root"] = root
            save_config(config)
            print("Configuration saved.\n")
            return root
        else:
            print(f"Error: Directory not found: {root}")
            print("Please try again.")

def check_exists(folder_type, filename, user_root=None):
    # 1. Try ComfyUI internal resolution (preferred)
    # This checks known paths including extra_model_paths.yaml if loaded correctly
    try:
        path = folder_paths.get_full_path(folder_type, filename)
        if path: return True
    except:
        pass # folder_paths might error on unknown types
    
    # 2. Manual check against User Root
    if user_root:
        # Logic: user_root + folder_type + filename
        # Normalize folder_type to directory name if needed (usually they match: loras -> loras)
        
        candidates = [folder_type]
        
        # Add symmetric fallbacks for checkpoints
        if folder_type in ["checkpoints", "diffusion_models", "unet"]:
            candidates = ["checkpoints", "diffusion_models", "unet"]
            
        for d in candidates:
            # Check direct join
            p = os.path.join(user_root, d, filename)
            if os.path.exists(p): return True
            
            # Recursive / Subdirectory check? 
            # Some users put things in models/checkpoints/SDXL/model.safetensors
            # But the registry filename is usually just "model.safetensors"
            # We won't do deep scan for performance unless requested, matching GUI manual approach
            
    return False

def sync_models():
    registry_path = os.path.join(current_dir, REGISTRY_FILE)
    log_path = os.path.join(current_dir, "model_scan.log")
    
    registry = load_registry(registry_path)
    
    if not registry:
        print(f"Registry not found at {registry_path}")
        return

    # Ensure we have a root to check against if Comfy fails
    user_root = get_user_root()

    print(f"Scanning models for registry: {registry_path}")
    changes = 0
    missing_models = []

    # Helper for processing lists
    def process_list(category_name, default_folder_type):
        nonlocal changes
        items = registry.get(category_name, [])
        if not items: return

        print(f"Scanning {category_name}...")
        for item in items:
            name = item.get("name")
            ftype = item.get("folder_type", default_folder_type)
            
            # Pass user_root to check_exists
            exists = check_exists(ftype, name, user_root)
            
            old_status = item.get("enabled", None)
            item["enabled"] = exists
            
            if old_status != exists:
                changes += 1
                status_str = "ENABLED" if exists else "DISABLED"
                print(f"  [{status_str}] {name}")
                
            if not exists:
                missing_models.append(f"[{category_name}] {name}")

    # Process all categories
    process_list("checkpoints", "checkpoints")
    process_list("loras", "loras")
    process_list("controlnets", "controlnet")
    process_list("vae", "vae")
    process_list("text_encoders", "clip")
    process_list("diffusion_models", "diffusion_models")
    process_list("clip", "clip")
    process_list("embeddings", "embeddings")

    if changes > 0:
        print(f"\nUpdating registry with {changes} status changes...")
        save_registry(registry_path, registry)
        print("Done.")
    else:
        print("\nAll model statuses are up to date.")
        
    # Write Log
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"Scan Date: {os.path.basename(current_dir)}\n")
        log.write(f"User Models Root: {user_root}\n")
        log.write(f"Missing Models ({len(missing_models)}):\n")
        for m in missing_models:
            log.write(f"{m}\n")
    
    print(f"\nScan log written to: {log_path}")
    if missing_models:
        print(f"WARNING: {len(missing_models)} models were not found. Check log for details.")

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
        
        # Interactive mode: Loop until exit
        while True:
            try:
                choice = input("\nSelect option (1-3): ").strip()
            except EOFError:
                break
            
            if choice == "1":
                sync_models()
            elif choice == "2":
                reset_registry()
            elif choice == "3":
                break
            else:
                print("Invalid option.")


