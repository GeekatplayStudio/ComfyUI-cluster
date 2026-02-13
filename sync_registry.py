import argparse
import json
import os
import sys
from datetime import datetime

# Add ComfyUI root to path to import folder_paths
# Assumes this script is in custom_nodes/ComfyUI-Cluster/
current_dir = os.path.dirname(os.path.abspath(__file__))
comfy_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if comfy_root not in sys.path:
    sys.path.append(comfy_root)

try:
    import folder_paths
except ImportError:
    print("Error: Could not import folder_paths.")
    print("Make sure this script runs inside the ComfyUI folder structure.")
    print(f"Constructed ComfyUI root path: {comfy_root}")
    sys.exit(1)


REGISTRY_FILE = "model_registry.json"
CONFIG_FILE = "cluster_config.json"
SCAN_LOG_FILE = "model_scan.log"

CATEGORY_SPECS = [
    ("checkpoints", "checkpoints"),
    ("loras", "loras"),
    ("controlnets", "controlnet"),
    ("vae", "vae"),
    ("text_encoders", "clip"),
    ("diffusion_models", "diffusion_models"),
    ("clip", "clip"),
    ("embeddings", "embeddings"),
]

FOLDER_TYPE_ALIAS = {
    "checkpoint": "checkpoints",
    "checkpoints": "checkpoints",
    "lora": "loras",
    "loras": "loras",
    "controlnet": "controlnet",
    "controlnets": "controlnet",
    "clip": "clip",
    "clips": "clip",
    "text_encoder": "clip",
    "text_encoders": "clip",
    "vae": "vae",
    "unet": "diffusion_models",
    "diffusion_model": "diffusion_models",
    "diffusion_models": "diffusion_models",
    "embedding": "embeddings",
    "embeddings": "embeddings",
}

_FOLDER_ROOTS_CACHE = {}
_FILE_INDEX_CACHE = {}


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


def normalize_path(path):
    return os.path.normpath(os.path.abspath(path))


def normalize_for_compare(path_value):
    if not isinstance(path_value, str):
        return ""
    value = path_value.strip().strip('"')
    if not value:
        return ""
    value = os.path.expandvars(os.path.expanduser(value))
    if not os.path.isabs(value):
        value = os.path.join(current_dir, value)
    return normalize_path(value)


def canonical_folder_type(folder_type, default_type):
    raw = (folder_type or default_type or "").strip().lower()
    if not raw:
        return default_type
    return FOLDER_TYPE_ALIAS.get(raw, raw)


def expand_folder_types(folder_type):
    canonical = canonical_folder_type(folder_type, "checkpoints")

    if canonical == "checkpoints":
        expanded = ["checkpoints", "diffusion_models", "unet"]
    elif canonical == "diffusion_models":
        expanded = ["diffusion_models", "unet", "checkpoints"]
    elif canonical == "clip":
        expanded = ["clip", "text_encoders", "checkpoints"]
    elif canonical == "vae":
        expanded = ["vae", "checkpoints"]
    elif canonical == "controlnet":
        expanded = ["controlnet"]
    elif canonical == "loras":
        expanded = ["loras"]
    elif canonical == "embeddings":
        expanded = ["embeddings"]
    else:
        expanded = [canonical]

    if canonical not in expanded:
        expanded.insert(0, canonical)

    deduped = []
    seen = set()
    for ft in expanded:
        if ft in seen:
            continue
        seen.add(ft)
        deduped.append(ft)
    return deduped


def get_folder_roots(folder_type):
    if folder_type in _FOLDER_ROOTS_CACHE:
        return _FOLDER_ROOTS_CACHE[folder_type]

    roots = []
    try:
        roots = folder_paths.get_folder_paths(folder_type) or []
    except Exception:
        roots = []

    cleaned = []
    seen = set()
    for root in roots:
        if not root:
            continue
        norm = normalize_path(root)
        if norm in seen:
            continue
        seen.add(norm)
        cleaned.append(norm)

    _FOLDER_ROOTS_CACHE[folder_type] = cleaned
    return cleaned


def build_filename_index(root):
    norm_root = normalize_path(root)
    if norm_root in _FILE_INDEX_CACHE:
        return _FILE_INDEX_CACHE[norm_root]

    index = {}
    if os.path.isdir(norm_root):
        for walk_root, _, files in os.walk(norm_root):
            for file_name in files:
                key = file_name.lower()
                index.setdefault(key, []).append(os.path.join(walk_root, file_name))

    _FILE_INDEX_CACHE[norm_root] = index
    return index


def is_subpath(path, root):
    try:
        path_n = normalize_path(path)
        root_n = normalize_path(root)
        return os.path.commonpath([path_n, root_n]) == root_n
    except Exception:
        return False


def infer_folder_type_from_path(path, candidate_types, user_root=None):
    for ft in candidate_types:
        for root in get_folder_roots(ft):
            if is_subpath(path, root):
                return ft

    if user_root and is_subpath(path, user_root):
        rel = os.path.relpath(path, user_root)
        first_part = rel.split(os.sep)[0].lower() if rel else ""
        if first_part:
            first_part = canonical_folder_type(first_part, first_part)
            for ft in candidate_types:
                if canonical_folder_type(ft, ft) == first_part:
                    return ft

    return candidate_types[0] if candidate_types else None


def validate_existing_path(existing_path, expected_name):
    resolved = normalize_for_compare(existing_path)
    if not resolved:
        return None
    if not os.path.isfile(resolved):
        return None
    if os.path.basename(resolved).lower() != expected_name.lower():
        return None
    return resolved


def resolve_model_path(filename, preferred_folder_type, user_root=None, current_path=None):
    if not filename:
        return (None, preferred_folder_type, "missing_name")

    candidate_types = expand_folder_types(preferred_folder_type)
    target_name = filename.strip()
    target_name_l = target_name.lower()

    # 1) If registry path already points to the expected file, trust it.
    existing = validate_existing_path(current_path, target_name)
    if existing:
        inferred = infer_folder_type_from_path(existing, candidate_types, user_root)
        return (existing, inferred, "registry_path")

    # 2) ComfyUI resolver.
    for ft in candidate_types:
        try:
            resolved = folder_paths.get_full_path(ft, target_name)
        except Exception:
            resolved = None
        if resolved and os.path.isfile(resolved):
            return (normalize_path(resolved), ft, "comfy_get_full_path")

    # 3) Search ComfyUI configured roots.
    searched_roots = set()
    for ft in candidate_types:
        for root in get_folder_roots(ft):
            if root in searched_roots:
                continue
            searched_roots.add(root)

            direct = os.path.join(root, target_name)
            if os.path.isfile(direct):
                return (normalize_path(direct), ft, "comfy_root_exact")

            index = build_filename_index(root)
            matches = index.get(target_name_l, [])
            if matches:
                return (normalize_path(matches[0]), ft, "comfy_root_recursive")

    # 4) Search user root (external models root), if configured.
    if user_root and os.path.isdir(user_root):
        direct = os.path.join(user_root, target_name)
        if os.path.isfile(direct):
            inferred = infer_folder_type_from_path(direct, candidate_types, user_root)
            return (normalize_path(direct), inferred, "user_root_direct")

        for ft in candidate_types:
            sub = os.path.join(user_root, ft, target_name)
            if os.path.isfile(sub):
                return (normalize_path(sub), ft, "user_root_subfolder")

        user_index = build_filename_index(user_root)
        matches = user_index.get(target_name_l, [])
        if matches:
            best = normalize_path(matches[0])
            inferred = infer_folder_type_from_path(best, candidate_types, user_root)
            return (best, inferred, "user_root_recursive")

    return (None, candidate_types[0], "not_found")


def get_user_root(prompt_if_missing=True):
    config = load_config()
    root = config.get("models_root")

    if root and os.path.isdir(root):
        root = normalize_path(root)
        print(f"Using configured models root: {root}")
        return root

    if not prompt_if_missing:
        return None

    print("\n--- Optional Configuration ---")
    print("Could not load a saved models root directory.")
    print("Enter your models root path if you keep models outside ComfyUI defaults.")
    print("Press Enter to skip.")

    while True:
        try:
            entered = input("Models Root Path: ").strip().strip('"')
        except EOFError:
            print("Input unavailable. Continuing without custom models root.")
            return None

        if not entered:
            print("Skipping custom models root.")
            return None

        if os.path.isdir(entered):
            root = normalize_path(entered)
            config["models_root"] = root
            save_config(config)
            print("Configuration saved.\n")
            return root

        print(f"Error: Directory not found: {entered}")


def write_scan_log(
    log_path,
    scan_time,
    user_root,
    stats,
    missing_models,
    status_changes,
    path_changes,
    folder_type_changes,
    dry_run,
):
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"Scan Date: {scan_time}\n")
        log.write(f"Dry Run: {dry_run}\n")
        log.write(f"User Models Root: {user_root or '(not set)'}\n")
        log.write(f"Scanned Items: {stats['scanned']}\n")
        log.write(f"Status Changes: {stats['status_changes']}\n")
        log.write(f"Path Changes: {stats['path_changes']}\n")
        log.write(f"Folder Type Changes: {stats['folder_type_changes']}\n")
        log.write(f"Missing Models: {stats['missing']}\n")

        log.write("\n== Status Changes ==\n")
        for entry in status_changes:
            log.write(f"{entry}\n")

        log.write("\n== Path Changes ==\n")
        for entry in path_changes:
            log.write(f"{entry}\n")

        log.write("\n== Folder Type Changes ==\n")
        for entry in folder_type_changes:
            log.write(f"{entry}\n")

        log.write("\n== Missing Models ==\n")
        for entry in missing_models:
            log.write(f"{entry}\n")


def sync_models(dry_run=False, prompt_for_root=True):
    registry_path = os.path.join(current_dir, REGISTRY_FILE)
    log_path = os.path.join(current_dir, SCAN_LOG_FILE)

    registry = load_registry(registry_path)
    if not registry:
        print(f"Registry not found at {registry_path}")
        return

    user_root = get_user_root(prompt_if_missing=prompt_for_root)

    print(f"\nScanning models for registry: {registry_path}")
    scan_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    stats = {
        "scanned": 0,
        "status_changes": 0,
        "path_changes": 0,
        "folder_type_changes": 0,
        "missing": 0,
    }
    missing_models = []
    status_change_details = []
    path_change_details = []
    folder_type_change_details = []

    for category_name, default_folder_type in CATEGORY_SPECS:
        items = registry.get(category_name, [])
        if not items:
            continue

        print(f"Scanning {category_name} ({len(items)} items)...")

        for item in items:
            name = str(item.get("name", "")).strip()
            if not name:
                continue

            stats["scanned"] += 1

            preferred_ft = canonical_folder_type(item.get("folder_type"), default_folder_type)

            if "folder_type" in item and item.get("folder_type") != preferred_ft:
                old_ft = item.get("folder_type")
                item["folder_type"] = preferred_ft
                stats["folder_type_changes"] += 1
                folder_type_change_details.append(
                    f"[{category_name}] {name}: {old_ft} -> {preferred_ft} (normalized alias)"
                )

            resolved_path, resolved_ft, source = resolve_model_path(
                filename=name,
                preferred_folder_type=preferred_ft,
                user_root=user_root,
                current_path=item.get("path"),
            )

            exists = resolved_path is not None
            old_enabled = item.get("enabled")
            if old_enabled != exists:
                item["enabled"] = exists
                stats["status_changes"] += 1
                status_word = "ENABLED" if exists else "DISABLED"
                print(f"  [{status_word}] {name}")
                status_change_details.append(
                    f"[{category_name}] {name}: {old_enabled} -> {exists}"
                )
            else:
                item["enabled"] = exists

            if exists:
                old_path_raw = item.get("path")
                old_path_norm = normalize_for_compare(old_path_raw)
                if old_path_norm != resolved_path:
                    item["path"] = resolved_path
                    stats["path_changes"] += 1
                    print(f"  [PATH FIXED] {name}")
                    path_change_details.append(
                        f"[{category_name}] {name}: {old_path_raw} -> {resolved_path} ({source})"
                    )

                should_store_folder_type = ("folder_type" in item) or (resolved_ft != default_folder_type)
                if should_store_folder_type and item.get("folder_type") != resolved_ft:
                    old_ft = item.get("folder_type")
                    item["folder_type"] = resolved_ft
                    stats["folder_type_changes"] += 1
                    print(f"  [TYPE FIXED] {name}: {old_ft} -> {resolved_ft}")
                    folder_type_change_details.append(
                        f"[{category_name}] {name}: {old_ft} -> {resolved_ft}"
                    )
            else:
                stats["missing"] += 1
                missing_models.append(f"[{category_name}] {name}")
                if "path" in item:
                    old_path_raw = item.get("path")
                    if old_path_raw:
                        stats["path_changes"] += 1
                        print(f"  [PATH CLEARED] {name}")
                        path_change_details.append(
                            f"[{category_name}] {name}: {old_path_raw} -> (removed, not found)"
                        )
                    item.pop("path", None)

    total_changes = stats["status_changes"] + stats["path_changes"] + stats["folder_type_changes"]

    if dry_run:
        print("\nDry run complete. No changes were written.")
    elif total_changes > 0:
        print(f"\nUpdating registry with {total_changes} changes...")
        save_registry(registry_path, registry)
        print("Done.")
    else:
        print("\nAll model statuses and paths are up to date.")

    write_scan_log(
        log_path=log_path,
        scan_time=scan_time,
        user_root=user_root,
        stats=stats,
        missing_models=missing_models,
        status_changes=status_change_details,
        path_changes=path_change_details,
        folder_type_changes=folder_type_change_details,
        dry_run=dry_run,
    )
    print(f"\nScan log written to: {log_path}")
    print(
        f"Summary: scanned={stats['scanned']}, "
        f"status_changes={stats['status_changes']}, "
        f"path_changes={stats['path_changes']}, "
        f"folder_type_changes={stats['folder_type_changes']}, "
        f"missing={stats['missing']}"
    )

    if stats["missing"] > 0:
        print(f"WARNING: {stats['missing']} models were not found. Check log for details.")


def reset_registry():
    registry_path = os.path.join(current_dir, REGISTRY_FILE)
    registry = load_registry(registry_path)

    if not registry:
        print(f"Registry not found at {registry_path}")
        return

    print("Resetting all models to ENABLED...")
    count = 0

    for category, _default_folder_type in CATEGORY_SPECS:
        if category in registry:
            for item in registry[category]:
                item["enabled"] = True
                count += 1

    save_registry(registry_path, registry)
    print(f"Done. {count} items enabled.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sync model_registry.json with actual disk paths and availability."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--sync",
        action="store_true",
        help="Scan models, fix path/folder_type, and update enabled status.",
    )
    mode.add_argument(
        "--reset",
        action="store_true",
        help="Set enabled=true for all known registry entries.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run sync logic without writing model_registry.json.",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Do not prompt for models_root when no saved root exists.",
    )
    return parser.parse_args()


def run_interactive_menu():
    print("\n--- Model Registry Manager ---")
    print("1. Sync + Repair Paths (write changes)")
    print("2. Dry Run (preview changes only)")
    print("3. Reset Registry (enable all)")
    print("4. Exit")

    while True:
        try:
            choice = input("\nSelect option (1-4): ").strip()
        except EOFError:
            break

        if choice == "1":
            sync_models(dry_run=False, prompt_for_root=True)
        elif choice == "2":
            sync_models(dry_run=True, prompt_for_root=True)
        elif choice == "3":
            reset_registry()
        elif choice == "4":
            break
        else:
            print("Invalid option.")


if __name__ == "__main__":
    args = parse_args()

    if args.reset:
        reset_registry()
    elif args.sync or args.dry_run:
        sync_models(dry_run=args.dry_run, prompt_for_root=not args.no_prompt)
    else:
        run_interactive_menu()
