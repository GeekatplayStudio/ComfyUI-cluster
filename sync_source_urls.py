import argparse
import json
import os
import re
import sys
from datetime import datetime
from urllib.parse import quote_plus, urlparse, urlunparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


REGISTRY_FILE = "model_registry.json"
URL_LOG_FILE = "model_url_scan.log"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

CATEGORY_SPECS = [
    "checkpoints",
    "loras",
    "controlnets",
    "vae",
    "text_encoders",
    "diffusion_models",
    "clip",
    "embeddings",
]


def load_registry(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_registry(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def normalize_key(value):
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower())


def filename_stem(filename):
    return os.path.splitext(filename or "")[0]


def is_huggingface_url(url):
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return False
    return host.endswith("huggingface.co")


def is_civitai_url(url):
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return False
    return host.endswith("civitai.com")


def normalize_huggingface_url(url, filename):
    parsed = urlparse(url.strip())
    scheme = parsed.scheme or "https"
    host = parsed.netloc
    path = parsed.path or ""

    path = path.replace("/blob/", "/resolve/")
    path = path.replace("/tree/", "/resolve/")

    clean_parts = [p for p in path.split("/") if p]
    if "resolve" in clean_parts:
        idx = clean_parts.index("resolve")
        repo_prefix = clean_parts[:idx]
        tail = clean_parts[idx + 1 :]

        branch = tail[0] if len(tail) >= 1 else "main"
        resolved_parts = repo_prefix + ["resolve", branch]
        if len(tail) >= 2:
            # keep any subpath, replace final filename with registry filename
            sub_parts = tail[1:-1] if len(tail) > 2 else []
            resolved_parts.extend(sub_parts)
        if filename:
            resolved_parts.append(filename)
        elif len(tail) >= 2:
            resolved_parts.append(tail[-1])
        else:
            resolved_parts.append("model.safetensors")
    else:
        if filename:
            resolved_parts = clean_parts + ["resolve", "main", filename]
        else:
            resolved_parts = clean_parts + ["resolve", "main"]

    new_path = "/" + "/".join(resolved_parts)
    return urlunparse((scheme, host, new_path, "", parsed.query, ""))


def extract_civitai_model_id(url):
    path = urlparse(url).path
    patterns = [
        r"/models/(\d+)",
        r"/api/v1/models/(\d+)",
    ]
    for pattern in patterns:
        m = re.search(pattern, path, flags=re.IGNORECASE)
        if m:
            return m.group(1)
    return None


def canonical_civitai_model_url(model_id):
    return f"https://civitai.com/models/{model_id}"


def fetch_json(url, timeout):
    req = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        },
    )
    with urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="ignore")
    return json.loads(raw)


def extract_slug_from_url(url):
    parts = [p for p in urlparse(url).path.split("/") if p]
    if len(parts) >= 2 and parts[0].lower() == "models":
        token = parts[1]
        if token.isdigit():
            return ""
        return token
    return ""


def score_civitai_candidate(candidate, filename, stem_key, slug_key):
    score = 0
    c_name = str(candidate.get("name", ""))
    c_key = normalize_key(c_name)

    if c_key and (c_key in stem_key or stem_key in c_key):
        score += 20
    if slug_key and c_key and (slug_key in c_key or c_key in slug_key):
        score += 30

    versions = candidate.get("modelVersions") or []
    for version in versions:
        for file_info in (version.get("files") or []):
            f_name = str(file_info.get("name", ""))
            f_key = normalize_key(f_name)
            if not f_key:
                continue
            if f_name.lower() == filename.lower():
                score += 100
            elif stem_key and (stem_key in f_key or f_key in stem_key):
                score += 25

    return score


def resolve_civitai_model_url(current_url, filename, timeout, cache):
    existing_id = extract_civitai_model_id(current_url)
    if existing_id:
        return canonical_civitai_model_url(existing_id), "existing_id"

    stem = filename_stem(filename)
    stem_key = normalize_key(stem)
    slug = extract_slug_from_url(current_url)
    slug_key = normalize_key(slug)

    queries = []
    if slug:
        queries.append(slug)
    if stem:
        queries.append(stem)
    # unique while preserving order
    dedup = []
    seen = set()
    for q in queries:
        qn = q.strip()
        if not qn:
            continue
        low = qn.lower()
        if low in seen:
            continue
        seen.add(low)
        dedup.append(qn)

    best_url = None
    best_score = -1
    best_reason = ""

    for query in dedup:
        cache_key = query.lower()
        if cache_key in cache:
            data = cache[cache_key]
        else:
            api_url = f"https://civitai.com/api/v1/models?query={quote_plus(query)}&limit=30"
            try:
                data = fetch_json(api_url, timeout=timeout)
            except Exception:
                data = None
            cache[cache_key] = data

        if not isinstance(data, dict):
            continue

        for candidate in (data.get("items") or []):
            model_id = candidate.get("id")
            if not model_id:
                continue

            score = score_civitai_candidate(candidate, filename, stem_key, slug_key)
            if score > best_score:
                best_score = score
                best_url = canonical_civitai_model_url(model_id)
                best_reason = f"search:{query}"

    if best_url and best_score >= 30:
        return best_url, best_reason

    return current_url, "unresolved_slug"


def check_url_status(url, timeout):
    # HEAD first, then lightweight GET fallback if HEAD is blocked.
    headers = {"User-Agent": USER_AGENT}
    try:
        req = Request(url, headers=headers, method="HEAD")
        with urlopen(req, timeout=timeout) as resp:
            return int(resp.getcode() or 0), "HEAD"
    except HTTPError as err:
        return int(err.code), "HEAD"
    except Exception:
        pass

    try:
        headers["Range"] = "bytes=0-0"
        req = Request(url, headers=headers, method="GET")
        with urlopen(req, timeout=timeout) as resp:
            return int(resp.getcode() or 0), "GET_RANGE"
    except HTTPError as err:
        return int(err.code), "GET_RANGE"
    except URLError:
        return 0, "NETWORK_ERROR"
    except Exception:
        return 0, "ERROR"


def iter_registry_items(registry, checkpoints_only=False, name_filter=None):
    categories = ["checkpoints"] if checkpoints_only else CATEGORY_SPECS
    pattern = re.compile(name_filter, re.IGNORECASE) if name_filter else None
    for category in categories:
        items = registry.get(category, [])
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict):
                if pattern:
                    name = str(item.get("name", ""))
                    if not pattern.search(name):
                        continue
                yield category, item


def sync_urls(
    registry_path,
    dry_run=False,
    offline=False,
    checkpoints_only=False,
    check_live=False,
    timeout=12,
    name_filter=None,
):
    registry = load_registry(registry_path)
    if not registry:
        print(f"Registry not found at: {registry_path}")
        return 1

    civitai_cache = {}
    scan_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path = os.path.join(os.path.dirname(registry_path), URL_LOG_FILE)

    stats = {
        "scanned": 0,
        "changed": 0,
        "live_checked": 0,
        "live_ok": 0,
        "live_bad": 0,
        "unresolved": 0,
    }
    change_lines = []
    status_lines = []
    unresolved_lines = []

    for category, item in iter_registry_items(
        registry,
        checkpoints_only=checkpoints_only,
        name_filter=name_filter,
    ):
        name = str(item.get("name", "")).strip()
        old_url = str(item.get("url", "")).strip()
        if not name or not old_url:
            continue

        stats["scanned"] += 1
        new_url = old_url
        reason = "unchanged"

        if is_huggingface_url(old_url):
            new_url = normalize_huggingface_url(old_url, name)
            if new_url != old_url:
                reason = "huggingface_normalized"

        elif is_civitai_url(old_url):
            model_id = extract_civitai_model_id(old_url)
            if model_id:
                canonical = canonical_civitai_model_url(model_id)
                if canonical != old_url:
                    new_url = canonical
                    reason = "civitai_canonicalized"
            elif not offline:
                resolved, resolve_reason = resolve_civitai_model_url(
                    old_url,
                    name,
                    timeout=timeout,
                    cache=civitai_cache,
                )
                new_url = resolved
                reason = f"civitai_{resolve_reason}"
                if resolved == old_url and resolve_reason == "unresolved_slug":
                    stats["unresolved"] += 1
                    unresolved_lines.append(f"[{category}] {name}: {old_url}")

        else:
            reason = "unknown_provider"

        if new_url != old_url:
            stats["changed"] += 1
            item["url"] = new_url
            change_lines.append(f"[{category}] {name}: {old_url} -> {new_url} ({reason})")
            print(f"[URL FIXED] [{category}] {name}")

        if check_live:
            code, method = check_url_status(new_url, timeout=timeout)
            stats["live_checked"] += 1
            ok = 200 <= code < 400
            if ok:
                stats["live_ok"] += 1
            else:
                stats["live_bad"] += 1
            status_lines.append(f"[{category}] {name}: {code} via {method} | {new_url}")

    if dry_run:
        print("\nDry run complete. No changes written.")
    elif stats["changed"] > 0:
        save_registry(registry_path, registry)
        print(f"\nUpdated registry URLs: {stats['changed']} changes written.")
    else:
        print("\nNo URL changes needed.")

    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"Scan Date: {scan_time}\n")
        log.write(f"Registry: {registry_path}\n")
        log.write(f"Dry Run: {dry_run}\n")
        log.write(f"Offline: {offline}\n")
        log.write(f"Check Live: {check_live}\n")
        log.write(f"Checkpoints Only: {checkpoints_only}\n")
        log.write(f"Scanned: {stats['scanned']}\n")
        log.write(f"Changed: {stats['changed']}\n")
        log.write(f"Unresolved Slugs: {stats['unresolved']}\n")
        log.write(f"Live Checked: {stats['live_checked']}\n")
        log.write(f"Live OK: {stats['live_ok']}\n")
        log.write(f"Live Bad: {stats['live_bad']}\n")

        log.write("\n== URL Changes ==\n")
        for line in change_lines:
            log.write(f"{line}\n")

        log.write("\n== Live Status ==\n")
        for line in status_lines:
            log.write(f"{line}\n")

        log.write("\n== Unresolved Civitai Slugs ==\n")
        for line in unresolved_lines:
            log.write(f"{line}\n")

    print(f"URL scan log written to: {log_path}")
    print(
        "Summary: "
        f"scanned={stats['scanned']}, "
        f"changed={stats['changed']}, "
        f"unresolved={stats['unresolved']}, "
        f"live_checked={stats['live_checked']}, "
        f"live_bad={stats['live_bad']}"
    )

    return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate and repair source download URLs in model_registry.json."
    )
    parser.add_argument("--registry", default=REGISTRY_FILE, help="Path to registry JSON.")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip network lookups (only normalize existing URL patterns).",
    )
    parser.add_argument(
        "--check-live",
        action="store_true",
        help="Check HTTP status for each final URL (HEAD + GET range fallback).",
    )
    parser.add_argument(
        "--checkpoints-only",
        action="store_true",
        help="Process checkpoints category only.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=12,
        help="HTTP timeout in seconds for URL lookup/check requests.",
    )
    parser.add_argument(
        "--name-filter",
        default="",
        help="Optional regex filter for model names (example: nightvision).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    registry_path = args.registry
    if not os.path.isabs(registry_path):
        registry_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), registry_path)

    rc = sync_urls(
        registry_path=registry_path,
        dry_run=args.dry_run,
        offline=args.offline,
        checkpoints_only=args.checkpoints_only,
        check_live=args.check_live,
        timeout=max(3, int(args.timeout)),
        name_filter=args.name_filter.strip() or None,
    )
    sys.exit(rc)
