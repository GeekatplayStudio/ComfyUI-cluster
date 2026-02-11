import re

def _score_tags_advanced(prompt: str, tags: list[str]) -> int:
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

# ... existing code ...
