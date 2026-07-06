import os
import re
import time
import json
import random
import logging
import urllib.request
import urllib.parse
from typing import Optional

logger = logging.getLogger("agents.effect_downloader")

# Directory configurations
CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "cache", "shared"))
STICKERS_DIR = os.path.join(CACHE_DIR, "stickers")
EFFECTS_DIR = os.path.join(CACHE_DIR, "effects")

LOCAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "data"))
LOCAL_STICKERS_DIR = os.path.join(LOCAL_DIR, "local_stickers")
LOCAL_EFFECTS_DIR = os.path.join(LOCAL_DIR, "local_effects")

def _ensure_directories():
    os.makedirs(STICKERS_DIR, exist_ok=True)
    os.makedirs(EFFECTS_DIR, exist_ok=True)
    os.makedirs(LOCAL_STICKERS_DIR, exist_ok=True)
    os.makedirs(LOCAL_EFFECTS_DIR, exist_ok=True)

def _get_local_fallback(directory: str, normalized_query: str) -> Optional[str]:
    """Scans the local directory for matching files, falling back to any file in that directory."""
    if not os.path.exists(directory):
        return None
        
    candidates = []
    try:
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        if not files:
            return None
            
        # 1. Look for a file containing the query in its name
        for f in files:
            if normalized_query in f.lower():
                candidates.append(os.path.join(directory, f))
                
        if candidates:
            return random.choice(candidates)
            
        # 2. Otherwise return the first/random file in the directory
        return os.path.join(directory, random.choice(files))
    except Exception as e:
        logger.warning(f"Error checking local fallback in {directory}: {e}")
    return None

def download_giphy_sticker(query: str, retries: int = 3) -> Optional[str]:
    """
    Downloads a transparent sticker GIF from Giphy matching the query.
    Retries on network failure, and falls back to local_stickers or None if all fails.
    """
    _ensure_directories()
    if not query:
        return None
        
    normalized = re.sub(r'[^a-zA-Z0-9]', '', query).lower()
    dest_path = os.path.join(STICKERS_DIR, f"{normalized}.gif")
    
    # Check if already cached
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 100:
        logger.info(f"Using cached Giphy sticker for query '{query}': {dest_path}")
        return dest_path
        
    api_key = os.getenv("GIPHY_API_KEY")
    if not api_key:
        logger.warning("GIPHY_API_KEY is not set. Trying local fallback.")
        return _get_local_fallback(LOCAL_STICKERS_DIR, normalized)
        
    encoded_query = urllib.parse.quote_plus(query)
    url = f"https://api.giphy.com/v1/stickers/search?api_key={api_key}&q={encoded_query}&limit=5"
    
    headers = {"User-Agent": "Mozilla/5.0"}
    
    for attempt in range(retries):
        try:
            logger.info(f"Searching Giphy stickers (Attempt {attempt+1}/{retries}) for query: '{query}'")
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                res_data = json.loads(response.read().decode('utf-8'))
                items = res_data.get("data", [])
                if not items:
                    logger.warning(f"No Giphy stickers found for query: '{query}'")
                    break
                    
                # Download the first sticker's fixed height url (ideal size for overlaying)
                img_url = items[0]["images"]["fixed_height"]["url"]
                
                logger.info(f"Downloading sticker from Giphy CDN: {img_url}")
                img_req = urllib.request.Request(img_url, headers=headers)
                with urllib.request.urlopen(img_req, timeout=15) as res:
                    gif_data = res.read()
                    
                with open(dest_path, "wb") as f:
                    f.write(gif_data)
                    
                logger.info(f"Successfully downloaded Giphy sticker: {dest_path}")
                return dest_path
                
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed to download Giphy sticker for query '{query}': {e}")
            if attempt < retries - 1:
                time.sleep(1)
                
    # All downloads failed, try local fallback
    logger.warning(f"All Giphy download retries failed for query '{query}'. Trying local fallback.")
    fallback = _get_local_fallback(LOCAL_STICKERS_DIR, normalized)
    if fallback:
        logger.info(f"Found local fallback sticker: {fallback}")
        return fallback
        
    return None

def download_pixabay_effect(query: str, retries: int = 3) -> Optional[str]:
    """
    Downloads an overlay loop video from Pixabay matching the query.
    Retries on network failure, and falls back to local_effects or None if all fails.
    """
    _ensure_directories()
    if not query:
        return None
        
    normalized = re.sub(r'[^a-zA-Z0-9]', '', query).lower()
    dest_path = os.path.join(EFFECTS_DIR, f"{normalized}.mp4")
    
    # Check if already cached
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        logger.info(f"Using cached Pixabay effect for query '{query}': {dest_path}")
        return dest_path
        
    api_key = os.getenv("PIXABAY_API_KEY")
    if not api_key:
        logger.warning("PIXABAY_API_KEY is not set. Trying local fallback.")
        return _get_local_fallback(LOCAL_EFFECTS_DIR, normalized)
        
    search_query = query
    if "black background" not in query.lower():
        search_query = f"{query} black background"

    encoded_query = urllib.parse.quote_plus(search_query)
    url = f"https://pixabay.com/api/videos/?key={api_key}&q={encoded_query}&video_type=film&per_page=20"
    
    headers = {"User-Agent": "Mozilla/5.0"}
    
    for attempt in range(retries):
        try:
            logger.info(f"Searching Pixabay videos (Attempt {attempt+1}/{retries}) for query: '{search_query}'")
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                res_data = json.loads(response.read().decode('utf-8'))
                hits = res_data.get("hits", [])
                if not hits:
                    logger.warning(f"No Pixabay videos found for query: '{query}'")
                    break
                    
                # Filter hits to find clean overlay loops (no scenic/landscape/nature footage)
                selected_hit = None
                scenic_keywords = {
                    "forest", "nature", "mountain", "sky", "trees", "landscape", "sea", 
                    "river", "lake", "ocean", "field", "grass", "sun", "sunset", "sunrise",
                    "scenery", "road", "city", "street", "car", "people", "man", "woman",
                    "house", "building", "real", "aerial", "drone", "beach", "clouds",
                    "park", "hill", "desert", "garden", "flower", "animal", "bird"
                }
                
                for hit in hits:
                    tags = [t.strip().lower() for t in hit.get("tags", "").split(",")]
                    is_clean = True
                    for tag in tags:
                        for kw in scenic_keywords:
                            if kw in tag or tag in kw:
                                is_clean = False
                                break
                        if not is_clean:
                            break
                    if is_clean:
                        selected_hit = hit
                        logger.info(f"Selected clean Pixabay overlay hit with tags: {hit.get('tags')}")
                        break
                        
                if not selected_hit:
                    logger.warning("No clean overlay hit found in Pixabay search results. Falling back to the first result.")
                    selected_hit = hits[0]
                    
                # Get the download URL for small or medium video format
                video_res = selected_hit.get("videos", {})
                # Try medium first, then small, then tiny, then large
                video_info = video_res.get("medium") or video_res.get("small") or video_res.get("tiny") or video_res.get("large")
                if not video_info or not video_info.get("url"):
                    logger.warning("Could not find video download URL in Pixabay hit.")
                    break
                    
                video_url = video_info["url"]
                logger.info(f"Downloading effect video from Pixabay CDN: {video_url}")
                
                # Fetch video file
                video_req = urllib.request.Request(video_url, headers=headers)
                with urllib.request.urlopen(video_req, timeout=20) as res:
                    video_data = res.read()
                    
                with open(dest_path, "wb") as f:
                    f.write(video_data)
                    
                logger.info(f"Successfully downloaded Pixabay effect: {dest_path}")
                return dest_path
                
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed to download Pixabay effect for '{query}': {e}")
            if attempt < retries - 1:
                time.sleep(1)
                
    # All downloads failed, try local fallback
    logger.warning(f"All Pixabay download retries failed for query '{query}'. Trying local fallback.")
    fallback = _get_local_fallback(LOCAL_EFFECTS_DIR, normalized)
    if fallback:
        logger.info(f"Found local fallback effect: {fallback}")
        return fallback
        
    return None
