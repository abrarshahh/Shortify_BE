import os
import re
import logging
import urllib.request
import urllib.parse
from fontTools.ttLib import TTFont

logger = logging.getLogger("agents.font_downloader")

FONTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "data", "fonts"))

def validate_font_file(file_path: str) -> bool:
    """Uses fonttools to validate the TTF file structure."""
    if not os.path.exists(file_path) or os.path.getsize(file_path) < 1000:
        return False
    try:
        font = TTFont(file_path)
        # Check standard required tables to verify it is not corrupt
        font.get('head')
        font.get('name')
        font.close()
        return True
    except Exception as e:
        logger.warning(f"Font validation failed for {file_path}: {e}")
        return False

def get_font_family_name(file_path: str) -> str:
    """Extracts the official font family name using fonttools metadata."""
    try:
        font = TTFont(file_path)
        name_table = font['name']
        for record in name_table.names:
            if record.nameID == 1:  # Family Name
                try:
                    name_str = record.toUnicode()
                    font.close()
                    return name_str
                except Exception:
                    pass
        font.close()
    except Exception:
        pass
    return "Unknown"

def download_font_from_google(font_name: str, weight: int = 700) -> str:
    """
    Downloads a font by family name from the Google Fonts CSS API
    using a User-Agent that triggers TTF font format response.
    """
    os.makedirs(FONTS_DIR, exist_ok=True)
    
    # Normalize font name for filenames
    normalized_name = re.sub(r'[^a-zA-Z0-9]', '', font_name).lower()
    dest_path = os.path.join(FONTS_DIR, f"{normalized_name}_{weight}.ttf")
    
    # 1. Check if already cached and valid
    if validate_font_file(dest_path):
        official_name = get_font_family_name(dest_path)
        logger.info(f"Using cached Google Font: {font_name} (Official: {official_name}) at {dest_path}")
        return dest_path
        
    # 2. Try fetching from Google Fonts API
    font_family_escaped = urllib.parse.quote_plus(font_name)
    css_url = f"https://fonts.googleapis.com/css2?family={font_family_escaped}:wght@{weight}"
    
    # Custom older Android User-Agent to force the CSS API to return .ttf format
    headers = {
        "User-Agent": "Mozilla/5.0 (Linux; U; Android 2.2; en-us; Nexus One Build/FRF91) AppleWebKit/533.1 (KHTML, like Gecko) Version/4.0 Mobile Safari/533.1"
    }
    
    logger.info(f"Fetching font CSS from Google: {css_url}")
    try:
        req = urllib.request.Request(css_url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            css_content = response.read().decode('utf-8')
            
        # Parse the url(...) target out of the CSS content
        match = re.search(r"url\((https://fonts\.gstatic\.com/[^)]+)\)", css_content)
        if not match:
            raise ValueError("Could not parse font file URL from Google Fonts CSS response.")
            
        font_url = match.group(1)
        logger.info(f"Downloading TTF from Google CDN: {font_url}")
        
        # Download font file
        font_req = urllib.request.Request(font_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(font_req, timeout=15) as res:
            font_data = res.read()
            
        # Write to temporary file first for validation
        temp_dest = dest_path + ".tmp"
        with open(temp_dest, "wb") as f:
            f.write(font_data)
            
        if validate_font_file(temp_dest):
            os.replace(temp_dest, dest_path)
            official_name = get_font_family_name(dest_path)
            logger.info(f"Successfully cached Google Font: {font_name} (Official: {official_name})")
            return dest_path
        else:
            if os.path.exists(temp_dest):
                os.remove(temp_dest)
            raise ValueError("Downloaded file failed fonttools validation.")
            
    except Exception as e:
        logger.warning(f"Failed to download Google Font '{font_name}': {e}")
        
    return ""

def get_font_path(font_name: str, weight: int = 700) -> str:
    """
    Tries to retrieve the path to the requested font.
    First tries Google Fonts downloader/cacher, and falls back to standard system fonts.
    """
    # Try downloading/retrieving the font
    if font_name:
        path = download_font_from_google(font_name, weight)
        if path:
            return path
            
    # System Fallback
    system_candidates = [
        "C:/Windows/Fonts/Impact.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/verdana.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for candidate in system_candidates:
        if os.path.exists(candidate):
            return candidate
            
    raise FileNotFoundError("No valid fonts could be loaded or found on the system.")
