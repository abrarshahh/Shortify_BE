import os
import logging
from typing import Optional, Dict

logger = logging.getLogger("utils.asset_manager")

# Root assets directory path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
ASSETS_DIR = os.path.join(BASE_DIR, "data", "assets")

OVERLAYS_DIR = os.path.join(ASSETS_DIR, "overlays")
STICKERS_DIR = os.path.join(ASSETS_DIR, "stickers")
SFX_DIR = os.path.join(ASSETS_DIR, "sfx")

# Pre-defined professional Asset ID mappings to default filenames
ASSET_CATALOG: Dict[str, Dict[str, str]] = {
    "overlays": {
        "overlay_film_grain": "film_grain.mp4",
        "overlay_light_leak": "light_leak.mp4",
        "overlay_particles": "particles.mp4",
        "overlay_smoke": "smoke.mp4",
    },
    "stickers": {
        "sticker_subscribe": "subscribe_pop.gif",
        "sticker_arrow": "highlight_arrow.png",
        "sticker_fire": "fire_emoji.png",
    },
    "sfx": {
        "sfx_whoosh": "whoosh.wav",
        "sfx_swoosh": "transition_swoosh.mp3",
        "sfx_bass_drop": "bass_drop.wav",
    }
}

import subprocess

def ensure_asset_directories():
    """Ensures all standard local asset directories and sample mock assets exist on disk."""
    os.makedirs(OVERLAYS_DIR, exist_ok=True)
    os.makedirs(STICKERS_DIR, exist_ok=True)
    os.makedirs(SFX_DIR, exist_ok=True)

    # Generate 1x1 transparent PNG
    png_path = os.path.join(STICKERS_DIR, "highlight_arrow.png")
    if not os.path.exists(png_path):
        try:
            png_bytes = bytes.fromhex("89504E470D0A1A0A0000000D49484452000000010000000108060000001F15C4890000000A49444154789C63000100000500010D0A2DB40000000049454E44AE426082")
            with open(png_path, "wb") as f:
                f.write(png_bytes)
            # Create fire_emoji.png as well
            with open(os.path.join(STICKERS_DIR, "fire_emoji.png"), "wb") as f:
                f.write(png_bytes)
        except Exception as e:
            logger.warning(f"Failed to generate highlight_arrow.png: {e}")

    # Generate 1x1 transparent GIF
    gif_path = os.path.join(STICKERS_DIR, "subscribe_pop.gif")
    if not os.path.exists(gif_path):
        try:
            gif_bytes = bytes.fromhex("47494638396101000100800000000000ffffff21f90401000000002c00000000010001000002024401003b")
            with open(gif_path, "wb") as f:
                f.write(gif_bytes)
        except Exception as e:
            logger.warning(f"Failed to generate subscribe_pop.gif: {e}")

    # Generate 5-second color overlay loops using FFmpeg
    overlays = {
        "film_grain.mp4": "black",
        "light_leak.mp4": "orange",
        "particles.mp4": "gold",
        "smoke.mp4": "gray"
    }
    for filename, color in overlays.items():
        out_path = os.path.join(OVERLAYS_DIR, filename)
        if not os.path.exists(out_path):
            try:
                # Use ffmpeg via subprocess to generate silent black/color video
                cmd = [
                    "ffmpeg", "-f", "lavfi", "-i", f"color=c={color}:s=640x360:d=10",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-y", out_path
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            except Exception as e:
                logger.warning(f"Failed to generate mock video {filename} using FFmpeg: {e}")

    # Generate 2-second silent audio loops using FFmpeg
    sfxs = {
        "whoosh.wav": 2,
        "transition_swoosh.mp3": 2,
        "bass_drop.wav": 3
    }
    for filename, duration in sfxs.items():
        out_path = os.path.join(SFX_DIR, filename)
        if not os.path.exists(out_path):
            try:
                cmd = [
                    "ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
                    "-t", str(duration), "-y", out_path
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            except Exception as e:
                logger.warning(f"Failed to generate mock audio {filename} using FFmpeg: {e}")

def resolve_asset_path(asset_type: str, asset_id: str) -> Optional[str]:
    """
    Resolves a curated Asset ID (e.g. 'overlay_light_leak') to its absolute path on disk.
    Returns None if the asset ID is unknown or the file does not exist.
    """
    ensure_asset_directories()
    
    if asset_type not in ASSET_CATALOG:
        logger.warning(f"Unknown asset type: {asset_type}")
        return None
        
    catalog = ASSET_CATALOG[asset_type]
    if asset_id not in catalog:
        logger.warning(f"Asset ID '{asset_id}' is not registered in the '{asset_type}' catalog.")
        return None
        
    filename = catalog[asset_id]
    
    # Locate target folder
    if asset_type == "overlays":
        target_dir = OVERLAYS_DIR
    elif asset_type == "stickers":
        target_dir = STICKERS_DIR
    else:  # sfx
        target_dir = SFX_DIR
        
    full_path = os.path.join(target_dir, filename)
    if os.path.exists(full_path):
        return full_path
        
    logger.warning(f"Curated asset file '{filename}' for ID '{asset_id}' not found on disk at: {full_path}")
    return None
