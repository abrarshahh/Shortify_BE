import os
from backend_ai.utils.asset_manager import (
    resolve_asset_path, ensure_asset_directories, STICKERS_DIR, OVERLAYS_DIR, SFX_DIR
)

def test_asset_manager_directories_creation():
    # Run ensure to make sure directories are created
    ensure_asset_directories()
    
    assert os.path.exists(STICKERS_DIR)
    assert os.path.exists(OVERLAYS_DIR)
    assert os.path.exists(SFX_DIR)

def test_asset_manager_mock_assets():
    ensure_asset_directories()
    
    # Check that PNGs/GIFs exist
    assert os.path.exists(os.path.join(STICKERS_DIR, "highlight_arrow.png"))
    assert os.path.exists(os.path.join(STICKERS_DIR, "fire_emoji.png"))
    assert os.path.exists(os.path.join(STICKERS_DIR, "subscribe_pop.gif"))
    
    # Check that MP4 overlays exist
    assert os.path.exists(os.path.join(OVERLAYS_DIR, "film_grain.mp4"))
    assert os.path.exists(os.path.join(OVERLAYS_DIR, "smoke.mp4"))
    
    # Check that SFX exist
    assert os.path.exists(os.path.join(SFX_DIR, "whoosh.wav"))
    assert os.path.exists(os.path.join(SFX_DIR, "bass_drop.wav"))

def test_resolve_asset_path():
    ensure_asset_directories()
    
    # Valid asset resolutions
    path_leak = resolve_asset_path("overlays", "overlay_light_leak")
    assert path_leak is not None
    assert path_leak.endswith("light_leak.mp4")
    assert os.path.exists(path_leak)
    
    path_subscribe = resolve_asset_path("stickers", "sticker_subscribe")
    assert path_subscribe is not None
    assert path_subscribe.endswith("subscribe_pop.gif")
    assert os.path.exists(path_subscribe)
    
    path_whoosh = resolve_asset_path("sfx", "sfx_whoosh")
    assert path_whoosh is not None
    assert path_whoosh.endswith("whoosh.wav")
    assert os.path.exists(path_whoosh)
    
    # Invalid resolution
    assert resolve_asset_path("stickers", "non_existent_id") is None
    assert resolve_asset_path("invalid_type", "overlay_light_leak") is None
