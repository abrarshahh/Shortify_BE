from typing import Dict, Any, Optional

# Predefined text style templates
TEXT_PRESETS: Dict[str, Dict[str, Any]] = {
    "bold_hype": {
        "font_name": "Impact",
        "font_color": "yellow",
        "font_size": 75,
        "outline_color": "black",
        "outline_width": 4,
        "default_animation": "slide_up"
    },
    "classic_clean": {
        "font_name": "Arial",
        "font_color": "white",
        "font_size": 70,
        "outline_color": "black",
        "outline_width": 2,
        "default_animation": "fade"
    },
    "neon_glow": {
        "font_name": "Courier",
        "font_color": "#00FFCC",
        "font_size": 80,
        "outline_color": "none",
        "outline_width": 0,
        "default_animation": "fade"
    },
    "minimal_pop": {
        "font_name": "Arial",
        "font_color": "white",
        "font_size": 65,
        "outline_color": "none",
        "outline_width": 0,
        "default_animation": "slide_down"
    }
}
