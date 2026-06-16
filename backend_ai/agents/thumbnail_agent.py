import os
import logging
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
from typing import Optional, Dict, Any

logger = logging.getLogger("agents.thumbnail")

class ThumbnailAgent:
    """
    Phase 9: Automated Thumbnail Generation Agent.
    Extracts a frame from the hook video, applies professional color enhancements
    (saturation boost, high contrast, sharpening, vignette), and overlays text
    for high click-through-rate (CTR) visuals.
    """

    def __init__(self):
        pass

    def generate_thumbnail(
        self,
        video_path: str,
        output_dir: str,
        overlay_text: Optional[str] = None,
        style: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generates a stylized thumbnail.jpg from the video clip.
        """
        logger.info(f"Generating thumbnail from: {video_path}")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        output_path = os.path.join(output_dir, "thumbnail.jpg")
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Extract frame at 1.0s (hook start)
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError("Could not open video file via cv2")
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0.0
            
            sample_t = min(1.0, max(0.0, duration - 0.1))
            cap.set(cv2.CAP_PROP_POS_MSEC, sample_t * 1000.0)
            success, frame_bgr = cap.read()
            cap.release()
            
            if not success:
                raise RuntimeError("Could not read frame from video via cv2")
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
        except Exception as e:
            logger.warning(f"OpenCV frame extraction failed: {e}. Creating fallback gray thumbnail")
            img = Image.new("RGB", (1280, 720), color=(80, 80, 80))

        # 2. Apply high-contrast and saturation enhancements
        try:
            # Saturation boost (1.35x)
            color_enhancer = ImageEnhance.Color(img)
            img = color_enhancer.enhance(1.35)
            
            # Contrast boost (1.20x)
            contrast_enhancer = ImageEnhance.Contrast(img)
            img = contrast_enhancer.enhance(1.20)
            
            # Sharpness boost (1.5x)
            sharpness_enhancer = ImageEnhance.Sharpness(img)
            img = sharpness_enhancer.enhance(1.50)
            
            # Apply subtle vignette effect
            img = self._apply_vignette(img)
        except Exception as e:
            logger.warning(f"Color adjustments failed: {e}")

        # 3. Text Overlay with legibility protections (drop shadow + black border)
        if overlay_text:
            try:
                self._draw_overlay_text(img, overlay_text, style=style)
            except Exception as e:
                logger.warning(f"Text overlay failed: {e}")

        img.save(output_path, "JPEG", quality=95)
        logger.info(f"Thumbnail generated -> {output_path}")
        return output_path

    def _apply_vignette(self, img: Image.Image) -> Image.Image:
        """Applies a high-quality radial gradient vignette mask."""
        width, height = img.size
        
        # Create gradient mask
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        X, Y = np.meshgrid(x, y)
        
        # Radial distance from center (scaled 0 to 1)
        r = np.sqrt(X**2 + Y**2)
        
        # Darkening factor: 1.0 at center, fading down to 0.4 at corners
        vignette_mask = np.clip(1.0 - (r * 0.45), 0.35, 1.0)
        
        # Convert back to image
        arr = np.array(img, dtype=np.float32)
        for c in range(3): # Apply to R, G, B
            arr[:, :, c] *= vignette_mask
            
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    def _draw_overlay_text(self, img: Image.Image, text: str, style: Optional[Dict[str, Any]] = None):
        """Draws large, professional dynamic subtitle text with drop shadow in center."""
        draw = ImageDraw.Draw(img)
        width, height = img.size
        
        # Clean text
        text = text.upper().strip()
        if len(text) > 40:
            text = text[:37] + "..."
            
        # Resolve font from subtitle style configuration
        from backend_ai.core.config_loader import AGENTS_CONFIG
        from backend_ai.utils.font_downloader import get_font_path
        
        if style:
            style_cfg = style.get("text_overlay_style") or style.get("subtitle_style") or style
            font_name = style_cfg.get("font_name", "Arial")
            font_color = style_cfg.get("font_color", "yellow")
            font_weight = style_cfg.get("font_weight", 700)
        else:
            style_name = AGENTS_CONFIG.get("subtitle_agent", {}).get("caption_style", "hormozi")
            caption_style_cfg = AGENTS_CONFIG.get("caption_styles", {}).get(style_name, {})
            font_name = caption_style_cfg.get("font_name", "Arial")
            font_color = caption_style_cfg.get("font_color", "yellow")
            font_weight = caption_style_cfg.get("font_weight", 700)
        
        font_size = int(height * 0.08)  # Dynamic font size based on height
        
        font = None
        try:
            fp = get_font_path(font_name, font_weight)
            font = ImageFont.truetype(fp, font_size)
        except Exception as e:
            logger.warning(f"Could not resolve font '{font_name}' for thumbnail: {e}")
            
        if font is None:
            # Fallback candidates
            font_paths = [
                "C:\\Windows\\Fonts\\Arial.ttf",
                "C:\\Windows\\Fonts\\arialbd.ttf",
                "C:\\Windows\\Fonts\\Impact.ttf",
                "C:\\Windows\\Fonts\\tahoma.ttf",
            ]
            for fp in font_paths:
                if os.path.exists(fp):
                    try:
                        font = ImageFont.truetype(fp, font_size)
                        break
                    except Exception:
                        continue

        if font is None:
            font = ImageFont.load_default()

        # Calculate bounding box of text for center alignment
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except AttributeError:
            # Fallback for old PIL versions
            text_width, text_height = draw.textsize(text, font=font)

        x = (width - text_width) // 2
        y = (height - text_height) // 2 + int(height * 0.15) # Place slightly lower

        # Draw drop shadow (offset offset x, y by 4px)
        draw.text((x + 4, y + 4), text, font=font, fill=(0, 0, 0))
        # Draw clean border outline for maximum visibility
        for dx, dy in [(-2, -2), (-2, 2), (2, -2), (2, 2), (0, -2), (0, 2), (-2, 0), (2, 0)]:
            draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0))

        # Main text (vibrant aesthetic dynamic color)
        draw.text((x, y), text, font=font, fill=font_color)


if __name__ == "__main__":
    agent = ThumbnailAgent()
    logger.info("ThumbnailAgent initialized successfully.")
