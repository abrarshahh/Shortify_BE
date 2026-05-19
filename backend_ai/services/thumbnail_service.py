import os
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
from moviepy.video.io.VideoFileClip import VideoFileClip
from typing import Optional

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
        overlay_text: Optional[str] = None
    ) -> str:
        """
        Generates a stylized thumbnail.jpg from the video clip.
        """
        print(f"ThumbnailAgent: generating cover thumbnail from {video_path}...")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        output_path = os.path.join(output_dir, "thumbnail.jpg")
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Extract frame at 1.0s (hook start)
        try:
            with VideoFileClip(video_path) as clip:
                duration = clip.duration
                sample_t = min(1.0, max(0.0, duration - 0.1))
                frame = clip.get_frame(sample_t)
                img = Image.fromarray(frame)
        except Exception as e:
            print(f"  Warning: MoviePy frame extraction failed: {e}. Creating fallback gray thumbnail.")
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
            print(f"  Warning: Color adjustments failed: {e}")

        # 3. Text Overlay with legibility protections (drop shadow + black border)
        if overlay_text:
            try:
                self._draw_overlay_text(img, overlay_text)
            except Exception as e:
                print(f"  Warning: Subtitle text overlay failed: {e}")

        img.save(output_path, "JPEG", quality=95)
        print(f"ThumbnailAgent: thumbnail successfully generated at {output_path}")
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

    def _draw_overlay_text(self, img: Image.Image, text: str):
        """Draws large, professional dynamic subtitle text with drop shadow in center."""
        draw = ImageDraw.Draw(img)
        width, height = img.size
        
        # Clean text
        text = text.upper().strip()
        if len(text) > 40:
            text = text[:37] + "..."
            
        # Select professional bold font if available, fallback otherwise
        font = None
        font_paths = [
            "C:\\Windows\\Fonts\\Arial.ttf",
            "C:\\Windows\\Fonts\\arialbd.ttf",
            "C:\\Windows\\Fonts\\Impact.ttf",
            "C:\\Windows\\Fonts\\tahoma.ttf",
        ]
        
        font_size = int(height * 0.08) # Dynamic font size based on height
        
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

        # Main text (vibrant aesthetic yellow)
        draw.text((x, y), text, font=font, fill=(255, 220, 0))


if __name__ == "__main__":
    agent = ThumbnailAgent()
    print("ThumbnailAgent initialized successfully.")
