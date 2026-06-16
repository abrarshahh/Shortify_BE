import os
import json
import logging
import warnings
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger("agents.subtitle")

# Suppress FP16 warning on CPU
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")


# ---------------------------------------------------------------------------
# Safe-Zone constants for vertical short-form (1080x1920)
# ---------------------------------------------------------------------------
# These are the "danger" regions where platform UI overlays live:
#   TikTok / Reels / Shorts:
#     - Top  bar  (status + camera icon): top 100px
#     - Bottom bar (CTA + captions):      bottom 300px
#     - Right side (like/share buttons):  right 120px

SAFE_ZONE = {
    "top_px":    100,
    "bottom_px": 300,
    "left_px":   0,
    "right_px":  120,
}


class SubtitleAgent:
    """
    Phase 6: Adds auto-generated subtitles/captions to the rendered video
    and validates that text overlays land in the platform's safe zone.

    Two main responsibilities:
    1. transcribe(video_path)  — local Whisper transcription with word-level
                                 timestamps, returns SRT-style caption list.
    2. check_safe_zones(edl)   — validates every text_overlay in the EDL
                                 against TikTok/Reels safe-zone rules.
    """

    def __init__(self, model_size: str = "base", device: str = "cpu", caption_style: str = "hormozi"):
        """
        Args:
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large').
                        'base' is a good balance of speed and accuracy.
            device:     'cpu' or 'cuda'
            caption_style: subtitle style config option ('minimal', 'bold', 'hormozi', 'outline')
        """
        self.model_size = model_size
        self.device = device
        self.caption_style = caption_style
        self._model = None  # Lazy-load

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self):
        """Lazy-loads the Whisper model on first use."""
        if self._model is None:
            import whisper
            logger.info(f"Loading Whisper model '{self.model_size}' on {self.device}...")
            self._model = whisper.load_model(self.model_size, device=self.device)
            logger.info("Whisper model loaded successfully")
        return self._model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(self, video_path: str) -> Dict[str, Any]:
        """
        Transcribes audio from a video file using local Whisper.

        Returns a dict with:
          - "full_text"  : complete transcript as a string
          - "language"   : detected language code (e.g. "en")
          - "segments"   : list of segment dicts with start, end, text
          - "captions"   : SRT-formatted list ready for burning onto video
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        model = self._load_model()
        logger.info(f"Transcribing audio: {video_path}")

        # word_timestamps=True gives per-word timing (Whisper >= 20230314)
        result = model.transcribe(
            video_path,
            word_timestamps=True,
            verbose=False
        )

        segments = result.get("segments", [])
        captions = self._build_captions(segments)

        return {
            "full_text": result.get("text", "").strip(),
            "language": result.get("language", "unknown"),
            "segments": [
                {
                    "start": round(seg["start"], 2),
                    "end":   round(seg["end"],   2),
                    "text":  seg["text"].strip()
                }
                for seg in segments
            ],
            "captions": captions
        }

    def burn_subtitles(
        self,
        video_path: str,
        captions: List[Dict[str, Any]],
        output_path: str,
        font_size: int = 52,
        font_color: str = "white",
        stroke_color: str = "black",
        stroke_width: int = 2,
        position_y_ratio: float = 0.80,   # 80% down the frame — inside safe zone
        style: Optional[str] = None,
    ) -> str:
        """
        Burns caption text onto the video using FFmpeg and drawtext filters written to a filter script.
        """
        import uuid
        import subprocess
        import cv2
        from PIL import ImageFont
        from backend_ai.core.config_loader import AGENTS_CONFIG

        # Resolve style name and details later

        # Fallback styles

        # Fallback styles
        DEFAULT_STYLES = {
            "minimal": {
                "font_size": 32,
                "font_color": "white",
                "outline_color": "none",
                "back_color": "none",
                "position_y_ratio": 0.85,
                "uppercase": False,
                "animate": False
            },
            "bold": {
                "font_size": 52,
                "font_color": "white",
                "outline_color": "black",
                "outline_width": 4,
                "back_color": "none",
                "position_y_ratio": 0.80,
                "uppercase": True,
                "animate": False
            },
            "outline": {
                "font_size": 64,
                "font_color": "white",
                "outline_color": "black",
                "outline_width": 4,
                "back_color": "none",
                "position_y_ratio": 0.80,
                "uppercase": True,
                "animate": False
            },
            "hormozi": {
                "font_size": 44,
                "font_color": "yellow",
                "inactive_color": "white",
                "outline_color": "none",
                "back_color": "black@0.5",
                "shadow_color": "black",
                "shadow_width": 2,
                "position_y_ratio": 0.80,
                "uppercase": True,
                "animate": True
            },
            "subtitle": {
                "font_size": 28,
                "font_color": "white",
                "outline_color": "none",
                "back_color": "black",
                "position_y_ratio": 0.90,
                "uppercase": False,
                "animate": False
            }
        }

        # Resolve config overrides
        style_cfg = {}
        if isinstance(style, dict):
            style_cfg = style.get("subtitle_style", style)
            style_name = style.get("style_name", "custom")
            logger.info(f"burn_subtitles: Using dynamic style: {style_name}")
        else:
            style_name = style if style else getattr(self, "caption_style", "hormozi")
            logger.info(f"burn_subtitles: Using style '{style_name}'")
            styles_from_config = AGENTS_CONFIG.get("caption_styles", {})
            style_cfg = styles_from_config.get(style_name) or DEFAULT_STYLES.get(style_name, DEFAULT_STYLES["hormozi"])

        # Determine font file path and weight
        font_name_cfg = style_cfg.get("font_name", "Arial")
        font_weight_cfg = style_cfg.get("font_weight", 700)
        font_path = self._find_font(font_name_cfg, font_weight_cfg)
        font_path_escaped = font_path.replace('\\', '/').replace(':', '\\:')

        cfg_font_size = style_cfg.get("font_size", font_size)
        cfg_font_color = style_cfg.get("font_color", font_color)
        cfg_outline_color = style_cfg.get("outline_color", stroke_color)
        cfg_outline_width = style_cfg.get("outline_width", stroke_width)
        cfg_back_color = style_cfg.get("back_color", "none")
        cfg_shadow_color = style_cfg.get("shadow_color", "none")
        cfg_shadow_width = style_cfg.get("shadow_width", 0)
        cfg_position_y_ratio = style_cfg.get("position_y_ratio", position_y_ratio)
        cfg_uppercase = style_cfg.get("uppercase", False)
        cfg_animate = style_cfg.get("animate", False)
        cfg_inactive_color = style_cfg.get("inactive_color", "white")

        # Outline and shadow checks
        has_outline = cfg_outline_color != "none" and cfg_outline_width > 0
        cfg_has_shadow_bool = style_cfg.get("has_shadow", True)
        has_shadow = cfg_has_shadow_bool and cfg_shadow_color != "none" and cfg_shadow_width > 0
        has_box = cfg_back_color != "none"

        # Load font in Pillow to measure text dimensions
        try:
            pil_font = ImageFont.truetype(font_path, cfg_font_size)
        except Exception as e:
            logger.warning(f"Could not load font {font_path} in Pillow: {e}. Falling back to default.")
            pil_font = ImageFont.load_default()

        # Query video dimensions
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if frame_width == 0 or frame_height == 0:
            frame_width, frame_height = 1080, 1920

        all_filters = []

        for chunk in captions:
            chunk_start = float(chunk["start"])
            chunk_end = float(chunk["end"])
            chunk_words = chunk.get("words", [])

            # Synthesize word timings if they don't exist
            if not chunk_words:
                text_clean = chunk.get("text", "").strip()
                words_list = text_clean.split()
                if words_list:
                    dur = chunk_end - chunk_start
                    word_dur = dur / len(words_list)
                    chunk_words = []
                    for idx, w in enumerate(words_list):
                        w_start = chunk_start + idx * word_dur
                        w_end = w_start + word_dur
                        chunk_words.append({
                            "word": w,
                            "start": round(w_start, 2),
                            "end": round(w_end, 2)
                        })

            if not chunk_words:
                continue

            # Greedily group words into lines up to 70% of frame width
            lines = []
            current_line = []
            for w in chunk_words:
                w_text = w["word"].upper() if cfg_uppercase else w["word"]
                test_words = current_line + [w]
                test_line_text = " ".join([x["word"].upper() if cfg_uppercase else x["word"] for x in test_words])
                line_w = pil_font.getlength(test_line_text)
                if current_line and line_w > 0.70 * frame_width:
                    lines.append(current_line)
                    current_line = [w]
                else:
                    current_line.append(w)
            if current_line:
                lines.append(current_line)

            # Compute stacked vertical layout
            N = len(lines)
            line_height = int(cfg_font_size * 1.3)
            total_height = N * line_height
            y_center = frame_height * cfg_position_y_ratio
            y_start = y_center - (total_height / 2)

            for i, line_words in enumerate(lines):
                line_y = int(y_start + i * line_height)
                line_text = " ".join([w["word"].upper() if cfg_uppercase else w["word"] for w in line_words])
                line_width = pil_font.getlength(line_text)

                escaped_line_text = line_text.replace('\\', '\\\\').replace("'", "\\'").replace('%', '\\%')

                if cfg_animate and style_name in ["hormozi", "custom"]:
                    # 1. Draw inactive line layer (contains box and shadow/outline if set)
                    inactive_drawtext = [
                        f"drawtext=fontfile='{font_path_escaped}'",
                        f"text='{escaped_line_text}'",
                        f"fontsize={cfg_font_size}",
                        f"fontcolor={cfg_inactive_color}",
                        f"x='(w-{int(line_width)})/2'",
                        f"y='{line_y}'",
                        f"enable='between(t,{chunk_start:.3f},{chunk_end:.3f})'"
                    ]

                    if has_outline:
                        inactive_drawtext.append(f"borderw={cfg_outline_width}")
                        inactive_drawtext.append(f"bordercolor={cfg_outline_color}")

                    if has_shadow:
                        inactive_drawtext.append(f"shadowx={cfg_shadow_width}")
                        inactive_drawtext.append(f"shadowy={cfg_shadow_width}")
                        inactive_drawtext.append(f"shadowcolor={cfg_shadow_color}")

                    if has_box:
                        inactive_drawtext.append("box=1")
                        inactive_drawtext.append(f"boxcolor={cfg_back_color}")
                        inactive_drawtext.append("boxborderw=10")

                    all_filters.append(":".join(inactive_drawtext))

                    # 2. Draw active word highlight layers
                    for k, w_info in enumerate(line_words):
                        word_text = w_info["word"].upper() if cfg_uppercase else w_info["word"]
                        w_start = float(w_info["start"])
                        w_end = float(w_info["end"])
                        if w_end <= w_start:
                            w_end = w_start + 0.05

                        # Calculate offset of this word in the line
                        prefix_words = line_words[:k]
                        if prefix_words:
                            prefix_text = " ".join([x["word"].upper() if cfg_uppercase else x["word"] for x in prefix_words]) + " "
                            word_offset_x = pil_font.getlength(prefix_text)
                        else:
                            word_offset_x = 0.0

                        escaped_word_text = word_text.replace('\\', '\\\\').replace("'", "\\'").replace('%', '\\%')

                        active_drawtext = [
                            f"drawtext=fontfile='{font_path_escaped}'",
                            f"text='{escaped_word_text}'",
                            f"fontsize={cfg_font_size}",
                            f"fontcolor={cfg_font_color}",
                            f"x='(w-{int(line_width)})/2+{int(word_offset_x)}'",
                            f"y='{line_y}'",
                            f"enable='between(t,{w_start:.3f},{w_end:.3f})'"
                        ]

                        if has_outline:
                            active_drawtext.append(f"borderw={cfg_outline_width}")
                            active_drawtext.append(f"bordercolor={cfg_outline_color}")

                        if has_shadow:
                            active_drawtext.append(f"shadowx={cfg_shadow_width}")
                            active_drawtext.append(f"shadowy={cfg_shadow_width}")
                            active_drawtext.append(f"shadowcolor={cfg_shadow_color}")

                        # NO BOX on active layer to avoid overlaying boxes
                        all_filters.append(":".join(active_drawtext))
                else:
                    # Draw static/non-animated text layer
                    static_drawtext = [
                        f"drawtext=fontfile='{font_path_escaped}'",
                        f"text='{escaped_line_text}'",
                        f"fontsize={cfg_font_size}",
                        f"fontcolor={cfg_font_color}",
                        f"x='(w-{int(line_width)})/2'",
                        f"y='{line_y}'",
                        f"enable='between(t,{chunk_start:.3f},{chunk_end:.3f})'"
                    ]

                    if has_outline:
                        static_drawtext.append(f"borderw={cfg_outline_width}")
                        static_drawtext.append(f"bordercolor={cfg_outline_color}")

                    if has_shadow:
                        static_drawtext.append(f"shadowx={cfg_shadow_width}")
                        static_drawtext.append(f"shadowy={cfg_shadow_width}")
                        static_drawtext.append(f"shadowcolor={cfg_shadow_color}")

                    if has_box:
                        static_drawtext.append("box=1")
                        static_drawtext.append(f"boxcolor={cfg_back_color}")
                        static_drawtext.append("boxborderw=10")

                    all_filters.append(":".join(static_drawtext))

        if not all_filters:
            import shutil
            shutil.copy(video_path, output_path)
            return output_path

        # Write filtergraph to temporary script file to avoid CLI character limit on Windows
        temp_filter = f"temp_filter_{uuid.uuid4().hex[:8]}.txt"
        with open(temp_filter, "w", encoding="utf-8") as f:
            f.write(",".join(all_filters))

        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-filter_script:v", temp_filter,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "22",
                "-c:a", "aac",
                output_path
            ]
            logger.info(f"burn_subtitles: Running FFmpeg command with filter script: {' '.join(cmd)}")
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                logger.error(f"FFmpeg subtitle burn failed with exit code {result.returncode}")
                logger.error(f"FFmpeg stderr: {result.stderr}")
                raise RuntimeError(f"FFmpeg subtitle burn failed: {result.stderr}")
        finally:
            if os.path.exists(temp_filter):
                try:
                    os.remove(temp_filter)
                except Exception as e:
                    logger.warning(f"Failed to remove temporary filter file {temp_filter}: {e}")

        logger.info(f"Subtitled video saved -> {output_path}")
        return output_path


    def check_safe_zones(
        self,
        edl: Dict[str, Any],
        frame_width: int = 1080,
        frame_height: int = 1920,
    ) -> Dict[str, Any]:
        """
        Validates that text overlays defined in the EDL don't land in
        the platform UI danger zones (top bar, bottom bar, right-side buttons).

        Returns a report dict with:
          - "safe_items"   : clips whose text_overlay is in the safe zone
          - "flagged_items": clips whose text_overlay may be obscured by UI
          - "summary"      : human-readable verdict
        """
        safe_items = []
        flagged_items = []

        for item in edl.get("timeline", []):
            text = item.get("text_overlay", "")
            if not text:
                continue

            # Default assumed text position: centered horizontally,
            # at 80% of frame height (our burn_subtitles default).
            assumed_y = int(frame_height * 0.80)
            
            # Subtitles are capped at 85% of width in burn_subtitles.
            # If centered, the right edge is at: (frame_width/2) + (text_width/2)
            # Max text width = frame_width * 0.75
            max_text_width = frame_width * 0.75
            assumed_x_right = int((frame_width / 2) + (max_text_width / 2))

            flags = []
            if assumed_y < SAFE_ZONE["top_px"]:
                flags.append(f"Too close to top (y={assumed_y}px, danger < {SAFE_ZONE['top_px']}px)")
            if assumed_y > frame_height - SAFE_ZONE["bottom_px"]:
                flags.append(
                    f"Too close to bottom (y={assumed_y}px, danger > {frame_height - SAFE_ZONE['bottom_px']}px)"
                )
            if assumed_x_right > frame_width - SAFE_ZONE["right_px"]:
                flags.append(
                    f"Text may overlap right-side buttons (right edge approx {assumed_x_right}px, danger > {frame_width - SAFE_ZONE['right_px']}px)"
                )

            entry = {
                "text_overlay": text,
                "timeline_start": item.get("timeline_start"),
                "timeline_end":   item.get("timeline_end"),
                "assumed_y_px":   assumed_y,
                "flags":          flags,
            }

            if flags:
                flagged_items.append(entry)
            else:
                safe_items.append(entry)

        total = len(safe_items) + len(flagged_items)
        verdict = "PASS" if not flagged_items else "WARN"
        summary = (
            f"{verdict}: {len(safe_items)}/{total} text overlays are in the safe zone."
            + (f" {len(flagged_items)} may be obscured by platform UI." if flagged_items else "")
        )

        return {
            "verdict": verdict,
            "summary": summary,
            "safe_items": safe_items,
            "flagged_items": flagged_items,
            "safe_zone_rules": SAFE_ZONE,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_captions(self, segments: List[Dict]) -> List[Dict[str, Any]]:
        """
        Converts Whisper segments into a caption list with start/end/text.
        Groups words into short phrases (max 6 words) for readability.
        """
        captions = []
        for seg in segments:
            words = seg.get("words", [])
            if not words:
                text_clean = seg["text"].strip()
                words_list = text_clean.split()
                if words_list:
                    dur = seg["end"] - seg["start"]
                    word_dur = dur / len(words_list)
                    chunk_words = []
                    for idx, w in enumerate(words_list):
                        w_start = seg["start"] + idx * word_dur
                        w_end = w_start + word_dur
                        chunk_words.append({
                            "word": w,
                            "start": round(w_start, 2),
                            "end": round(w_end, 2)
                        })
                    captions.append({
                        "start": round(seg["start"], 2),
                        "end":   round(seg["end"],   2),
                        "text":  text_clean,
                        "words": chunk_words
                    })
                else:
                    captions.append({
                        "start": round(seg["start"], 2),
                        "end":   round(seg["end"],   2),
                        "text":  text_clean,
                        "words": []
                    })
                continue

            # Group into chunks of up to 6 words
            chunk_size = 6
            for i in range(0, len(words), chunk_size):
                chunk = words[i: i + chunk_size]
                text  = " ".join(w["word"].strip() for w in chunk)
                start = round(chunk[0]["start"], 2)
                end   = round(chunk[-1]["end"],  2)
                chunk_words = [
                    {
                        "word": w["word"].strip(),
                        "start": round(w["start"], 2),
                        "end": round(w["end"], 2)
                    }
                    for w in chunk
                ]
                if text:
                    captions.append({
                        "start": start,
                        "end": end,
                        "text": text,
                        "words": chunk_words
                    })

        return captions

    def generate_aesthetic_style(
        self,
        prompt: str,
        storyline: str,
        video_style: str
    ) -> Dict[str, Any]:
        """
        Dynamically analyzes the video's prompt, storyline, and video_style
        to choose a Google Font and custom caption parameters that match the theme.
        """
        import os
        import json
        from groq import Groq
        from backend_ai.core.config_loader import AGENTS_CONFIG
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.warning("GROQ_API_KEY not found in SubtitleAgent; falling back to default style.")
            return {}
            
        client = Groq(api_key=api_key)
        
        system_prompt = (
            "You are a typography and video style expert. Given a video project's title/prompt, storyline, and style, "
            "select the perfect typography settings for both subtitles (burnt-in captions synced with speech) "
            "and text overlays (on-screen title cards, keywords, or section headers chosen by the director).\n"
            "Also decide if subtitles/captions are needed/appropriate to be burned onto this video based on the user's prompt (e.g. if they request captions, subtitles, speech text, or if the content would benefit from it).\n"
            "Choose suitable Google Font families (e.g. 'Poppins', 'Montserrat', 'Lilita One', 'Bebas Neue', 'Cinzel', "
            "'Oswald', 'Roboto', 'Permanent Marker', 'Fredoka', 'Playfair Display') and appropriate options.\n"
            "Ensure the font fits the mood (e.g., 'Lilita One' or 'Bebas Neue' for bold social media reels, "
            "'Cinzel' for dramatic cinematic videos, 'Montserrat' or 'Poppins' for clean modern vlogs).\n"
            "Your response must be a single JSON object (with no markdown wrapping) matching this JSON schema:\n"
            "{\n"
            "  \"requires_subtitles\": \"boolean (true if subtitles/captions should be burnt onto the video based on the user prompt/intent, false otherwise)\",\n"
            "  \"subtitle_style\": {\n"
            "    \"font_name\": \"string (the Google Font name)\",\n"
            "    \"font_size\": \"integer (usually between 32 and 60)\",\n"
            "    \"font_weight\": \"integer (standard Google Font weight: 400, 500, 700, 800, 900)\",\n"
            "    \"font_color\": \"string (color name of active/highlighted text like 'white', 'yellow', 'cyan', 'lime', 'orange', 'red', 'magenta')\",\n"
            "    \"inactive_color\": \"string (color name of inactive/background text like 'white', 'gray', 'lightgray')\",\n"
            "    \"outline_color\": \"string (color name like 'black' or 'none')\",\n"
            "    \"outline_width\": \"integer (between 0 and 5)\",\n"
            "    \"back_color\": \"string (like 'black@0.4' for translucent background, or 'none')\",\n"
            "    \"has_shadow\": \"boolean\",\n"
            "    \"shadow_color\": \"string (like 'black' or 'none')\",\n"
            "    \"shadow_width\": \"integer (between 0 and 4)\",\n"
            "    \"uppercase\": \"boolean\",\n"
            "    \"animate\": \"boolean\"\n"
            "  },\n"
            "  \"text_overlay_style\": {\n"
            "    \"font_name\": \"string (the Google Font name for headers)\",\n"
            "    \"font_size\": \"integer (larger for headers, usually between 60 and 90)\",\n"
            "    \"font_weight\": \"integer (standard Google Font weight: 700, 800, 900)\",\n"
            "    \"font_color\": \"string (color name like 'white', 'yellow', 'cyan', 'lime', 'orange', 'red', 'magenta')\",\n"
            "    \"outline_color\": \"string (color name like 'black' or 'none')\",\n"
            "    \"outline_width\": \"integer (between 0 and 5)\",\n"
            "    \"back_color\": \"string (like 'black@0.4' or 'none')\",\n"
            "    \"has_shadow\": \"boolean\",\n"
            "    \"shadow_color\": \"string (like 'black' or 'none')\",\n"
            "    \"shadow_width\": \"integer (between 0 and 4)\",\n"
            "    \"uppercase\": \"boolean\"\n"
            "  }\n"
            "}"
        )
        
        user_msg = (
            f"Project Title: {prompt}\n"
            f"Storyline: {storyline}\n"
            f"Video Style: {video_style}"
        )
        
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg}
                ],
                model=AGENTS_CONFIG.get("creative_director", {}).get("model", "llama-3.3-70b-versatile"),
                response_format={"type": "json_object"}
            )
            response_text = chat_completion.choices[0].message.content
            style_data = json.loads(response_text)
            logger.info(f"Dynamic subtitle style generated: {style_data}")
            return style_data
        except Exception as e:
            logger.warning(f"Failed to generate dynamic subtitle style via LLM: {e}")
            return {}

    def _find_font(self, font_name: Optional[str] = None, weight: int = 700) -> str:
        """
        Finds a suitable TTF font on the system or downloads it via Google Fonts.
        Falls back to a known Windows system font.
        """
        from backend_ai.utils.font_downloader import get_font_path
        try:
            return get_font_path(font_name, weight)
        except Exception as e:
            logger.warning(f"Could not resolve font '{font_name}' using downloader: {e}. Falling back to default candidates.")
            
        candidates = [
            "C:/Windows/Fonts/Impact.ttf",     # High CTR impact social media font
            "C:/Windows/Fonts/arialbd.ttf",   # Arial Bold
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibri.ttf",
            "C:/Windows/Fonts/verdana.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(
            "No suitable font found. Install Arial or provide a .ttf path."
        )


if __name__ == "__main__":
    logger.info("SubtitleAgent module loaded successfully.")
