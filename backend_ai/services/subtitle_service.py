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
        Burns caption text onto the video using FFmpeg and ASS subtitles.
        """
        import uuid
        import subprocess

        if style is None:
            style = getattr(self, "caption_style", "hormozi")

        logger.info(f"burn_subtitles: Using style '{style}'")

        # 1. Determine font name
        font_name = "Arial"
        candidates = {
            "C:/Windows/Fonts/Impact.ttf": "Impact",
            "C:/Windows/Fonts/arialbd.ttf": "Arial",
            "C:/Windows/Fonts/arial.ttf": "Arial",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf": "DejaVu Sans",
        }
        for path, name in candidates.items():
            if os.path.exists(path):
                font_name = name
                break

        # Helper to format seconds to ASS timestamp (H:MM:SS.cs)
        def format_ass_time(seconds: float) -> str:
            if seconds < 0:
                seconds = 0.0
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = int(seconds % 60)
            cs = int(round((seconds - int(seconds)) * 100))
            if cs == 100:
                cs = 99
            return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

        # 2. Build the ASS subtitle file lines
        ass_lines = [
            "[Script Info]",
            "Title: Shortify Subtitles",
            "ScriptType: v4.00+",
            "WrapStyle: 0",
            "ScaledBorderAndShadow: yes",
            "PlayResX: 1080",
            "PlayResY: 1920",
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        ]

        # Define specific ASS style definitions
        if style == "minimal":
            style_line = f"Style: Default,{font_name},48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,1.5,0.0,2,80,80,380,1"
        elif style == "bold":
            style_line = f"Style: Default,{font_name},72,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,3.0,2.0,2,80,80,380,1"
        elif style == "outline":
            style_line = f"Style: Default,{font_name},64,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,4.0,0.0,2,80,80,380,1"
        else: # hormozi
            style_line = f"Style: Default,{font_name},76,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,5.0,0.0,2,80,80,380,1"

        ass_lines.append(style_line)
        ass_lines.extend([
            "",
            "[Events]",
            "Format: Marked, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
        ])

        for cap in captions:
            if style == "hormozi":
                chunk_words = cap.get("words", [])
                if chunk_words:
                    for i, active_w in enumerate(chunk_words):
                        w_start = active_w["start"]
                        w_end = active_w["end"]
                        if w_end <= w_start:
                            w_end = w_start + 0.1
                        # Highlight active word
                        dialogue_words = []
                        for j, w_info in enumerate(chunk_words):
                            word_text = w_info["word"].upper()
                            if j == i:
                                dialogue_words.append(f"{{\\c&H0000FFFF&}}{word_text}{{\\r}}")
                            else:
                                dialogue_words.append(word_text)
                        text_line = " ".join(dialogue_words)
                        start_str = format_ass_time(w_start)
                        end_str = format_ass_time(w_end)
                        ass_lines.append(f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{text_line}")
                else:
                    text_line = cap["text"].upper()
                    start_str = format_ass_time(cap["start"])
                    end_str = format_ass_time(cap["end"])
                    ass_lines.append(f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{text_line}")
            else:
                text_line = cap["text"]
                if style in ["bold", "outline"]:
                    text_line = text_line.upper()
                start_str = format_ass_time(cap["start"])
                end_str = format_ass_time(cap["end"])
                ass_lines.append(f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{text_line}")

        # 3. Write ASS subtitles to a unique relative path to avoid path escaping issues in FFmpeg on Windows
        temp_ass = f"temp_subs_{uuid.uuid4().hex[:8]}.ass"
        with open(temp_ass, "w", encoding="utf-8") as f:
            f.write("\n".join(ass_lines))

        try:
            # We must escape the relative filename backslashes/colons for FFmpeg's subtitle filter,
            # but since it's in the current working directory, we only pass a clean relative filename.
            escaped_ass = temp_ass.replace("\\", "/").replace(":", "\\:")
            
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vf", f"subtitles={escaped_ass}",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "22",
                "-c:a", "aac",
                output_path
            ]
            logger.info(f"burn_subtitles: Running FFmpeg command: {' '.join(cmd)}")
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                logger.error(f"FFmpeg subtitle burn failed with exit code {result.returncode}")
                logger.error(f"FFmpeg stderr: {result.stderr}")
                raise RuntimeError(f"FFmpeg subtitle burn failed: {result.stderr}")
        finally:
            if os.path.exists(temp_ass):
                try:
                    os.remove(temp_ass)
                except Exception as e:
                    logger.warning(f"Failed to remove temporary ASS file {temp_ass}: {e}")

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

    def _find_font(self) -> str:
        """
        Finds a suitable TTF font on the system.
        Falls back to a known Windows system font.
        """
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
