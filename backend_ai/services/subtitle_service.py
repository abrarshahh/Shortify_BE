import os
import json
import warnings
from typing import List, Dict, Any, Optional, Tuple

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

    def __init__(self, model_size: str = "base", device: str = "cpu"):
        """
        Args:
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large').
                        'base' is a good balance of speed and accuracy.
            device:     'cpu' or 'cuda'
        """
        self.model_size = model_size
        self.device = device
        self._model = None  # Lazy-load

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self):
        """Lazy-loads the Whisper model on first use."""
        if self._model is None:
            import whisper
            print(f"Loading Whisper model '{self.model_size}' on {self.device}...")
            self._model = whisper.load_model(self.model_size, device=self.device)
            print("Whisper model loaded.")
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
        print(f"Transcribing: {video_path}")

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
    ) -> str:
        """
        Burns caption text onto the video using MoviePy.

        Args:
            video_path:       Path to the source video.
            captions:         Caption list from transcribe().
            output_path:      Where to write the subtitled video.
            font_size:        Subtitle font size in pixels.
            font_color:       Text colour.
            stroke_color:     Outline colour for readability.
            stroke_width:     Outline width in pixels.
            position_y_ratio: Vertical position as a fraction of frame height
                              (0.80 = 80% down = safely above the bottom bar).

        Returns:
            Absolute path to the output video.
        """
        from moviepy import VideoFileClip, TextClip, CompositeVideoClip

        video = VideoFileClip(video_path)
        w, h = video.size
        y_pos = int(h * position_y_ratio)

        text_clips = []
        for cap in captions:
            # Find a font — try common Windows locations
            font = self._find_font()

            txt = TextClip(
                text=cap["text"],
                font=font,
                font_size=font_size,
                color=font_color,
                stroke_color=stroke_color,
                stroke_width=stroke_width,
                method="caption",
                size=(int(w * 0.85), None),   # wrap within 85% of width
            )
            txt = (
                txt
                .with_start(cap["start"])
                .with_end(cap["end"])
                .with_position(("center", y_pos))
            )
            text_clips.append(txt)

        final = CompositeVideoClip([video, *text_clips])
        final.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            logger="bar",
        )
        video.close()
        final.close()
        print(f"Subtitled video saved -> {output_path}")
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
            assumed_x_right = frame_width  # full width, centred

            flags = []
            if assumed_y < SAFE_ZONE["top_px"]:
                flags.append(f"Too close to top (y={assumed_y}px, danger < {SAFE_ZONE['top_px']}px)")
            if assumed_y > frame_height - SAFE_ZONE["bottom_px"]:
                flags.append(
                    f"Too close to bottom (y={assumed_y}px, danger > {frame_height - SAFE_ZONE['bottom_px']}px)"
                )
            if assumed_x_right > frame_width - SAFE_ZONE["right_px"]:
                flags.append(
                    f"Text may overlap right-side buttons (right edge > {frame_width - SAFE_ZONE['right_px']}px)"
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
                # Fall back to segment-level timing
                captions.append({
                    "start": round(seg["start"], 2),
                    "end":   round(seg["end"],   2),
                    "text":  seg["text"].strip()
                })
                continue

            # Group into chunks of up to 6 words
            chunk_size = 6
            for i in range(0, len(words), chunk_size):
                chunk = words[i: i + chunk_size]
                text  = " ".join(w["word"].strip() for w in chunk)
                start = round(chunk[0]["start"], 2)
                end   = round(chunk[-1]["end"],  2)
                if text:
                    captions.append({"start": start, "end": end, "text": text})

        return captions

    def _find_font(self) -> str:
        """
        Finds a suitable TTF font on the system.
        Falls back to a known Windows system font.
        """
        candidates = [
            "C:/Windows/Fonts/arialbd.ttf",   # Arial Bold — clean for subtitles
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
    print("SubtitleAgent module loaded successfully.")
