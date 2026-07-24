import os
import logging
import subprocess
from typing import Dict, Any, Optional

from backend_ai.core.config import COLOR_GRADING_ENABLED, FFMPEG_PATH

logger = logging.getLogger("agents.color_grading")


# ---------------------------------------------------------------------------
# Color grade presets — one per style.
# Each preset is a dict of scalar adjustments that map directly to FFmpeg
# filter parameters. All values are relative to a neutral baseline of 1.0
# unless documented otherwise.
#
# Parameters:
#   contrast    : eq filter contrast  (1.0 = no change, >1 = more contrast)
#   brightness  : eq filter brightness (-1.0 to 1.0, 0.0 = no change)
#   saturation  : eq filter saturation (1.0 = no change, 0.0 = grayscale)
#   gamma       : eq filter gamma      (1.0 = no change, <1 = brighter mids)
#   gamma_r     : red channel gamma    (warm/cool toning)
#   gamma_b     : blue channel gamma   (warm/cool toning)
#   vignette    : vignette strength    (0.0 = off, 1.0 = strong)
# ---------------------------------------------------------------------------

STYLE_PRESETS: Dict[str, Dict[str, float]] = {
    "cinematic": {
        "contrast":   1.15,
        "brightness": -0.03,
        "saturation": 0.82,
        "gamma":      0.95,
        "gamma_r":    0.97,   # slight cool shadow
        "gamma_b":    1.03,
        "vignette":   0.6,
    },
    "fast_cut": {
        "contrast":   1.25,
        "brightness": 0.02,
        "saturation": 1.12,
        "gamma":      1.0,
        "gamma_r":    1.0,
        "gamma_b":    1.0,
        "vignette":   0.3,
    },
    "travel": {
        "contrast":   1.05,
        "brightness": 0.04,
        "saturation": 1.22,
        "gamma":      1.02,
        "gamma_r":    1.03,   # slight warm push
        "gamma_b":    0.97,
        "vignette":   0.2,
    },
    "dramatic": {
        "contrast":   1.32,
        "brightness": -0.06,
        "saturation": 0.68,
        "gamma":      0.90,
        "gamma_r":    0.95,
        "gamma_b":    1.05,
        "vignette":   0.8,
    },
    "birthday": {
        "contrast":   1.00,
        "brightness": 0.06,
        "saturation": 1.32,
        "gamma":      1.05,
        "gamma_r":    1.05,
        "gamma_b":    0.96,
        "vignette":   0.1,
    },
    "adventure": {
        "contrast":   1.18,
        "brightness": 0.01,
        "saturation": 1.10,
        "gamma":      0.98,
        "gamma_r":    1.02,
        "gamma_b":    0.98,
        "vignette":   0.4,
    },
    "romantic": {
        "contrast":   0.95,
        "brightness": 0.05,
        "saturation": 0.90,
        "gamma":      1.03,
        "gamma_r":    1.06,   # warm, soft highlights
        "gamma_b":    0.95,
        "vignette":   0.5,
    },
    "funny": {
        "contrast":   1.08,
        "brightness": 0.05,
        "saturation": 1.18,
        "gamma":      1.02,
        "gamma_r":    1.01,
        "gamma_b":    1.01,
        "vignette":   0.0,
    },
    # Fallback for unknown or 'general' style — neutral pass-through
    "general": {
        "contrast":   1.0,
        "brightness": 0.0,
        "saturation": 1.0,
        "gamma":      1.0,
        "gamma_r":    1.0,
        "gamma_b":    1.0,
        "vignette":   0.0,
    },
}


class ColorGradingAgent:
    """
    Phase 3: Applies style-matched color grading to the rendered video
    using FFmpeg's eq and vignette filters via a subprocess call.

    This runs as a post-process step after VideoEditor produces the raw
    assembled video and before SubtitleAgent burns captions. It writes
    a graded intermediate file to the same exports directory.

    Design decisions:
    - Uses FFmpeg directly (via subprocess) rather than ffmpeg-python's
      fluent API to keep the filter chain string explicit and debuggable.
    - The vignette is applied as a separate filter in the chain so it
      can be disabled cleanly by setting strength to 0.0.
    - No ML models, no API calls — runs locally in seconds.
    """

    def __init__(self):
        self.enabled = COLOR_GRADING_ENABLED
        self.ffmpeg_path = FFMPEG_PATH
        
        # Ensure LUTs directory exists
        self.luts_dir = os.path.abspath(os.path.join("data", "luts"))
        os.makedirs(self.luts_dir, exist_ok=True)
        
        self._generate_default_luts()

    def _generate_default_luts(self):
        # We define a helper to generate a LUT file of size 17 if it doesn't exist
        # Or if it exists but is of size 2 (outdated) or lacks the loop fix, we overwrite it.
        def should_regenerate(path):
            if not os.path.exists(path):
                return True
            try:
                with open(path, "r") as f:
                    content = f.read(500)
                    if "LUT_3D_SIZE 2" in content or "# Corrected Loop Order" not in content:
                        return True
            except Exception:
                pass
            return False

        size = 17
        
        # 1. Cinematic
        cinematic_path = os.path.join(self.luts_dir, "cinematic.cube")
        if should_regenerate(cinematic_path):
            logger.info("Generating high-quality Cinematic Teal-Orange LUT...")
            lines = [
                '# Corrected Loop Order',
                'TITLE "Cinematic Teal Orange"',
                f'LUT_3D_SIZE {size}',
                'DOMAIN_MIN 0.0 0.0 0.0',
                'DOMAIN_MAX 1.0 1.0 1.0'
            ]
            def s_curve(x):
                return 3 * (x**2) - 2 * (x**3)
            for b_idx in range(size):
                b = b_idx / (size - 1)
                for g_idx in range(size):
                    g = g_idx / (size - 1)
                    for r_idx in range(size):
                        r = r_idx / (size - 1)
                        r_c, g_c, b_c = s_curve(r), s_curve(g), s_curve(b)
                        luma = 0.299 * r_c + 0.587 * g_c + 0.114 * b_c
                        if luma > 0.5:
                            w = (luma - 0.5) * 2.0
                            ro = r_c + 0.08 * w
                            go = g_c + 0.03 * w
                            bo = b_c - 0.05 * w
                        else:
                            w = (0.5 - luma) * 2.0
                            ro = r_c - 0.05 * w
                            go = g_c + 0.01 * w
                            bo = b_c + 0.06 * w
                        ro = max(0.0, min(1.0, ro))
                        go = max(0.0, min(1.0, go))
                        bo = max(0.0, min(1.0, bo))
                        lines.append(f"{ro:.6f} {go:.6f} {bo:.6f}")
            with open(cinematic_path, "w") as f:
                f.write("\n".join(lines) + "\n")

        # 2. Vintage
        vintage_path = os.path.join(self.luts_dir, "vintage.cube")
        if should_regenerate(vintage_path):
            logger.info("Generating high-quality Vintage Retro Warm LUT...")
            lines = [
                '# Corrected Loop Order',
                'TITLE "Vintage Faded Warm"',
                f'LUT_3D_SIZE {size}',
                'DOMAIN_MIN 0.0 0.0 0.0',
                'DOMAIN_MAX 1.0 1.0 1.0'
            ]
            for b_idx in range(size):
                b = b_idx / (size - 1)
                for g_idx in range(size):
                    g = g_idx / (size - 1)
                    for r_idx in range(size):
                        r = r_idx / (size - 1)
                        # Lift blacks, compress whites
                        ro = 0.07 + 0.88 * r
                        go = 0.07 + 0.88 * g
                        bo = 0.07 + 0.88 * b
                        luma = 0.299 * ro + 0.587 * go + 0.114 * bo
                        # Desaturate slightly
                        ro = ro * 0.75 + luma * 0.25
                        go = go * 0.75 + luma * 0.25
                        bo = bo * 0.75 + luma * 0.25
                        # Warm toning push
                        ro += 0.04
                        go += 0.01
                        bo -= 0.02
                        ro = max(0.0, min(1.0, ro))
                        go = max(0.0, min(1.0, go))
                        bo = max(0.0, min(1.0, bo))
                        lines.append(f"{ro:.6f} {go:.6f} {bo:.6f}")
            with open(vintage_path, "w") as f:
                f.write("\n".join(lines) + "\n")

        # 3. Monochrome
        monochrome_path = os.path.join(self.luts_dir, "monochrome.cube")
        if should_regenerate(monochrome_path):
            logger.info("Generating high-quality Monochrome Noir LUT...")
            lines = [
                '# Corrected Loop Order',
                'TITLE "Monochrome Noir"',
                f'LUT_3D_SIZE {size}',
                'DOMAIN_MIN 0.0 0.0 0.0',
                'DOMAIN_MAX 1.0 1.0 1.0'
            ]
            for b_idx in range(size):
                b = b_idx / (size - 1)
                for g_idx in range(size):
                    g = g_idx / (size - 1)
                    for r_idx in range(size):
                        r = r_idx / (size - 1)
                        luma = 0.299 * r + 0.587 * g + 0.114 * b
                        val = 3 * (luma**2) - 2 * (luma**3)
                        ro = val
                        go = val
                        bo = val * 1.02
                        ro = max(0.0, min(1.0, ro))
                        go = max(0.0, min(1.0, go))
                        bo = max(0.0, min(1.0, bo))
                        lines.append(f"{ro:.6f} {go:.6f} {bo:.6f}")
            with open(monochrome_path, "w") as f:
                f.write("\n".join(lines) + "\n")

        # 4. Nordic
        nordic_path = os.path.join(self.luts_dir, "nordic.cube")
        if should_regenerate(nordic_path):
            logger.info("Generating high-quality Nordic Cool LUT...")
            lines = [
                '# Corrected Loop Order',
                'TITLE "Nordic Cool"',
                f'LUT_3D_SIZE {size}',
                'DOMAIN_MIN 0.0 0.0 0.0',
                'DOMAIN_MAX 1.0 1.0 1.0'
            ]
            for b_idx in range(size):
                b = b_idx / (size - 1)
                for g_idx in range(size):
                    g = g_idx / (size - 1)
                    for r_idx in range(size):
                        r = r_idx / (size - 1)
                        luma = 0.299 * r + 0.587 * g + 0.114 * b
                        ro = r * 0.6 + luma * 0.4
                        go = g * 0.6 + luma * 0.4
                        bo = b * 0.6 + luma * 0.4
                        if luma < 0.5:
                            w = (0.5 - luma) * 2.0
                            ro -= 0.03 * w
                            go -= 0.01 * w
                            bo += 0.05 * w
                        else:
                            w = (luma - 0.5) * 2.0
                            ro -= 0.01 * w
                            bo += 0.02 * w
                        ro = max(0.0, min(1.0, ro))
                        go = max(0.0, min(1.0, go))
                        bo = max(0.0, min(1.0, bo))
                        lines.append(f"{ro:.6f} {go:.6f} {bo:.6f}")
            with open(nordic_path, "w") as f:
                f.write("\n".join(lines) + "\n")

    def _get_preset(self, style: str) -> Dict[str, float]:
        """
        Returns the color grade preset for the given style.
        Falls back to 'general' (neutral) if the style is unknown.
        """
        return STYLE_PRESETS.get(style.lower(), STYLE_PRESETS["general"])

    def _build_filter_chain(self, preset: Dict[str, float]) -> str:
        """
        Constructs the FFmpeg -vf filter chain string from a preset dict.

        The chain is:
          eq (contrast, brightness, saturation, gamma, gamma_r, gamma_b)
          -> vignette (optional, only appended if strength > 0)

        Returns a string suitable for passing to ffmpeg -vf.
        """
        eq_filter = (
            f"eq="
            f"contrast={preset['contrast']:.4f}:"
            f"brightness={preset['brightness']:.4f}:"
            f"saturation={preset['saturation']:.4f}:"
            f"gamma={preset['gamma']:.4f}:"
            f"gamma_r={preset['gamma_r']:.4f}:"
            f"gamma_b={preset['gamma_b']:.4f}"
        )

        filters = [eq_filter]

        if preset.get("vignette", 0.0) > 0.0:
            # FFmpeg vignette: angle parameter controls the strength.
            # PI/4 at strength 1.0, scaled down proportionally.
            import math
            angle = math.pi / 4 * preset["vignette"]
            filters.append(f"vignette=angle={angle:.4f}")

        return ",".join(filters)

    def apply_grade(
        self,
        video_path: str,
        style: str,
        output_dir: str,
        output_suffix: str = "_graded"
    ) -> str:
        """
        Applies color grading to the input video and writes a new file.

        Args:
            video_path:    Absolute path to the rendered video (input).
            style:         Style name matching a key in STYLE_PRESETS.
            output_dir:    Directory to write the graded output file.
            output_suffix: Suffix appended to the filename before extension.

        Returns:
            Absolute path to the graded output video.

        Raises:
            FileNotFoundError: If video_path does not exist.
            RuntimeError:      If FFmpeg exits with a non-zero return code.
        """
        if not self.enabled:
            logger.info("ColorGradingAgent: grading disabled in config, skipping")
            return video_path

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Input video not found: {video_path}")

        # Check for LUT file input
        # Case A: Style is a direct path to a .cube file
        # Case B: Style corresponds to a .cube file in the data/luts directory
        lut_path = None
        if style.lower().endswith(".cube") and os.path.exists(style):
            lut_path = os.path.abspath(style)
        else:
            lut_candidate = os.path.join(self.luts_dir, f"{style}.cube")
            if os.path.exists(lut_candidate):
                lut_path = lut_candidate

        if lut_path:
            # Escape path for FFmpeg filter on Windows
            lut_path_escaped = lut_path.replace("\\", "/").replace(":", "\\:")
            filter_chain = f"lut3d='{lut_path_escaped}'"
            logger.info(f"Detected custom LUT file -> {lut_path}")
        else:
            preset = self._get_preset(style)
            filter_chain = self._build_filter_chain(preset)

        # Build output path
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        ext = os.path.splitext(video_path)[1]
        output_filename = f"{base_name}{output_suffix}{ext}"
        output_path = os.path.join(output_dir, output_filename)

        logger.info(f"Applying '{style}' color grade")
        logger.info(f"  Input:   {video_path}")
        logger.info(f"  Output:  {output_path}")
        logger.info(f"  Filters: {filter_chain}")

        cmd = [
            self.ffmpeg_path,
            "-y",                        # overwrite output without asking
            "-nostdin",
            "-i", video_path,            # input
            "-vf", filter_chain,         # video filter chain
            "-c:v", "libx264",           # re-encode video with H.264
            "-preset", "fast",           # fast preset — quality vs speed tradeoff
            "-crf", "18",                # high quality (lower = better, 18 is near-lossless)
            "-c:a", "copy",              # copy audio stream unchanged (no re-encode)
            output_path
        ]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                stdin=subprocess.DEVNULL
            )
        except FileNotFoundError:
            raise RuntimeError(
                f"FFmpeg not found at '{self.ffmpeg_path}'. "
                "Ensure FFmpeg is installed and available on PATH."
            )

        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg color grading failed (exit code {result.returncode}).\n"
                f"Command: {' '.join(cmd)}\n"
                f"stderr:\n{result.stderr[-2000:]}"  # last 2000 chars to avoid huge logs
            )

        logger.info(f"Color grading complete -> {output_path}")
        return output_path

    def get_preset_info(self, style: str) -> Dict[str, Any]:
        """
        Returns the preset parameters for a given style.
        Useful for logging and debugging without running FFmpeg.
        """
        preset = self._get_preset(style)
        return {
            "style": style,
            "resolved_preset": style.lower() if style.lower() in STYLE_PRESETS else "general",
            "parameters": preset,
            "filter_chain": self._build_filter_chain(preset)
        }


if __name__ == "__main__":
    # Quick smoke test — prints preset info for all styles without running FFmpeg
    agent = ColorGradingAgent()
    for style_name in STYLE_PRESETS:
        info = agent.get_preset_info(style_name)
        logger.info(f"[{info['style']}] Filter: {info['filter_chain']}")
