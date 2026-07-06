from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

def _to_float(frame: np.ndarray) -> np.ndarray:
    """uint8 [0,255] -> float32 [0,255]. Working in float avoids uint8
    wraparound (e.g. 250 + 20 -> 14 instead of clipping to 255) at every
    intermediate step; we only cast back to uint8 once, at the very end."""
    return frame.astype(np.float32)


def _to_uint8(frame: np.ndarray) -> np.ndarray:
    """float32 -> uint8, clamping into the valid range first."""
    return np.clip(frame, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Brightness / Contrast / Exposure
# ---------------------------------------------------------------------------

def adjust_brightness(frame: np.ndarray, value: float = 1.0) -> np.ndarray:
    """
    Multiplicative brightness. value=1.0 is neutral.
    value=1.2  -> 20% brighter
    value=0.8  -> 20% darker

    Multiplicative (rather than additive offset) is the standard choice
    here because it scales proportionally to existing pixel value, so
    highlights and shadows don't blow out/clip at the same flat rate --
    closer to how a camera ISO/exposure change behaves than a flat "+20".
    """
    if value == 1.0:
        return frame
    f = _to_float(frame) * value
    return _to_uint8(f)


def adjust_contrast(frame: np.ndarray, value: float = 1.0) -> np.ndarray:
    """
    Contrast around the mid-gray pivot (128). value=1.0 is neutral.
    value > 1.0 -> more contrast (darks darker, lights lighter)
    value < 1.0 -> flatter / less contrast
    """
    if value == 1.0:
        return frame
    f = _to_float(frame)
    f = (f - 128.0) * value + 128.0
    return _to_uint8(f)


def adjust_gamma(frame: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Gamma / exposure correction. gamma=1.0 is neutral.
    gamma > 1.0 -> brightens midtones (lifts shadows) without blowing out
                   highlights the way adjust_brightness would
    gamma < 1.0 -> darkens midtones, protects highlights

    This is the right tool for "footage is dark but I don't want to wash
    out the sky" -- which a flat brightness multiply can't do, since
    brightness scales every tone equally.
    """
    if gamma == 1.0:
        return frame
    # Build once per call; frame sizes vary so this isn't worth a global
    # cache, but it IS worth doing as a lookup table rather than a
    # per-pixel `** ` call, which is meaningfully slower at 1080x1920.
    inv_gamma = 1.0 / max(gamma, 1e-6)
    table = (np.linspace(0, 1, 256) ** inv_gamma * 255.0).astype(np.uint8)
    return cv2.LUT(frame, table)


# ---------------------------------------------------------------------------
# Saturation / Vibrance / Hue / Temperature  (HSV-space operations)
# ---------------------------------------------------------------------------

def adjust_saturation(frame: np.ndarray, value: float = 1.0) -> np.ndarray:
    """
    Saturation scale. value=1.0 is neutral.
    value=0.0 -> full grayscale
    value=2.0 -> strongly saturated

    Flat scale on the S channel -- affects already-vivid colors and
    near-gray colors by the same proportion. See adjust_vibrance() for
    the perceptually softer version that protects skin tones.
    """
    if value == 1.0:
        return frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * value, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def adjust_vibrance(frame: np.ndarray, value: float = 1.0) -> np.ndarray:
    """
    Saturation boost weighted by *current* saturation: pixels that are
    already vivid get boosted less than pixels that are nearly gray.
    This is the standard "vibrance" behavior found in most editors --
    it avoids pushing skin tones (already mid-saturation) as hard as
    flat saturation would, which is what makes flat saturation pushes
    look unnatural on faces.

    value=1.0 is neutral. value=1.5 is a noticeable vibrance lift.
    """
    if value == 1.0:
        return frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
    s = hsv[..., 1]
    # weight: pixels with low current saturation (s near 0) get close to
    # the full `value` boost; already-saturated pixels (s near 255) get
    # close to no extra boost. This is the "protection" curve.
    weight = 1.0 - (s / 255.0)
    boost = 1.0 + (value - 1.0) * weight
    hsv[..., 1] = np.clip(s * boost, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def adjust_hue(frame: np.ndarray, degrees: float = 0.0) -> np.ndarray:
    """
    Rotates hue around the color wheel. degrees=0 is neutral.
    OpenCV's H channel is stored on a 0-179 scale (not 0-360), so we
    convert the input degrees proportionally.
    """
    if degrees == 0.0:
        return frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.int16)
    shift = int(round(degrees / 360.0 * 180.0))
    hsv[..., 0] = (hsv[..., 0] + shift) % 180
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def adjust_temperature(frame: np.ndarray, value: float = 0.0) -> np.ndarray:
    """
    White balance / temperature shift. value=0.0 is neutral, range
    roughly -50..50.
    value > 0 -> warmer (boost red, pull down blue)
    value < 0 -> cooler (boost blue, pull down red)

    This is a simple, fast per-channel offset rather than a true
    Kelvin-based white-balance model -- it's the same approximation
    CapCut/Premiere's quick "Temp" slider uses, not full color science,
    which is the right tradeoff for a fast per-frame transform.
    """
    if value == 0.0:
        return frame
    f = _to_float(frame)
    f[..., 0] = f[..., 0] + value        # R channel
    f[..., 2] = f[..., 2] - value        # B channel
    return _to_uint8(f)


# ---------------------------------------------------------------------------
# Vignette
# ---------------------------------------------------------------------------

def apply_vignette(
    frame: np.ndarray,
    strength: float = 0.0,
    radius: float = 0.75,
) -> np.ndarray:
    """
    Darkens the frame edges with a radial falloff.
    strength=0.0 -> neutral (no vignette)
    strength=1.0 -> strong, edges go near-black
    radius       -> how far from center the falloff starts (0-1, fraction
                    of the half-diagonal). Smaller radius = falloff starts
                    closer to center = more aggressive-looking vignette.

    Built once per frame size via a normalized radial distance grid.
    For a fixed-resolution render pipeline (you render at a constant
    1080x1920) this grid is identical on every call, so if vignette ends
    up being used heavily, it's a good candidate to cache by frame shape
    rather than rebuild every frame -- noted here, not prematurely
    optimized into this function.
    """
    if strength <= 0.0:
        return frame

    h, w = frame.shape[:2]
    y, x = np.ogrid[:h, :w]
    cx, cy = w / 2.0, h / 2.0
    # distance from center, normalized so the corner = 1.0
    max_dist = np.sqrt(cx ** 2 + cy ** 2)
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / max_dist

    # mask is 1.0 inside `radius`, falls off smoothly to (1 - strength)
    # at the corners
    falloff = np.clip((dist - radius) / max(1e-6, (1.0 - radius)), 0, 1)
    mask = 1.0 - falloff * strength
    mask = mask[..., np.newaxis]  # broadcast over RGB channels

    f = _to_float(frame) * mask
    return _to_uint8(f)


# ---------------------------------------------------------------------------
# Combined params + single-pass combinator
# ---------------------------------------------------------------------------

@dataclass
class ColorGradeParams:
    """
    Mirrors the ColorGradeParams Pydantic model that belongs in the EDL
    schema (backend_ai/core/edl_validator.py). Keep the field names and
    neutral defaults identical in both places -- the Pydantic side is
    responsible for clamping (Field(ge=..., le=...)) before this dataclass
    ever gets constructed from director output; this side assumes
    values already arrived validated, and applies them.
    """
    brightness: float = 1.0
    contrast: float = 1.0
    gamma: float = 1.0
    saturation: float = 1.0
    vibrance: float = 1.0
    hue: float = 0.0
    temperature: float = 0.0
    vignette_strength: float = 0.0
    vignette_radius: float = 0.75

    def is_neutral(self) -> bool:
        """True if every field is at its default -- lets callers skip the
        whole pipeline (and the per-frame closure) entirely for clips the
        director didn't grade."""
        return self == ColorGradeParams()


def _grade_frame(frame: np.ndarray, params: ColorGradeParams) -> np.ndarray:
    """
    Applies every non-neutral adjustment in ONE pass per frame, in a
    fixed, documented order. Order matters for color math (e.g. you want
    exposure/contrast settled before saturation reads the values), so
    this order is intentional, not incidental:

        1. gamma        (exposure / midtone correction)
        2. brightness   (overall level)
        3. contrast     (tonal range)
        4. temperature  (white balance)
        5. hue          (color wheel rotation)
        6. saturation   (overall color intensity)
        7. vibrance     (perceptual / skin-protected saturation)
        8. vignette     (applied last -- it's a spatial mask, not a
                         tonal operation, so it should sit on top of
                         whatever color grade preceded it)
    """
    f = frame
    if params.gamma != 1.0:
        f = adjust_gamma(f, params.gamma)
    if params.brightness != 1.0:
        f = adjust_brightness(f, params.brightness)
    if params.contrast != 1.0:
        f = adjust_contrast(f, params.contrast)
    if params.temperature != 0.0:
        f = adjust_temperature(f, params.temperature)
    if params.hue != 0.0:
        f = adjust_hue(f, params.hue)
    if params.saturation != 1.0:
        f = adjust_saturation(f, params.saturation)
    if params.vibrance != 1.0:
        f = adjust_vibrance(f, params.vibrance)
    if params.vignette_strength > 0.0:
        f = apply_vignette(f, params.vignette_strength, params.vignette_radius)
    return f


def apply_color_grade(clip, params: ColorGradeParams):
    """
    The one function in this module that touches MoviePy.

    Wraps _grade_frame as a single `image_transform`, so a clip with
    several non-default params still only takes ONE extra pass over its
    pixels per frame at render time, instead of stacking N separate
    MoviePy effect wrappers (each of which would re-walk every pixel).

    Usage from editor_service.py's per-clip loop:

        if item.get("color_grade"):
            clip = apply_color_grade(clip, ColorGradeParams(**item["color_grade"]))

    Returns the clip unchanged if params is neutral -- cheap to call
    unconditionally for every clip without an `if` guard at the call site.
    """
    if params.is_neutral():
        return clip
    return clip.image_transform(lambda frame: _grade_frame(frame, params))


# ---------------------------------------------------------------------------
# Optional: histogram-based auto-suggestion (no LLM call required)
# ---------------------------------------------------------------------------

def suggest_auto_grade(frame: np.ndarray) -> ColorGradeParams:
    """
    Computes a baseline correction directly from a sampled frame's own
    histogram -- no AI call needed. Intended as either:
      (a) a fallback when the director didn't specify color_grade, or
      (b) a pre-pass whose numbers get fed INTO the director's prompt
          context alongside Gemini's text "lighting" description, so
          Groq is reasoning from a measured value instead of guessing
          blind from a word like "dim".

    Deliberately conservative: nudges exposure/contrast toward a
    reasonable target, doesn't touch saturation/hue/temperature, since
    those are stylistic choices this function has no basis to guess.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mean_luma = float(gray.mean())  # 0-255

    target_luma = 128.0
    params = ColorGradeParams()

    # Only correct if it's meaningfully off-target; avoids nudging
    # already-fine footage for no reason.
    if abs(mean_luma - target_luma) > 15:
        # Gamma correction toward target, gently clamped so a single
        # very dark/bright frame sample can't produce an extreme swing.
        ratio = target_luma / max(mean_luma, 1.0)
        gamma = float(np.clip(ratio, 0.7, 1.5))
        params.gamma = gamma

    std_luma = float(gray.std())
    if std_luma < 40:  # flat/low-contrast footage
        params.contrast = 1.15

def build_ffmpeg_color_filter(params: ColorGradeParams) -> str:
    """
    Translates ColorGradeParams to an optimized FFmpeg filter chain string.
    Chains eq, colorbalance, hue, and vignette filters if they are non-default.
    """
    if params.is_neutral():
        return ""

    filters = []

    # 1. eq filter for contrast, brightness, saturation, gamma
    eq_parts = []
    if params.contrast != 1.0:
        eq_parts.append(f"contrast={params.contrast:.4f}")
    if params.brightness != 1.0:
        # map multiplicative brightness default 1.0 to additive offset default 0.0 in FFmpeg eq
        brightness_val = params.brightness - 1.0
        eq_parts.append(f"brightness={brightness_val:.4f}")

    # Combine saturation and vibrance into a single saturation factor approximation
    sat_factor = params.saturation
    if params.vibrance != 1.0:
        # Approximate vibrance by scaling the saturation boost
        sat_factor *= (1.0 + (params.vibrance - 1.0) * 0.5)

    if sat_factor != 1.0:
        eq_parts.append(f"saturation={sat_factor:.4f}")
    if params.gamma != 1.0:
        eq_parts.append(f"gamma={params.gamma:.4f}")

    if eq_parts:
        filters.append("eq=" + ":".join(eq_parts))

    # 2. colorbalance filter for temperature
    if params.temperature != 0.0:
        # Map -50..50 to -0.3..0.3 to keep balance subtle
        val = params.temperature * (0.3 / 50.0)
        filters.append(f"colorbalance=rm={val:.4f}:bm={-val:.4f}:rh={val:.4f}:bh={-val:.4f}")

    # 3. hue filter
    if params.hue != 0.0:
        filters.append(f"hue=h={params.hue:.4f}")

    # 4. vignette filter
    if params.vignette_strength > 0.0:
        import math
        angle = params.vignette_strength * (math.pi / 4.0)
        filters.append(f"vignette=angle={angle:.4f}")

    return ",".join(filters)


if __name__ == "__main__":
    # Smoke test: every function should run without error on a synthetic
    # frame and the neutral case should be a true no-op (identical array).
    test_frame = np.random.randint(0, 256, (1920, 1080, 3), dtype=np.uint8)

    assert np.array_equal(adjust_brightness(test_frame, 1.0), test_frame)
    assert np.array_equal(adjust_contrast(test_frame, 1.0), test_frame)
    assert np.array_equal(adjust_gamma(test_frame, 1.0), test_frame)
    assert np.array_equal(adjust_saturation(test_frame, 1.0), test_frame)
    assert np.array_equal(adjust_vibrance(test_frame, 1.0), test_frame)
    assert np.array_equal(adjust_hue(test_frame, 0.0), test_frame)
    assert np.array_equal(adjust_temperature(test_frame, 0.0), test_frame)
    assert np.array_equal(apply_vignette(test_frame, 0.0), test_frame)

    graded = _grade_frame(test_frame, ColorGradeParams(
        brightness=1.2, contrast=1.1, saturation=0.9,
        temperature=10, vignette_strength=0.3,
    ))
    assert graded.shape == test_frame.shape
    assert graded.dtype == np.uint8

    suggestion = suggest_auto_grade(test_frame)
    print("Auto-suggested grade for random noise frame:", suggestion)
    print("All color.py smoke tests passed.")