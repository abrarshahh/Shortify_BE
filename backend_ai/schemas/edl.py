import json
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class TransitionType(str, Enum):
    none = "none"
    jump_cut = "jump_cut"
    fade = "fade"
    crossfade = "crossfade"
    dip_to_black = "dip_to_black"
    fade_to_white = "fade_to_white"
    slide_left = "slide_left"
    slide_right = "slide_right"
    slide_up = "slide_up"
    slide_down = "slide_down"
    slide_push = "slide_push"
    wipe_left = "wipe_left"
    wipe_right = "wipe_right"
    wipe_up = "wipe_up"
    wipe_down = "wipe_down"
    wipe_diagonal_tl = "wipe_diagonal_tl"
    wipe_diagonal_tr = "wipe_diagonal_tr"
    wipe_diagonal_bl = "wipe_diagonal_bl"
    wipe_diagonal_br = "wipe_diagonal_br"
    split_horizontal = "split_horizontal"
    split_vertical = "split_vertical"
    iris = "iris"
    iris_circle = "iris_circle"
    diamond = "diamond"
    heart = "heart"
    blinds_horizontal = "blinds_horizontal"
    blinds_vertical = "blinds_vertical"
    checkerboard = "checkerboard"
    clock_wipe = "clock_wipe"
    zoom_in = "zoom_in"
    zoom_out = "zoom_out"
    glitch = "glitch"
    pixelate = "pixelate"
    spin = "spin"
    ripple = "ripple"
    blur = "blur"
    light_leak = "light_leak"


class PacingStyle(str, Enum):
    speed_ramp = "speed-ramp"
    jump_cut = "jump-cut"
    cinematic_slow = "cinematic-slow"


class EDLClipDetails(BaseModel):
    model_config = ConfigDict(extra="forbid")

    visual_cue: str
    sound_design: str
    pacing_style: PacingStyle
    is_hook: bool = False
    keep_original_audio: bool = True
    effect_type: Optional[str] = "none"
    effect_query: Optional[str] = ""
    sticker_query: Optional[str] = ""
    sticker_position: Optional[str] = "bottom-center"


class ColorGradeParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    brightness: float = Field(1.0, ge=0.5, le=1.8)
    contrast: float = Field(1.0, ge=0.5, le=2.0)
    gamma: float = Field(1.0, ge=0.1, le=10.0)
    saturation: float = Field(1.0, ge=0.0, le=2.0)
    vibrance: float = Field(1.0, ge=0.0, le=2.0)
    hue: float = Field(0.0, ge=-180.0, le=180.0)
    temperature: float = Field(0.0, ge=-50.0, le=50.0)
    vignette_strength: float = Field(0.0, ge=0.0, le=1.0)
    vignette_radius: float = Field(0.75, ge=0.0, le=2.0)


class AudioDuckingParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    original_audio_volume: float = Field(1.0, ge=0.0, le=1.0)
    music_volume_during_segment: float = Field(0.22, ge=0.0, le=1.0)


class EDLTimelineItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clip_name: str
    start_in_clip: float = Field(ge=0)
    end_in_clip: float = Field(ge=0)
    timeline_start: float = Field(ge=0)
    timeline_end: float = Field(ge=0)
    transition: TransitionType
    transition_params: Optional[Dict[str, Any]] = None
    color_grade: Optional[ColorGradeParams] = None
    audio_ducking: Optional[AudioDuckingParams] = None
    speed_preset: Optional[str] = None
    speed_keyframes: Optional[List[Tuple[float, float]]] = None
    reverse: Optional[bool] = False
    stabilize: Optional[bool] = False
    stabilize_strength: Optional[float] = Field(0.5, ge=0.0, le=1.0)
    text_overlay: str = ""
    text_preset: Optional[str] = None
    text_animation: Optional[str] = None
    sticker_animation: Optional[str] = None
    sticker_path: Optional[str] = None
    sticker_position: Optional[str] = None
    effect_path: Optional[str] = None
    details: EDLClipDetails

    @model_validator(mode="after")
    def validate_ranges(self):
        if self.end_in_clip <= self.start_in_clip:
            raise ValueError("start_in_clip must be less than end_in_clip")
        if self.timeline_end <= self.timeline_start:
            raise ValueError("timeline_end must be greater than timeline_start")
            
        if self.speed_preset and self.speed_keyframes:
            raise ValueError("speed_preset and speed_keyframes are mutually exclusive; you cannot specify both.")
        
        if self.speed_preset:
            valid_presets = {"constant_fast", "constant_slow", "ramp_up", "ramp_down", "speed_bump", "freeze_frame"}
            if self.speed_preset not in valid_presets:
                raise ValueError(f"Invalid speed_preset '{self.speed_preset}'. Must be one of: {', '.join(valid_presets)}")
                
        if self.speed_keyframes:
            if len(self.speed_keyframes) < 2:
                raise ValueError("speed_keyframes must contain at least 2 points.")
            
            prev_x = -1.0
            for i, point in enumerate(self.speed_keyframes):
                if len(point) != 2:
                    raise ValueError(f"Keyframe at index {i} must be a Tuple of (time_fraction, speed_multiplier).")
                x, s = point
                if not (0.0 <= x <= 1.0):
                    raise ValueError(f"Keyframe time fraction at index {i} must be in [0.0, 1.0]. Got {x}")
                if s <= 0.0:
                    raise ValueError(f"Keyframe speed multiplier at index {i} must be positive. Got {s}")
                if x < prev_x:
                    raise ValueError("speed_keyframes must be sorted in ascending order of time fraction.")
                prev_x = x
            
            if self.speed_keyframes[0][0] != 0.0 or self.speed_keyframes[-1][0] != 1.0:
                raise ValueError("speed_keyframes must start at time fraction 0.0 and end at 1.0.")
                
        if self.text_preset:
            valid_text_presets = {"bold_hype", "classic_clean", "neon_glow", "minimal_pop"}
            if self.text_preset not in valid_text_presets:
                raise ValueError(f"Invalid text_preset '{self.text_preset}'. Must be one of: {', '.join(valid_text_presets)}")
                
        if self.text_animation:
            valid_anims = {"none", "fade", "slide_up", "slide_down", "slide_left", "slide_right"}
            if self.text_animation not in valid_anims:
                raise ValueError(f"Invalid text_animation '{self.text_animation}'. Must be one of: {', '.join(valid_anims)}")
                
        if self.sticker_animation:
            valid_anims = {"none", "fade", "slide_up", "slide_down", "slide_left", "slide_right"}
            if self.sticker_animation not in valid_anims:
                raise ValueError(f"Invalid sticker_animation '{self.sticker_animation}'. Must be one of: {', '.join(valid_anims)}")
                
        return self


class EDLDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    storyline: str
    total_duration: float = Field(gt=0)
    music_start_offset: float = Field(ge=0)
    timeline: List[EDLTimelineItem]

    @model_validator(mode="after")
    def validate_total_duration(self):
        if not self.timeline:
            raise ValueError("timeline must contain at least one clip")

        timeline_sum = sum(item.timeline_end - item.timeline_start for item in self.timeline)
        if timeline_sum <= 0:
            raise ValueError("timeline durations must sum to a positive value")

        tolerance = self.total_duration * 0.10
        if abs(timeline_sum - self.total_duration) > tolerance:
            raise ValueError(
                f"total_duration must match the sum of timeline clip durations within 10% "
                f"(expected {self.total_duration:.3f}, got {timeline_sum:.3f})"
            )

        return self


class EDLValidationError(ValueError):
    def __init__(self, issues: List[Dict[str, Any]], raw_error: Optional[str] = None):
        self.issues = issues
        self.raw_error = raw_error
        super().__init__(self.format_message())

    @classmethod
    def from_pydantic(cls, error: ValidationError) -> "EDLValidationError":
        issues: List[Dict[str, Any]] = []
        for item in error.errors():
            loc = ".".join(str(part) for part in item.get("loc", ()))
            issues.append(
                {
                    "type": item.get("type", "validation_error"),
                    "field": loc,
                    "message": item.get("msg", "Invalid value"),
                }
            )
        return cls(issues, raw_error=str(error))

    def format_message(self) -> str:
        return self.to_feedback()

    def to_feedback(self) -> str:
        lines = ["EDL validation failed:"]
        for issue in self.issues:
            field = issue.get("field") or issue.get("clip_name") or "unknown"
            message = issue.get("message") or issue.get("error") or "Invalid EDL item"
            if issue.get("type") in {"target_duration_mismatch", "render_duration_mismatch"}:
                actual = issue.get("actual_duration")
                requested = issue.get("requested_duration")
                if actual is not None and requested is not None:
                    message = (
                        f"You only made {float(actual):.1f}s, make it {float(requested):.1f}s. "
                        f"{message}"
                    )
            lines.append(f"- {field}: {message}")
        return "\n".join(lines)


class EDLGenerationError(RuntimeError):
    def __init__(self, retry_count: int, last_error: str, issues: Optional[List[Dict[str, Any]]] = None):
        self.retry_count = retry_count
        self.last_error = last_error
        self.issues = issues or []
        super().__init__(self.format_message())

    def format_message(self) -> str:
        payload = {
            "retry_count": self.retry_count,
            "last_error": self.last_error,
            "issues": self.issues,
        }
        return f"EDLGenerationError: {json.dumps(payload, ensure_ascii=True)}"
