import os
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
    effect_asset_id: Optional[str] = ""
    sticker_asset_id: Optional[str] = ""


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


class ClipEffectParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    effect_type: str = Field("none", description="Type of visual filter/effect to apply. E.g. 'none', 'blur', 'glitch', 'pixelate', 'ripple', 'spin', 'light_leak'")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Variables custom-tailoring the filter")


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
    clip_effect: Optional[ClipEffectParams] = None
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
            valid_text_presets = {"bold_hype", "classic_clean", "neon_glow", "minimal_pop", "none"}
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
    global_color_grade: Optional[ColorGradeParams] = None
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


# =====================================================================
# Phase 10: Multi-Track Timeline & Editing IR Schemas
# =====================================================================

class VisualProperties(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: float = 0.0          # Normalized screen position x offset from center
    y: float = 0.0          # Normalized screen position y offset from center
    scale: float = 1.0      # Scale multiplier
    rotation: float = 0.0   # Rotation in degrees
    opacity: float = Field(1.0, ge=0.0, le=1.0)
    anchor: str = "center"  # "center" | "top-left" | "top-right" | "bottom-left" | "bottom-right"
    crop: Optional[Tuple[float, float, float, float]] = None # (x1, y1, x2, y2)
    blur: float = 0.0       # Blur radius/strength

class TimelineClip(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    source: str
    start_in_clip: float = Field(ge=0.0)
    end_in_clip: float = Field(ge=0.0)
    timeline_start: float = Field(ge=0.0)
    timeline_end: float = Field(ge=0.0)
    layer: int = Field(1, ge=0, le=5)  # 0: Background, 1-2: Video/Main, 3: Overlay/B-roll, 4: Captions, 5: Stickers

    speed: float = Field(1.0, gt=0.0)
    reverse: bool = False
    mute: bool = False
    stabilize: bool = False
    stabilize_strength: float = Field(0.5, ge=0.0, le=1.0)

    transition_in: TransitionType = TransitionType.none
    transition_in_duration: float = Field(0.0, ge=0.0)
    transition_out: TransitionType = TransitionType.none
    transition_out_duration: float = Field(0.0, ge=0.0)
    transition_params: Optional[Dict[str, Any]] = None

    color_grade: Optional[ColorGradeParams] = None
    visual_properties: Optional[VisualProperties] = None

    clip_effect: Optional[ClipEffectParams] = None
    effect_asset_id: Optional[str] = ""
    sticker_asset_id: Optional[str] = ""

    # Keyframes representing list of (time_fraction, value)
    scale_keyframes: Optional[List[Tuple[float, float]]] = None
    opacity_keyframes: Optional[List[Tuple[float, float]]] = None

    @model_validator(mode="after")
    def validate_clip_timing(self):
        if self.end_in_clip <= self.start_in_clip:
            raise ValueError("end_in_clip must be greater than start_in_clip")
        if self.timeline_end <= self.timeline_start:
            raise ValueError("timeline_end must be greater than timeline_start")
        return self

class TimelineAudio(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    source: str
    start_in_audio: float = Field(ge=0.0)
    end_in_audio: float = Field(ge=0.0)
    timeline_start: float = Field(ge=0.0)
    timeline_end: float = Field(ge=0.0)

    volume: float = Field(1.0, ge=0.0, le=2.0)
    pitch: float = Field(1.0, ge=0.5, le=2.0)
    speed: float = Field(1.0, gt=0.0)
    fade_in: float = Field(0.0, ge=0.0)
    fade_out: float = Field(0.0, ge=0.0)
    loop: bool = False
    ducking_enabled: bool = False
    ducking_target_volume: float = Field(0.1, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_audio_timing(self):
        if self.end_in_audio <= self.start_in_audio:
            raise ValueError("end_in_audio must be greater than start_in_audio")
        if self.timeline_end <= self.timeline_start:
            raise ValueError("timeline_end must be greater than timeline_start")
        return self

class TimelineText(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    text: str
    timeline_start: float = Field(ge=0.0)
    timeline_end: float = Field(ge=0.0)
    layer: int = Field(4, ge=0, le=5)

    font: str = "Arial"
    font_size: int = Field(40, ge=1)
    weight: int = Field(700, ge=100, le=900)
    italic: bool = False
    underline: bool = False
    color: str = "white"
    opacity: float = Field(1.0, ge=0.0, le=1.0)

    alignment: str = "center" # "center", "left", "right"
    x: float = 0.0
    y: float = 0.0

    stroke_color: str = "black"
    stroke_width: int = Field(2, ge=0)
    shadow_color: str = "none"
    shadow_width: int = Field(0, ge=0)
    background_color: str = "none"

    animation: str = "none" # "none", "fade", "slide_up", "slide_down", "slide_left", "slide_right"
    animation_duration: float = Field(0.3, ge=0.0)

    @model_validator(mode="after")
    def validate_text_timing(self):
        if self.timeline_end <= self.timeline_start:
            raise ValueError("timeline_end must be greater than timeline_start")
        return self

class TimelineSticker(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    sticker_asset_id: str
    timeline_start: float = Field(ge=0.0)
    timeline_end: float = Field(ge=0.0)
    layer: int = Field(5, ge=0, le=5)

    x: float = 0.0
    y: float = 0.0
    scale: float = 1.0
    rotation: float = 0.0
    opacity: float = Field(1.0, ge=0.0, le=1.0)
    animation: str = "none"

    @model_validator(mode="after")
    def validate_sticker_timing(self):
        if self.timeline_end <= self.timeline_start:
            raise ValueError("timeline_end must be greater than timeline_start")
        return self

class TimelineIR(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    storyline: str
    total_duration: float = Field(gt=0.0)
    style: str = "general"
    global_color_grade: Optional[ColorGradeParams] = None
    
    video_clips: List[TimelineClip] = Field(default_factory=list)
    audio_clips: List[TimelineAudio] = Field(default_factory=list)
    text_overlays: List[TimelineText] = Field(default_factory=list)
    stickers: List[TimelineSticker] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_durations(self):
        if not self.video_clips and not self.audio_clips:
            raise ValueError("TimelineIR must contain at least one video_clip or audio_clip")
        return self


def convert_edl_to_timeline_ir(edl: EDLDocument) -> TimelineIR:
    """Converts a legacy EDLDocument into the new TimelineIR format."""
    video_clips = []
    text_overlays = []
    stickers = []
    
    for i, item in enumerate(edl.timeline):
        clip_id = f"clip_{i}_{os.path.basename(item.clip_name).split('.')[0]}"
        
        # Parse transition duration mapping if set
        t_in_dur = 0.0
        if item.transition != TransitionType.none:
            t_in_dur = 0.3 # Legacy fade/crossfade default
            
        clip = TimelineClip(
            id=clip_id,
            source=item.clip_name,
            start_in_clip=item.start_in_clip,
            end_in_clip=item.end_in_clip,
            timeline_start=item.timeline_start,
            timeline_end=item.timeline_end,
            layer=1, # Legacy always maps to Layer 1 (Video)
            speed=1.0,
            reverse=item.reverse or False,
            mute=False,
            stabilize=item.stabilize or False,
            stabilize_strength=item.stabilize_strength if item.stabilize_strength is not None else 0.5,
            transition_in=item.transition,
            transition_in_duration=t_in_dur,
            transition_out=TransitionType.none,
            transition_out_duration=0.0,
            transition_params=item.transition_params,
            color_grade=item.color_grade or edl.global_color_grade,
            visual_properties=None,
            clip_effect=item.clip_effect,
            effect_asset_id=item.details.effect_asset_id or "",
            sticker_asset_id=item.details.sticker_asset_id or ""
        )
        
        # Legacy speed multiplier parsing
        if item.speed_preset == "constant_fast":
            clip.speed = 2.0
        elif item.speed_preset == "constant_slow":
            clip.speed = 0.5
        elif item.speed_preset in ("ramp_up", "ramp_down"):
            clip.speed = 1.13
        elif item.speed_preset == "speed_bump":
            clip.speed = 1.2
            
        video_clips.append(clip)
        
        # Parse text overlay
        if item.text_overlay:
            text_id = f"text_{i}"
            text_item = TimelineText(
                id=text_id,
                text=item.text_overlay,
                timeline_start=item.timeline_start,
                timeline_end=item.timeline_end,
                layer=4, # Text layer
                font="Arial",
                font_size=40,
                color="white"
            )
            # Map presets/animations
            if item.text_preset:
                if item.text_preset == "bold_hype":
                    text_item.font_size = 52
                    text_item.color = "yellow"
                elif item.text_preset == "neon_glow":
                    text_item.color = "cyan"
            if item.text_animation:
                text_item.animation = item.text_animation
            text_overlays.append(text_item)
            
        # Parse stickers
        # Legacy items set sticker_path if they resolved stickers
        # Fall back to matching standard asset IDs
        sticker_src = item.sticker_path or ""
        if sticker_src:
            sticker_id = f"sticker_{i}"
            asset_id = "sticker_subscribe"
            if "arrow" in sticker_src.lower():
                asset_id = "sticker_arrow"
            elif "fire" in sticker_src.lower():
                asset_id = "sticker_fire"
                
            sticker_item = TimelineSticker(
                id=sticker_id,
                sticker_asset_id=asset_id,
                timeline_start=item.timeline_start,
                timeline_end=item.timeline_end,
                layer=5, # Sticker layer
                x=0.0,
                y=0.0,
                scale=1.0,
                animation=item.sticker_animation or "none"
            )
            stickers.append(sticker_item)
            
    # Legacy background music mapping
    audio_clips = []
    if edl.music_start_offset is not None:
        # Resolve background music
        music_clip = TimelineAudio(
            id="bg_music",
            source="background_music", # dynamic mapping key
            start_in_audio=edl.music_start_offset,
            end_in_audio=edl.music_start_offset + edl.total_duration,
            timeline_start=0.0,
            timeline_end=edl.total_duration,
            volume=0.70,
            ducking_enabled=True,
            ducking_target_volume=0.22
        )
        audio_clips.append(music_clip)
        
    return TimelineIR(
        title=edl.title,
        storyline=edl.storyline,
        total_duration=edl.total_duration,
        video_clips=video_clips,
        audio_clips=audio_clips,
        text_overlays=text_overlays,
        stickers=stickers,
        style=getattr(edl, "style", "general") or "general",
        global_color_grade=edl.global_color_grade
    )
