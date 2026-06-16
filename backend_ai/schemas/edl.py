import json
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class TransitionType(str, Enum):
    none = "none"
    jump_cut = "jump_cut"
    crossfade = "crossfade"
    dip_to_black = "dip_to_black"
    slide_left = "slide_left"
    slide_right = "slide_right"
    zoom_in = "zoom_in"
    zoom_out = "zoom_out"
    glitch = "glitch"


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


class EDLTimelineItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clip_name: str
    start_in_clip: float = Field(ge=0)
    end_in_clip: float = Field(ge=0)
    timeline_start: float = Field(ge=0)
    timeline_end: float = Field(ge=0)
    transition: TransitionType
    text_overlay: str = ""
    details: EDLClipDetails

    @model_validator(mode="after")
    def validate_ranges(self):
        if self.end_in_clip <= self.start_in_clip:
            raise ValueError("start_in_clip must be less than end_in_clip")
        if self.timeline_end <= self.timeline_start:
            raise ValueError("timeline_end must be greater than timeline_start")
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
