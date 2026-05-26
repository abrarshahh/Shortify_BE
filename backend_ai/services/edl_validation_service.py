import os
from typing import Any, Dict, List, Optional, Tuple

from moviepy import VideoFileClip
from pydantic import ValidationError

from backend_ai.core.config import (
    TARGET_DURATION_TOLERANCE,
    VIDEO_EDITOR_DEFAULT_FADE_DURATION,
    VIDEO_EDITOR_PACING_SPEEDS,
)
from backend_ai.schemas.edl import EDLDocument, EDLTimelineItem, EDLValidationError


def _target_duration_tolerance() -> float:
    try:
        return float(TARGET_DURATION_TOLERANCE)
    except Exception:
        return 0.15


def _parse_virtual_clip_name(clip_name: str) -> Optional[Tuple[str, float, float]]:
    parts = clip_name.split(":")
    if len(parts) != 3:
        return None

    source_filename, start_str, end_str = parts
    try:
        virtual_start = float(start_str)
        virtual_end = float(end_str)
    except ValueError:
        return None

    return source_filename, virtual_start, virtual_end


def _validate_clip_item(item: EDLTimelineItem, clips_dir: str, index: int) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    parsed = _parse_virtual_clip_name(item.clip_name)

    if parsed is None:
        clip_path = os.path.join(clips_dir, item.clip_name)
        if not os.path.exists(clip_path):
            issues.append(
                {
                    "index": index,
                    "clip_name": item.clip_name,
                    "type": "missing_clip",
                    "message": f"Clip file not found in clips_dir: {item.clip_name}",
                }
            )
        return issues

    source_filename, virtual_start, virtual_end = parsed
    clip_path = os.path.join(clips_dir, source_filename)

    if not os.path.exists(clip_path):
        issues.append(
            {
                "index": index,
                "clip_name": item.clip_name,
                "type": "missing_clip",
                "message": f"Base file for virtual segment not found: {source_filename}",
            }
        )
        return issues

    if virtual_start < 0 or virtual_end <= virtual_start:
        issues.append(
            {
                "index": index,
                "clip_name": item.clip_name,
                "type": "invalid_virtual_segment",
                "message": "Virtual segment bounds must be non-negative and end must be greater than start",
            }
        )
        return issues

    clip = VideoFileClip(clip_path)
    try:
        duration = float(getattr(clip, "duration", 0.0) or 0.0)
        if virtual_end > duration:
            issues.append(
                {
                    "index": index,
                    "clip_name": item.clip_name,
                    "type": "virtual_segment_out_of_bounds",
                    "message": (
                        f"Virtual segment end {virtual_end:.3f}s exceeds source duration {duration:.3f}s"
                    ),
                }
            )

        segment_duration = virtual_end - virtual_start
        if item.start_in_clip < 0 or item.end_in_clip > segment_duration:
            issues.append(
                {
                    "index": index,
                    "clip_name": item.clip_name,
                    "type": "virtual_subclip_out_of_bounds",
                    "message": (
                        f"Requested clip range {item.start_in_clip:.3f}s-{item.end_in_clip:.3f}s exceeds "
                        f"virtual segment length {segment_duration:.3f}s"
                    ),
                }
            )
    finally:
        clip.close()

    return issues


def validate_clip_existence(edl: EDLDocument, clips_dir: str) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    for index, item in enumerate(edl.timeline):
        issues.extend(_validate_clip_item(item, clips_dir, index))
    return issues


def validate_timeline_continuity(edl: EDLDocument) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    previous = None

    for index, item in enumerate(edl.timeline):
        if previous is None:
            previous = item
            continue

        if item.timeline_start <= previous.timeline_start:
            issues.append(
                {
                    "index": index,
                    "clip_name": item.clip_name,
                    "type": "timeline_order",
                    "message": (
                        f"Timeline is not in ascending order: clip at index {index} starts at "
                        f"{item.timeline_start:.3f}s after previous clip started at {previous.timeline_start:.3f}s"
                    ),
                }
            )

        gap = item.timeline_start - previous.timeline_end
        if gap < -1e-6:
            issues.append(
                {
                    "index": index,
                    "clip_name": item.clip_name,
                    "type": "timeline_overlap",
                    "message": (
                        f"Clip overlaps previous clip by {abs(gap):.3f}s "
                        f"(previous end {previous.timeline_end:.3f}s, current start {item.timeline_start:.3f}s)"
                    ),
                }
            )
        elif gap > 1.0:
            issues.append(
                {
                    "index": index,
                    "clip_name": item.clip_name,
                    "type": "timeline_gap",
                    "message": (
                        f"Gap of {gap:.3f}s between clips exceeds 1.0s "
                        f"(previous end {previous.timeline_end:.3f}s, current start {item.timeline_start:.3f}s)"
                    ),
                }
            )

        previous = item

    return issues


def _is_image_clip(clip_name: str) -> bool:
    source_name = clip_name.split(":", 1)[0]
    ext = os.path.splitext(source_name)[1].lower()
    return ext in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".heic", ".heif"}


def validate_expected_render_duration(edl: EDLDocument, target_duration: float) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    estimated_duration = 0.0

    for index, item in enumerate(edl.timeline):
        raw_duration = max(0.0, float(item.timeline_end) - float(item.timeline_start))
        if raw_duration <= 0:
            continue

        if _is_image_clip(item.clip_name):
            effective_duration = raw_duration
        else:
            pacing = item.details.pacing_style.value
            speed = VIDEO_EDITOR_PACING_SPEEDS.get(pacing, 1.0)
            if speed <= 0:
                speed = 1.0
            effective_duration = min(raw_duration, raw_duration / speed)

        transition = item.transition.value
        if index > 0 and transition in {"crossfade", "fade", "slide_left", "slide_right"}:
            estimated_duration -= min(VIDEO_EDITOR_DEFAULT_FADE_DURATION, effective_duration / 2)
        elif index > 0 and transition == "dip_to_black":
            estimated_duration += min(VIDEO_EDITOR_DEFAULT_FADE_DURATION, effective_duration / 2)

        estimated_duration += effective_duration

    tolerance_ratio = _target_duration_tolerance()
    tolerance = max(1.0, float(target_duration) * tolerance_ratio)
    if abs(estimated_duration - float(target_duration)) > tolerance:
        issues.append(
            {
                "index": 0,
                "clip_name": "edl",
                "type": "render_duration_mismatch",
                "message": (
                    f"Estimated rendered duration {estimated_duration:.3f}s does not match requested "
                    f"target_duration {float(target_duration):.3f}s within {tolerance:.3f}s tolerance"
                ),
                "actual_duration": estimated_duration,
                "requested_duration": float(target_duration),
                "tolerance_ratio": tolerance_ratio,
                "tolerance_seconds": tolerance,
            }
        )

    return issues


def validate_edl(
    edl_data: Dict[str, Any],
    clips_dir: str,
    target_duration: Optional[float] = None,
) -> EDLDocument:
    try:
        edl = EDLDocument.model_validate(edl_data)
    except ValidationError as exc:
        raise EDLValidationError.from_pydantic(exc)
    except Exception as exc:
        raise EDLValidationError(
            [
                {
                    "type": "validation_error",
                    "field": "edl",
                    "message": str(exc),
                }
            ],
            raw_error=str(exc),
        )

    issues = validate_clip_existence(edl, clips_dir)
    issues.extend(validate_timeline_continuity(edl))
    if target_duration is not None:
        issues.extend(validate_expected_render_duration(edl, float(target_duration)))

    if target_duration is not None:
        tolerance_ratio = _target_duration_tolerance()
        tolerance = max(1.0, float(target_duration) * tolerance_ratio)
        actual_duration = float(edl.total_duration)
        requested_duration = float(target_duration)
        if abs(actual_duration - requested_duration) > tolerance:
            issues.append(
                {
                    "index": 0,
                    "clip_name": "edl",
                    "type": "target_duration_mismatch",
                    "message": (
                        f"EDL total_duration {actual_duration:.3f}s does not match "
                        f"requested target_duration {requested_duration:.3f}s within {tolerance:.3f}s tolerance"
                    ),
                    "actual_duration": actual_duration,
                    "requested_duration": requested_duration,
                    "tolerance_ratio": tolerance_ratio,
                    "tolerance_seconds": tolerance,
                }
            )

    if issues:
        raise EDLValidationError(issues)

    return edl
