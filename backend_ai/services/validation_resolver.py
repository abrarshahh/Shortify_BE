import logging
from typing import Dict, Any, List
from backend_ai.schemas.edl import TimelineIR

logger = logging.getLogger("services.validation_resolver")

class ValidationConflictResolver:
    """
    Validates parameter boundaries and resolves resource/property conflicts
    within the Multi-Track Timeline IR.
    """
    
    def resolve_conflicts(self, timeline: TimelineIR) -> List[Dict[str, Any]]:
        """
        Validates the timeline IR, modifies conflicting values in-place to enforce boundaries,
        and returns a list of warning/resolution log dicts describing what was fixed.
        """
        resolutions = []
        
        # 1. Validate and resolve Video Clips
        for clip in timeline.video_clips:
            # Clamp Speed factor
            if clip.speed < 0.25:
                old_speed = clip.speed
                clip.speed = 0.25
                resolutions.append({
                    "component": "video_clip",
                    "id": clip.id,
                    "field": "speed",
                    "message": f"Clamped speed factor from {old_speed}x to minimum 0.25x."
                })
            elif clip.speed > 8.0:
                old_speed = clip.speed
                clip.speed = 8.0
                resolutions.append({
                    "component": "video_clip",
                    "id": clip.id,
                    "field": "speed",
                    "message": f"Clamped speed factor from {old_speed}x to maximum 8.0x."
                })
                
            # Compute effective duration
            eff_dur = (clip.end_in_clip - clip.start_in_clip) / clip.speed
            
            # Transition overlap checks
            max_trans_dur = eff_dur * 0.5
            if clip.transition_in_duration > max_trans_dur:
                old_dur = clip.transition_in_duration
                clip.transition_in_duration = round(max_trans_dur, 2)
                resolutions.append({
                    "component": "video_clip",
                    "id": clip.id,
                    "field": "transition_in_duration",
                    "message": f"Reduced transition_in_duration from {old_dur}s to {clip.transition_in_duration}s (capped at 50% of effective clip duration: {eff_dur:.2f}s)."
                })
            if clip.transition_out_duration > max_trans_dur:
                old_dur = clip.transition_out_duration
                clip.transition_out_duration = round(max_trans_dur, 2)
                resolutions.append({
                    "component": "video_clip",
                    "id": clip.id,
                    "field": "transition_out_duration",
                    "message": f"Reduced transition_out_duration from {old_dur}s to {clip.transition_out_duration}s (capped at 50% of effective clip duration: {eff_dur:.2f}s)."
                })
                
            # Stabilization & Reverse conflict
            if clip.reverse and clip.stabilize:
                clip.stabilize = False
                resolutions.append({
                    "component": "video_clip",
                    "id": clip.id,
                    "field": "stabilize",
                    "message": "Disabled stabilization because reverse playback is enabled (mutually incompatible features)."
                })
                
        # 2. Validate and resolve Audio Clips
        for audio in timeline.audio_clips:
            # Clamp Speed factor
            if audio.speed < 0.25:
                old_speed = audio.speed
                audio.speed = 0.25
                resolutions.append({
                    "component": "audio_clip",
                    "id": audio.id,
                    "field": "speed",
                    "message": f"Clamped audio speed factor from {old_speed}x to minimum 0.25x."
                })
            elif audio.speed > 8.0:
                old_speed = audio.speed
                audio.speed = 8.0
                resolutions.append({
                    "component": "audio_clip",
                    "id": audio.id,
                    "field": "speed",
                    "message": f"Clamped audio speed factor from {old_speed}x to maximum 8.0x."
                })
                
        # 3. Validate and resolve Text Safe-Zone
        for text in timeline.text_overlays:
            # Clamp y coordinates to safe-zones (avoiding top-bottom 15% header/footer overlap)
            # Normalized ranges: -1.0 to 1.0 (where -1.0 is top and 1.0 is bottom)
            # Safe zone y range: -0.7 to 0.7
            if text.y < -0.7:
                old_y = text.y
                text.y = -0.7
                resolutions.append({
                    "component": "text_overlay",
                    "id": text.id,
                    "field": "y",
                    "message": f"Clamped text y position from {old_y} to -0.7 to stay within vertical safe zone."
                })
            elif text.y > 0.7:
                old_y = text.y
                text.y = 0.7
                resolutions.append({
                    "component": "text_overlay",
                    "id": text.id,
                    "field": "y",
                    "message": f"Clamped text y position from {old_y} to 0.7 to stay within vertical safe zone."
                })
                
            # Safe zone x range: -0.8 to 0.8
            if text.x < -0.8:
                old_x = text.x
                text.x = -0.8
                resolutions.append({
                    "component": "text_overlay",
                    "id": text.id,
                    "field": "x",
                    "message": f"Clamped text x position from {old_x} to -0.8 to stay within horizontal safe zone."
                })
            elif text.x > 0.8:
                old_x = text.x
                text.x = 0.8
                resolutions.append({
                    "component": "text_overlay",
                    "id": text.id,
                    "field": "x",
                    "message": f"Clamped text x position from {old_x} to 0.8 to stay within horizontal safe zone."
                })

        return resolutions
