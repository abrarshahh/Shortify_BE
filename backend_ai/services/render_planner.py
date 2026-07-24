import os
import logging
from typing import Dict, Any, List, Tuple, Optional
from backend_ai.schemas.edl import TimelineIR, TransitionType
from backend_ai.utils.asset_manager import resolve_asset_path

logger = logging.getLogger("services.render_planner")

class RenderPlanner:
    def __init__(self, clips_dir: str, output_dir: str):
        self.clips_dir = clips_dir
        self.output_dir = output_dir

    def _check_has_audio(self, file_path: str) -> bool:
        import subprocess
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "error",
                    "-select_streams", "a",
                    "-show_entries", "stream=codec_type",
                    "-of", "csv=p=0",
                    file_path
                ],
                capture_output=True,
                text=True,
                check=True,
                stdin=subprocess.DEVNULL
            )
            return "audio" in result.stdout.lower()
        except Exception:
            return False

    def _find_font(self, font_name: str = "Arial") -> str:
        candidates = [
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibri.ttf",
            "C:/Windows/Fonts/verdana.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
        for path in candidates:
            if os.path.exists(path):
                return path.replace("\\", "/")
        return "Arial"

    def compile_timeline_to_ffmpeg_cmd(
        self,
        timeline: TimelineIR,
        output_filename: str,
        aspect_ratio: str = "9:16",
        clip_scores: Optional[Dict[str, Any]] = None,
        audio_ducking: bool = True,
        music_path: Optional[str] = None
    ) -> List[str]:
        """
        Parses the multi-track TimelineIR and compiles it into a single, optimized FFmpeg Command.
        """
        dimensions_map = {
            "9:16": (1080, 1920),
            "16:9": (1920, 1080),
            "1:1": (1080, 1080),
        }
        target_w, target_h = dimensions_map.get(aspect_ratio, (1080, 1920))
        output_path = os.path.join(self.output_dir, output_filename)

        # 1. Map all inputs and assign input indices
        inputs = []
        input_map = {}  # source_path -> input_index

        def add_input(path: str) -> int:
            normalized_path = os.path.abspath(path).replace("\\", "/")
            if normalized_path not in input_map:
                input_map[normalized_path] = len(inputs)
                inputs.append(normalized_path)
            return input_map[normalized_path]

        # Register video clips
        for clip in timeline.video_clips:
            # Check if source exists in clips_dir
            clip_file = os.path.join(self.clips_dir, clip.source)
            if not os.path.exists(clip_file):
                # Fallback to absolute/relative check
                clip_file = clip.source
            add_input(clip_file)

        # Register custom transition masks
        for clip in timeline.video_clips:
            if clip.transition_in != TransitionType.none and clip.transition_params:
                if clip.transition_params.get("mask_mode") == "custom":
                    mask_path = clip.transition_params.get("custom_mask_path")
                    if mask_path and os.path.exists(mask_path):
                        add_input(mask_path)

        # Register overlays (Layer 3 effects)
        for clip in timeline.video_clips:
            if clip.color_grade or (hasattr(clip, "effect_asset_id") and getattr(clip, "effect_asset_id")):
                asset_id = getattr(clip, "effect_asset_id", "")
                if asset_id:
                    path = resolve_asset_path("overlays", asset_id)
                    if path:
                        add_input(path)

        # Register stickers (Layer 5)
        for sticker in timeline.stickers:
            path = resolve_asset_path("stickers", sticker.sticker_asset_id)
            if path:
                add_input(path)

        # Register audios (ambient, SFX)
        for audio in timeline.audio_clips:
            if audio.source == "background_music" and music_path:
                path = music_path
            # Check if SFX
            elif audio.source.startswith("sfx_"):
                path = resolve_asset_path("sfx", audio.source)
            else:
                path = os.path.join(self.clips_dir, audio.source)
                if not os.path.exists(path):
                    path = audio.source
            if path and os.path.exists(path):
                add_input(path)

        # 2. Build Filter Complex strings
        filter_complex_parts = []
        
        # Video streams process
        video_concat_inputs = []
        audio_concat_inputs = []

        for i, clip in enumerate(timeline.video_clips):
            clip_file = os.path.join(self.clips_dir, clip.source)
            if not os.path.exists(clip_file):
                clip_file = clip.source
            input_idx = input_map[os.path.abspath(clip_file).replace("\\", "/")]

            ext = os.path.splitext(clip_file)[1].lower()
            is_image = ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            has_audio = (not is_image) and self._check_has_audio(clip_file)

            # Get original video/image size
            clip_w, clip_h = target_w, target_h
            if not is_image:
                import cv2
                cap = cv2.VideoCapture(clip_file)
                if cap.isOpened():
                    clip_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    clip_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()

            # Horizontally center the crop relative to face_anchor_x
            face_x = 0.5
            if clip_scores and clip.source in clip_scores:
                face_x = clip_scores[clip.source].get("face_anchor_x", 0.5)

            if face_x == 0.5 and not is_image:
                import cv2
                mid_time = (clip.start_in_clip + clip.end_in_clip) / 2
                cap = cv2.VideoCapture(clip_file)
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_idx = int(mid_time * fps) if fps > 0 else 0
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        try:
                            from backend_ai.services.editor_service import detect_face_center
                            face_center = detect_face_center(frame_rgb)
                            if face_center:
                                face_x = face_center[0]
                        except Exception:
                            pass
                cap.release()

            scale = max(target_w / clip_w, target_h / clip_h)
            new_w, new_h = int(clip_w * scale), int(clip_h * scale)
            crop_center_x = face_x * new_w
            x1 = max(0, min(new_w - target_w, int(crop_center_x - target_w / 2)))
            y1 = (new_h - target_h) // 2

            # Process Video: trim, speed, scale, crop
            v_label = f"v_proc_{i}"
            speed_val = clip.speed if clip.speed else 1.0
            pts_multiplier = 1.0 / speed_val
            
            color_filter_str = ""
            clip_color_grade = clip.color_grade
            if clip_color_grade and timeline.global_color_grade and clip_color_grade.model_dump() == timeline.global_color_grade.model_dump():
                # Avoid double color grading if a global style/LUT is active
                style_lower = (timeline.style or "general").lower()
                luts_dir = os.path.abspath(os.path.join("data", "luts"))
                lut_exists = os.path.exists(os.path.join(luts_dir, f"{style_lower}.cube"))
                if not lut_exists and style_lower != "general" and style_lower != "none":
                    lut_exists = True
                if lut_exists:
                    clip_color_grade = None

            if clip_color_grade:
                from backend_ai.effects.color import ColorGradeParams as EffectColorGradeParams, build_ffmpeg_color_filter
                params_dict = clip_color_grade.model_dump()
                effect_params = EffectColorGradeParams(**params_dict)
                color_filter_str = build_ffmpeg_color_filter(effect_params)

            clip_effect_str = ""
            if clip.clip_effect and clip.clip_effect.effect_type != "none":
                effect_type = clip.clip_effect.effect_type
                params = clip.clip_effect.parameters or {}
                if effect_type == "blur":
                    max_blur_size = int(params.get("max_blur_size", 51))
                    if max_blur_size % 2 == 0:
                        max_blur_size = max(1, max_blur_size - 1)
                    radius = max(1, max_blur_size // 4)
                    clip_effect_str = f"boxblur=lr={radius}:lp=2"
                elif effect_type == "pixelate":
                    cell_size = int(params.get("max_cell_size", 32))
                    if cell_size <= 1:
                        cell_size = 32
                    clip_effect_str = f"scale=iw/{cell_size}:ih/{cell_size},scale={target_w}:{target_h}:flags=neighbor"
                elif effect_type == "vignette":
                    vignette_strength = float(params.get("vignette_strength", 0.5))
                    import math
                    angle = vignette_strength * (math.pi / 4.0)
                    clip_effect_str = f"vignette=angle={angle:.4f}"
                elif effect_type == "mirror":
                    direction = str(params.get("direction", "horizontal")).lower()
                    if direction == "vertical":
                        clip_effect_str = "vflip"
                    else:
                        clip_effect_str = "hflip"
                elif effect_type == "glitch":
                    clip_effect_str = "noise=alls=20:allf=t+u"
                elif effect_type == "light_leak":
                    clip_effect_str = "eq=brightness=0.1:contrast=1.1"

            scale_filter = f"scale={new_w}:{new_h},crop={target_w}:{target_h}:{x1}:{y1}"
            if color_filter_str:
                scale_filter = f"{scale_filter},{color_filter_str}"
            if clip_effect_str:
                scale_filter = f"{scale_filter},{clip_effect_str}"
            
            if is_image:
                dur = clip.end_in_clip - clip.start_in_clip
                filter_complex_parts.append(
                    f"[{input_idx}:v]loop=loop=-1:size=1:start=0,trim=duration={dur},{scale_filter}[{v_label}]"
                )
            else:
                trim_filter = f"trim=start={clip.start_in_clip}:end={clip.end_in_clip},setpts={pts_multiplier}*(PTS-STARTPTS)"
                filter_complex_parts.append(
                    f"[{input_idx}:v]{trim_filter},{scale_filter}[{v_label}]"
                )
            video_concat_inputs.append(f"[{v_label}]")

            # Process Audio
            a_label = f"a_proc_{i}"
            vol_val = 0.0 if (clip.mute or audio_ducking) else 1.0
            dur = clip.end_in_clip - clip.start_in_clip
            eff_dur = dur * pts_multiplier

            if has_audio and not audio_ducking:
                atrim_filter = f"atrim=start={clip.start_in_clip}:end={clip.end_in_clip},asetpts={pts_multiplier}*(PTS-STARTPTS)"
                filter_complex_parts.append(
                    f"[{input_idx}:a]{atrim_filter},volume={vol_val}[{a_label}]"
                )
            else:
                filter_complex_parts.append(
                    f"anullsrc=channel_layout=stereo:sample_rate=44100:d={eff_dur},volume={vol_val}[{a_label}]"
                )
            audio_concat_inputs.append(f"[{a_label}]")

        # 3. Apply Transitions and Segmentation
        final_v_streams = []
        final_a_streams = []
        
        transition_map = {
            "crossfade": "fade",
            "fade": "fade",
            "dip_to_black": "fadeblack",
            "slide_left": "slideleft",
            "slide_right": "slideright",
            "slide_up": "slideup",
            "slide_down": "slidedown",
            "wipe_left": "wipeleft",
            "wipe_right": "wiperight",
            "wipe_up": "wipeup",
            "wipe_down": "wipedown",
            "iris_circle": "circleopen",
            "circleopen": "circleopen",
            "zoom_in": "zoomin",
            "zoom_out": "zoomout",
            "glitch": "slidedown",
            "pixelate": "fade",
            "spin": "fade",
            "ripple": "fade",
            "blur": "fade",
        }

        # Precompute clip durations
        clip_durations = []
        for clip in timeline.video_clips:
            speed_val = clip.speed if clip.speed else 1.0
            dur = (clip.end_in_clip - clip.start_in_clip) / speed_val
            clip_durations.append(dur)

        # Generate segments and transitions in chronological order
        for i, clip in enumerate(timeline.video_clips):
            dur_i = clip_durations[i]
            
            fade_in_dur = 0.0
            if i > 0:
                prev_dur = clip_durations[i - 1]
                fade_in_dur = min(clip.transition_in_duration or 0.5, prev_dur / 2, dur_i / 2)
                
            fade_out_dur = 0.0
            if i < len(timeline.video_clips) - 1:
                next_clip = timeline.video_clips[i + 1]
                next_dur = clip_durations[i + 1]
                fade_out_dur = min(next_clip.transition_in_duration or 0.5, dur_i / 2, next_dur / 2)

            # Slice clip if transition exists on either side
            if fade_in_dur > 0.05 or fade_out_dur > 0.05:
                # Slice start transition
                if fade_in_dur > 0.05:
                    filter_complex_parts.append(
                        f"[v_proc_{i}]trim=start=0:duration={fade_in_dur},setpts=PTS-STARTPTS[v_proc_{i}_start]"
                    )
                    filter_complex_parts.append(
                        f"[a_proc_{i}]atrim=start=0:duration={fade_in_dur},asetpts=PTS-STARTPTS[a_proc_{i}_start]"
                    )

                # Slice end transition
                if fade_out_dur > 0.05:
                    filter_complex_parts.append(
                        f"[v_proc_{i}]trim=start={dur_i - fade_out_dur}:duration={fade_out_dur},setpts=PTS-STARTPTS[v_proc_{i}_end]"
                    )
                    filter_complex_parts.append(
                        f"[a_proc_{i}]atrim=start={dur_i - fade_out_dur}:duration={fade_out_dur},asetpts=PTS-STARTPTS[a_proc_{i}_end]"
                    )

                # Slice main body
                main_dur = dur_i - fade_in_dur - fade_out_dur
                if main_dur > 0.05:
                    filter_complex_parts.append(
                        f"[v_proc_{i}]trim=start={fade_in_dur}:duration={main_dur},setpts=PTS-STARTPTS[v_proc_{i}_main]"
                    )
                    filter_complex_parts.append(
                        f"[a_proc_{i}]atrim=start={fade_in_dur}:duration={main_dur},asetpts=PTS-STARTPTS[a_proc_{i}_main]"
                    )
            
            # Append to streams chronologically
            if i > 0 and fade_in_dur > 0.05:
                # 1. Bake and append the transition from clip i-1 to clip i
                params = clip.transition_params or {}
                if params.get("mask_mode") == "custom" and params.get("custom_mask_path"):
                    mask_path = params["custom_mask_path"]
                    norm_mask_path = os.path.abspath(mask_path).replace("\\", "/")
                    if norm_mask_path in input_map:
                        mask_input_idx = input_map[norm_mask_path]
                        filter_complex_parts.append(
                            f"[{mask_input_idx}:v]scale={target_w}:{target_h},trim=duration={fade_in_dur},setpts=PTS-STARTPTS[v_mask_{i}]"
                        )
                        filter_complex_parts.append(
                            f"[v_proc_{i-1}_end][v_proc_{i}_start][v_mask_{i}]maskedmerge[v_trans_{i}]"
                        )
                        final_v_streams.append(f"[v_trans_{i}]")
                    else:
                        filter_complex_parts.append(
                            f"[v_proc_{i-1}_end][v_proc_{i}_start]xfade=transition=fade:duration={fade_in_dur}:offset=0[v_trans_{i}]"
                        )
                        final_v_streams.append(f"[v_trans_{i}]")
                else:
                    trans_val = str(clip.transition_in.value).lower()
                    xfade_trans = transition_map.get(trans_val, "fade")
                    filter_complex_parts.append(
                        f"[v_proc_{i-1}_end][v_proc_{i}_start]xfade=transition={xfade_trans}:duration={fade_in_dur}:offset=0[v_trans_{i}]"
                    )
                    final_v_streams.append(f"[v_trans_{i}]")

                # Bake and append the audio transition (acrossfade)
                filter_complex_parts.append(
                    f"[a_proc_{i-1}_end][a_proc_{i}_start]acrossfade=d={fade_in_dur}[a_trans_{i}]"
                )
                final_a_streams.append(f"[a_trans_{i}]")

            # 2. Append the main body of clip i
            if fade_in_dur > 0.05 or fade_out_dur > 0.05:
                main_dur = dur_i - fade_in_dur - fade_out_dur
                if main_dur > 0.05:
                    final_v_streams.append(f"[v_proc_{i}_main]")
                    final_a_streams.append(f"[a_proc_{i}_main]")
            else:
                final_v_streams.append(f"[v_proc_{i}]")
                final_a_streams.append(f"[a_proc_{i}]")

        # Video Concatenation
        filter_complex_parts.append(
            f"{''.join(final_v_streams)}concat=n={len(final_v_streams)}:v=1:a=0[v_main_concat]"
        )
        
        # Audio Concatenation
        filter_complex_parts.append(
            f"{''.join(final_a_streams)}concat=n={len(final_a_streams)}:v=0:a=1[a_main_concat]"
        )

        current_v = "v_main_concat"

        # Apply Overlays (blend effect overlays if any)
        for i, clip in enumerate(timeline.video_clips):
            asset_id = getattr(clip, "effect_asset_id", "")
            if asset_id:
                path = resolve_asset_path("overlays", asset_id)
                if path:
                    overlay_idx = input_map[os.path.abspath(path).replace("\\", "/")]
                    next_v = f"v_overlay_{i}"
                    # Overlay logic: place overlay layer over main stream with blend opacity
                    # We can use overlay filter with shortest=1
                    filter_complex_parts.append(
                        f"[{current_v}][{overlay_idx}:v]overlay=shortest=1[{next_v}]"
                    )
                    current_v = next_v

        # Apply Stickers
        for i, sticker in enumerate(timeline.stickers):
            path = resolve_asset_path("stickers", sticker.sticker_asset_id)
            if path:
                stk_idx = input_map[os.path.abspath(path).replace("\\", "/")]
                
                # Scale sticker to reasonable dimensions
                stk_v_label = f"stk_scale_{i}"
                filter_complex_parts.append(f"[{stk_idx}:v]scale=250:-1[{stk_v_label}]")
                
                next_v = f"v_stk_{i}"
                # Convert normalized coordinates x, y to pixel offsets
                pixel_x = int((sticker.x + 1.0) / 2.0 * target_w) - 125
                pixel_y = int((sticker.y + 1.0) / 2.0 * target_h) - 125
                
                enable_filter = f"between(t,{sticker.timeline_start},{sticker.timeline_end})"
                
                filter_complex_parts.append(
                    f"[{current_v}][{stk_v_label}]overlay=x={pixel_x}:y={pixel_y}:enable='{enable_filter}'[{next_v}]"
                )
                current_v = next_v

        # Apply Text/Captions using drawtext
        font_file = self._find_font()
        for i, text in enumerate(timeline.text_overlays):
            next_v = f"v_txt_{i}"
            pixel_x = int((text.x + 1.0) / 2.0 * target_w) - 150
            pixel_y = int((text.y + 1.0) / 2.0 * target_h) - 30
            
            # Simple text escaping
            clean_text = text.text.replace("'", "'\\''").replace(":", "\\:")
            enable_filter = f"between(t,{text.timeline_start},{text.timeline_end})"
            
            drawtext_str = (
                f"drawtext=fontfile='{font_file}':text='{clean_text}':"
                f"fontsize={text.font_size}:fontcolor={text.color}:"
                f"x={pixel_x}:y={pixel_y}:enable='{enable_filter}':"
                f"borderw={text.stroke_width}:bordercolor={text.stroke_color}"
            )
            
            filter_complex_parts.append(
                f"[{current_v}]{drawtext_str}[{next_v}]"
            )
            current_v = next_v

        # Final video output stream label
        filter_complex_parts.append(f"[{current_v}]copy[outv]")

        # Mix Audio Tracks
        mixed_audio_sources = ["[a_main_concat]"]
        for i, audio in enumerate(timeline.audio_clips):
            # Check path
            if audio.source == "background_music" and music_path:
                path = music_path
            elif audio.source.startswith("sfx_"):
                path = resolve_asset_path("sfx", audio.source)
            else:
                path = os.path.join(self.clips_dir, audio.source)
                if not os.path.exists(path):
                    path = audio.source
            if path and os.path.exists(path):
                audio_idx = input_map[os.path.abspath(path).replace("\\", "/")]
                a_label = f"a_ext_{i}"
                
                # Trim external audio
                atrim_filter = f"atrim=start={audio.start_in_audio}:end={audio.end_in_audio},asetpts=PTS-STARTPTS"
                
                # Add delay matching its timeline start
                delay_ms = int(audio.timeline_start * 1000)
                delay_filter = f"adelay={delay_ms}|{delay_ms}"
                
                filter_complex_parts.append(
                    f"[{audio_idx}:a]{atrim_filter},{delay_filter},volume={audio.volume}[{a_label}]"
                )
                mixed_audio_sources.append(f"[{a_label}]")

        # Mix audio inputs together
        filter_complex_parts.append(
            f"{''.join(mixed_audio_sources)}amix=inputs={len(mixed_audio_sources)}:duration=first[outa]"
        )

        filter_graph = ";".join(filter_complex_parts)

        # 3. Assemble command list
        cmd = ["ffmpeg", "-y", "-nostdin"]
        for ip in inputs:
            cmd.extend(["-i", ip])

        cmd.extend([
            "-filter_complex", filter_graph,
            "-map", "[outv]",
            "-map", "[outa]",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            output_path
        ])

        return cmd
