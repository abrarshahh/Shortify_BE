import os
import cv2
import numpy as np
import subprocess
import tempfile
import logging

logger = logging.getLogger("agents.editor")


def split_horizontal_mask(t: float, duration: float, h: int, w: int, params: dict) -> np.ndarray:
    progress = min(1.0, max(0.0, t / duration))
    mask = np.zeros((h, w), dtype=np.float32)
    split_w = int(w * progress / 2.0)
    if split_w > 0:
        mask[:, :split_w] = 1.0
        mask[:, w - split_w:] = 1.0
    return mask


def split_vertical_mask(t: float, duration: float, h: int, w: int, params: dict) -> np.ndarray:
    progress = min(1.0, max(0.0, t / duration))
    mask = np.zeros((h, w), dtype=np.float32)
    split_h = int(h * progress / 2.0)
    if split_h > 0:
        mask[:split_h, :] = 1.0
        mask[h - split_h:, :] = 1.0
    return mask


def wipe_left_mask(t: float, duration: float, h: int, w: int, params: dict) -> np.ndarray:
    progress = min(1.0, max(0.0, t / duration))
    mask = np.zeros((h, w), dtype=np.float32)
    wipe_x = int(w * (1.0 - progress))
    mask[:, wipe_x:] = 1.0
    return mask


def wipe_right_mask(t: float, duration: float, h: int, w: int, params: dict) -> np.ndarray:
    progress = min(1.0, max(0.0, t / duration))
    mask = np.zeros((h, w), dtype=np.float32)
    wipe_x = int(w * progress)
    mask[:, :wipe_x] = 1.0
    return mask


def wipe_up_mask(t: float, duration: float, h: int, w: int, params: dict) -> np.ndarray:
    progress = min(1.0, max(0.0, t / duration))
    mask = np.zeros((h, w), dtype=np.float32)
    wipe_y = int(h * (1.0 - progress))
    mask[wipe_y:, :] = 1.0
    return mask


def wipe_down_mask(t: float, duration: float, h: int, w: int, params: dict) -> np.ndarray:
    progress = min(1.0, max(0.0, t / duration))
    mask = np.zeros((h, w), dtype=np.float32)
    wipe_y = int(h * progress)
    mask[:wipe_y, :] = 1.0
    return mask


def wipe_diagonal_tl_mask(t: float, duration: float, h: int, w: int, params: dict) -> np.ndarray:
    progress = min(1.0, max(0.0, t / duration))
    mask = np.zeros((h, w), dtype=np.float32)
    y, x = np.ogrid[:h, :w]
    mask[(x / w) + (y / h) <= progress * 2.0] = 1.0
    return mask


def wipe_diagonal_tr_mask(t: float, duration: float, h: int, w: int, params: dict) -> np.ndarray:
    progress = min(1.0, max(0.0, t / duration))
    mask = np.zeros((h, w), dtype=np.float32)
    y, x = np.ogrid[:h, :w]
    mask[(1.0 - x / w) + (y / h) <= progress * 2.0] = 1.0
    return mask


def wipe_diagonal_bl_mask(t: float, duration: float, h: int, w: int, params: dict) -> np.ndarray:
    progress = min(1.0, max(0.0, t / duration))
    mask = np.zeros((h, w), dtype=np.float32)
    y, x = np.ogrid[:h, :w]
    mask[(x / w) + (1.0 - y / h) <= progress * 2.0] = 1.0
    return mask


def wipe_diagonal_br_mask(t: float, duration: float, h: int, w: int, params: dict) -> np.ndarray:
    progress = min(1.0, max(0.0, t / duration))
    mask = np.zeros((h, w), dtype=np.float32)
    y, x = np.ogrid[:h, :w]
    mask[(1.0 - x / w) + (1.0 - y / h) <= progress * 2.0] = 1.0
    return mask


def iris_circle_mask(t: float, duration: float, h: int, w: int, params: dict) -> np.ndarray:
    progress = min(1.0, max(0.0, t / duration))
    mask = np.zeros((h, w), dtype=np.float32)
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h / 2.0, w / 2.0
    dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)
    mask[dist_from_center <= max_dist * progress] = 1.0
    return mask


def diamond_mask(t: float, duration: float, h: int, w: int, params: dict) -> np.ndarray:
    progress = min(1.0, max(0.0, t / duration))
    mask = np.zeros((h, w), dtype=np.float32)
    y, x = np.ogrid[:h, :w]
    cx, cy = w / 2.0, h / 2.0
    dist = np.abs(x - cx) / cx + np.abs(y - cy) / cy
    mask[dist <= progress * 2.0] = 1.0
    return mask


def heart_mask(t: float, duration: float, h: int, w: int, params: dict) -> np.ndarray:
    progress = min(1.0, max(0.0, t / duration))
    mask = np.zeros((h, w), dtype=np.float32)
    y, x = np.ogrid[:h, :w]
    cx, cy = w / 2.0, h / 2.5
    # Normalize coordinates
    nx = (x - cx) / (w * 0.4)
    ny = (cy - y) / (h * 0.4)
    dist = nx**2 + (ny - np.sqrt(np.abs(nx)))**2
    mask[dist <= progress * 1.5] = 1.0
    return mask


def blinds_horizontal_mask(t: float, duration: float, h: int, w: int, params: dict) -> np.ndarray:
    progress = min(1.0, max(0.0, t / duration))
    mask = np.zeros((h, w), dtype=np.float32)
    num_bars = int(params.get("num_bars", 10))
    bar_w = w / max(1, num_bars)
    progress_w = bar_w * progress
    x = np.arange(w)
    mask[:, (x % bar_w) <= progress_w] = 1.0
    return mask


def blinds_vertical_mask(t: float, duration: float, h: int, w: int, params: dict) -> np.ndarray:
    progress = min(1.0, max(0.0, t / duration))
    mask = np.zeros((h, w), dtype=np.float32)
    num_bars = int(params.get("num_bars", 10))
    bar_h = h / max(1, num_bars)
    progress_h = bar_h * progress
    y = np.arange(h)
    mask[(y % bar_h) <= progress_h, :] = 1.0
    return mask


def checkerboard_mask(t: float, duration: float, h: int, w: int, params: dict) -> np.ndarray:
    progress = min(1.0, max(0.0, t / duration))
    mask = np.zeros((h, w), dtype=np.float32)
    grid_size = int(params.get("grid_size", 8))
    cell_w = w / max(1, grid_size)
    cell_h = h / max(1, grid_size)
    y, x = np.ogrid[:h, :w]
    
    np.random.seed(42)
    triggers = np.random.rand(grid_size, grid_size)
    cell_x = np.minimum(grid_size - 1, (x // cell_w).astype(int))
    cell_y = np.minimum(grid_size - 1, (y // cell_h).astype(int))
    pixel_triggers = triggers[cell_y, cell_x]
    mask[progress >= pixel_triggers] = 1.0
    return mask


def clock_wipe_mask(t: float, duration: float, h: int, w: int, params: dict) -> np.ndarray:
    progress = min(1.0, max(0.0, t / duration))
    mask = np.zeros((h, w), dtype=np.float32)
    y, x = np.ogrid[:h, :w]
    angle = np.arctan2(y - h / 2.0, x - w / 2.0)
    angle = (angle + np.pi) / (2.0 * np.pi)
    mask[angle <= progress] = 1.0
    return mask


def make_mask(transition_name: str, duration: float, frame_size: tuple, params: dict):
    h, w = frame_size
    name = transition_name.lower()
    
    # Custom parameter overrides
    if params is None:
        params = {}
        
    if name in ("split_horizontal", "split_reveal_horizontal"):
        return lambda t: split_horizontal_mask(t, duration, h, w, params)
    elif name in ("split_vertical", "split_reveal_vertical"):
        return lambda t: split_vertical_mask(t, duration, h, w, params)
    elif name == "wipe_left":
        return lambda t: wipe_left_mask(t, duration, h, w, params)
    elif name == "wipe_right":
        return lambda t: wipe_right_mask(t, duration, h, w, params)
    elif name == "wipe_up":
        return lambda t: wipe_up_mask(t, duration, h, w, params)
    elif name == "wipe_down":
        return lambda t: wipe_down_mask(t, duration, h, w, params)
    elif name in ("wipe_diagonal_tl", "wipe_tl"):
        return lambda t: wipe_diagonal_tl_mask(t, duration, h, w, params)
    elif name in ("wipe_diagonal_tr", "wipe_tr"):
        return lambda t: wipe_diagonal_tr_mask(t, duration, h, w, params)
    elif name in ("wipe_diagonal_bl", "wipe_bl"):
        return lambda t: wipe_diagonal_bl_mask(t, duration, h, w, params)
    elif name in ("wipe_diagonal_br", "wipe_br"):
        return lambda t: wipe_diagonal_br_mask(t, duration, h, w, params)
    elif name in ("iris", "iris_circle"):
        return lambda t: iris_circle_mask(t, duration, h, w, params)
    elif name == "diamond":
        return lambda t: diamond_mask(t, duration, h, w, params)
    elif name == "heart":
        return lambda t: heart_mask(t, duration, h, w, params)
    elif name == "blinds_horizontal":
        return lambda t: blinds_horizontal_mask(t, duration, h, w, params)
    elif name == "blinds_vertical":
        return lambda t: blinds_vertical_mask(t, duration, h, w, params)
    elif name == "checkerboard":
        return lambda t: checkerboard_mask(t, duration, h, w, params)
    elif name == "clock_wipe":
        return lambda t: clock_wipe_mask(t, duration, h, w, params)
    else:
        return lambda t: np.full((h, w), min(1.0, max(0.0, t / duration)), dtype=np.float32)


def slide_frames(frame_a: np.ndarray, frame_b: np.ndarray, direction: str, progress: float, params: dict) -> np.ndarray:
    h, w, c = frame_a.shape
    out = np.zeros_like(frame_a)
    progress = min(1.0, max(0.0, progress))
    
    direction = direction.lower()
    if "left" in direction or direction == "slide_push":
        shift = int(w * progress)
        if shift > 0:
            out[:, :w - shift] = frame_a[:, shift:]
            out[:, w - shift:] = frame_b[:, :shift]
        else:
            out[:] = frame_a
    elif "right" in direction:
        shift = int(w * progress)
        if shift > 0:
            out[:, :shift] = frame_b[:, w - shift:]
            out[:, shift:] = frame_a[:, :w - shift]
        else:
            out[:] = frame_a
    elif "up" in direction:
        shift = int(h * progress)
        if shift > 0:
            out[:h - shift, :] = frame_a[shift:, :]
            out[h - shift:, :] = frame_b[:shift, :]
        else:
            out[:] = frame_a
    elif "down" in direction:
        shift = int(h * progress)
        if shift > 0:
            out[:shift, :] = frame_b[h - shift:, :]
            out[shift:, :] = frame_a[:h - shift]
        else:
            out[:] = frame_a
    else:
        out = (frame_b * progress + frame_a * (1.0 - progress)).astype(np.uint8)
        
    return out


def glitch_frames(frame_a: np.ndarray, frame_b: np.ndarray, progress: float, params: dict) -> np.ndarray:
    h, w, c = frame_a.shape
    intensity = float(params.get("intensity", 1.0))
    glitch_progress = 1.0 - 2.0 * abs(progress - 0.5)
    frame = frame_b if progress >= 0.5 else frame_a
    frame = frame.copy()
    
    shift = int(20 * glitch_progress * intensity)
    if shift > 0:
        g = frame[:, :, 1]
        b = frame[:, :, 0]
        r = frame[:, :, 2]
        r_shifted = np.roll(r, -shift, axis=1)
        b_shifted = np.roll(b, shift, axis=1)
        frame = cv2.merge([b_shifted, g, r_shifted])
        
    num_jitters = int(10 * glitch_progress * intensity)
    for _ in range(num_jitters):
        y = np.random.randint(0, h)
        h_jitter = np.random.randint(5, 30)
        shift_x = np.random.randint(-30, 30)
        y_end = min(h, y + h_jitter)
        frame[y:y_end, :] = np.roll(frame[y:y_end, :], shift_x, axis=1)
        
    return frame


def pixelate_frames(frame_a: np.ndarray, frame_b: np.ndarray, progress: float, params: dict) -> np.ndarray:
    h, w, c = frame_a.shape
    max_cell_size = int(params.get("max_cell_size", 32))
    cell_progress = 1.0 - 2.0 * abs(progress - 0.5)
    cell_size = int(1 + (max_cell_size - 1) * cell_progress)
    frame = frame_b if progress >= 0.5 else frame_a
    
    if cell_size > 1:
        temp = cv2.resize(frame, (max(1, w // cell_size), max(1, h // cell_size)), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
    return frame


def blur_frames(frame_a: np.ndarray, frame_b: np.ndarray, progress: float, params: dict) -> np.ndarray:
    h, w, c = frame_a.shape
    max_blur_size = int(params.get("max_blur_size", 51))
    blur_progress = 1.0 - 2.0 * abs(progress - 0.5)
    ksize = int(max_blur_size * blur_progress)
    if ksize % 2 == 0:
        ksize = max(1, ksize - 1)
    ksize = max(1, ksize | 1)
    frame = frame_b if progress >= 0.5 else frame_a
    
    if ksize > 1:
        return cv2.GaussianBlur(frame, (ksize, ksize), 0)
    return frame


def light_leak_frames(frame_a: np.ndarray, frame_b: np.ndarray, progress: float, params: dict) -> np.ndarray:
    intensity_multiplier = float(params.get("intensity", 0.8))
    flash_progress = 1.0 - 2.0 * abs(progress - 0.5)
    color_name = str(params.get("color", "white")).lower()
    
    blended = (frame_b.astype(np.float32) * progress + frame_a.astype(np.float32) * (1.0 - progress)).astype(np.uint8)
    
    if color_name == "orange":
        flash_overlay = np.zeros_like(blended)
        flash_overlay[:, :, 0] = 50
        flash_overlay[:, :, 1] = 150
        flash_overlay[:, :, 2] = 255
    elif color_name == "red":
        flash_overlay = np.zeros_like(blended)
        flash_overlay[:, :, 2] = 255
    elif color_name == "blue":
        flash_overlay = np.zeros_like(blended)
        flash_overlay[:, :, 0] = 255
    else:
        flash_overlay = np.full_like(blended, 255)
        
    alpha = flash_progress * intensity_multiplier
    return cv2.addWeighted(blended, 1.0 - alpha, flash_overlay, alpha, 0)


def spin_frames(frame_a: np.ndarray, frame_b: np.ndarray, progress: float, params: dict) -> np.ndarray:
    h, w, c = frame_a.shape
    angle_delta = float(params.get("angle_delta", 360.0))
    zoom_scale = float(params.get("zoom_scale", 0.3))
    angle = progress * angle_delta
    frame = frame_b if progress >= 0.5 else frame_a
    
    center = (w / 2.0, h / 2.0)
    scale = 1.0 - zoom_scale * np.sin(progress * np.pi)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(frame, matrix, (w, h))


def ripple_frames(frame_a: np.ndarray, frame_b: np.ndarray, progress: float, params: dict) -> np.ndarray:
    h, w, c = frame_a.shape
    freq = float(params.get("wave_frequency", 10.0))
    amp = float(params.get("wave_amplitude", 30.0))
    intensity = np.sin(progress * np.pi) * amp
    
    y, x = np.mgrid[:h, :w].astype(np.float32)
    dist_x = np.sin(y / freq) * intensity
    dist_y = np.cos(x / freq) * intensity
    map_x = np.clip(x + dist_x, 0, w - 1)
    map_y = np.clip(y + dist_y, 0, h - 1)
    frame = frame_b if progress >= 0.5 else frame_a
    return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)


def check_has_audio_stream(file_path: str) -> bool:
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
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


def get_video_duration(file_path: str) -> float:
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                file_path
            ],
            capture_output=True,
            text=True,
            check=True,
            stdin=subprocess.DEVNULL
        )
        return float(result.stdout.strip())
    except Exception:
        cap = cv2.VideoCapture(file_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            return frames / fps
        return 0.0


def create_spatial_transition(
    clip_a_path: str,
    clip_b_path: str,
    transition_name: str,
    duration: float,
    output_path: str,
    fps: int = 30,
    frame_size: tuple = (1080, 1920),
    transition_params: dict = None
) -> None:
    h, w = frame_size
    if transition_params is None:
        transition_params = {}
        
    cap_a = cv2.VideoCapture(clip_a_path)
    cap_b = cv2.VideoCapture(clip_b_path)
    
    if not cap_a.isOpened() or not cap_b.isOpened():
        raise RuntimeError("Failed to open video sources for spatial transition")
        
    total_frames_a = int(cap_a.get(cv2.CAP_PROP_FRAME_COUNT))
    
    num_frames = int(round(duration * fps))
    if num_frames <= 0:
        num_frames = 1
        
    start_frame_a = max(0, total_frames_a - num_frames)
    cap_a.set(cv2.CAP_PROP_POS_FRAMES, start_frame_a)
    cap_b.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    temp_silent_video = tempfile.mktemp(suffix="_silent.mp4", dir=os.path.dirname(output_path))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_silent_video, fourcc, fps, (w, h))
    
    name = transition_name.lower()
    
    try:
        is_mask = name in (
            "split_horizontal", "split_reveal_horizontal", "split_vertical", "split_reveal_vertical",
            "wipe_left", "wipe_right", "wipe_up", "wipe_down",
            "wipe_diagonal_tl", "wipe_diagonal_tr", "wipe_diagonal_bl", "wipe_diagonal_br",
            "iris", "iris_circle", "diamond", "heart", 
            "blinds_horizontal", "blinds_vertical", "checkerboard", "clock_wipe"
        )
        is_slide = name in ("slide_left", "slide_right", "slide_up", "slide_down", "slide_push")
        
        if is_mask:
            mask_func = make_mask(transition_name, duration, frame_size, transition_params)
            
        for f_idx in range(num_frames):
            ret_a, frame_a = cap_a.read()
            ret_b, frame_b = cap_b.read()
            
            if not ret_a:
                frame_a = np.zeros((h, w, 3), dtype=np.uint8)
            else:
                if frame_a.shape[:2] != (h, w):
                    frame_a = cv2.resize(frame_a, (w, h))
                    
            if not ret_b:
                frame_b = np.zeros((h, w, 3), dtype=np.uint8)
            else:
                if frame_b.shape[:2] != (h, w):
                    frame_b = cv2.resize(frame_b, (w, h))
                    
            t = f_idx / fps
            progress = f_idx / (num_frames - 1) if num_frames > 1 else 1.0
            
            if is_slide:
                blended = slide_frames(frame_a, frame_b, transition_name, progress, transition_params)
            elif is_mask:
                mask = mask_func(t)
                mask_3d = np.expand_dims(mask, axis=2)
                blended = (frame_b.astype(np.float32) * mask_3d + frame_a.astype(np.float32) * (1.0 - mask_3d)).astype(np.uint8)
            elif name == "glitch":
                blended = glitch_frames(frame_a, frame_b, progress, transition_params)
            elif name == "pixelate":
                blended = pixelate_frames(frame_a, frame_b, progress, transition_params)
            elif name == "blur":
                blended = blur_frames(frame_a, frame_b, progress, transition_params)
            elif name == "light_leak":
                blended = light_leak_frames(frame_a, frame_b, progress, transition_params)
            elif name == "spin":
                blended = spin_frames(frame_a, frame_b, progress, transition_params)
            elif name == "ripple":
                blended = ripple_frames(frame_a, frame_b, progress, transition_params)
            else:
                # Default crossfade
                blended = (frame_b.astype(np.float32) * progress + frame_a.astype(np.float32) * (1.0 - progress)).astype(np.uint8)
                
            out.write(blended)
    finally:
        cap_a.release()
        cap_b.release()
        out.release()
        
    has_audio_a = check_has_audio_stream(clip_a_path)
    has_audio_b = check_has_audio_stream(clip_b_path)
    
    temp_audio_a = tempfile.mktemp(suffix="_a.aac", dir=os.path.dirname(output_path))
    temp_audio_b = tempfile.mktemp(suffix="_b.aac", dir=os.path.dirname(output_path))
    temp_audio_cross = tempfile.mktemp(suffix="_cross.aac", dir=os.path.dirname(output_path))
    
    try:
        if has_audio_a:
            dur_a = get_video_duration(clip_a_path)
            start_a = max(0.0, dur_a - duration)
            cmd_a = [
                "ffmpeg", "-y", "-nostdin", "-i", clip_a_path,
                "-ss", f"{start_a:.3f}", "-t", f"{duration:.3f}",
                "-vn", "-c:a", "aac", temp_audio_a
            ]
            subprocess.run(cmd_a, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, stdin=subprocess.DEVNULL)
            
        if has_audio_b:
            cmd_b = [
                "ffmpeg", "-y", "-nostdin", "-i", clip_b_path,
                "-t", f"{duration:.3f}",
                "-vn", "-c:a", "aac", temp_audio_b
            ]
            subprocess.run(cmd_b, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, stdin=subprocess.DEVNULL)
            
        if has_audio_a and has_audio_b:
            cmd_cross = [
                "ffmpeg", "-y", "-nostdin", "-i", temp_audio_a, "-i", temp_audio_b,
                "-filter_complex", f"acrossfade=d={duration:.3f}",
                "-c:a", "aac", temp_audio_cross
            ]
            subprocess.run(cmd_cross, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, stdin=subprocess.DEVNULL)
            audio_input = ["-i", temp_audio_cross]
        elif has_audio_a:
            cmd_fade = [
                "ffmpeg", "-y", "-nostdin", "-i", temp_audio_a,
                "-filter_complex", f"afade=t=out=st=0=d={duration:.3f}",
                "-c:a", "aac", temp_audio_cross
            ]
            subprocess.run(cmd_fade, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, stdin=subprocess.DEVNULL)
            audio_input = ["-i", temp_audio_cross]
        elif has_audio_b:
            cmd_fade = [
                "ffmpeg", "-y", "-nostdin", "-i", temp_audio_b,
                "-filter_complex", f"afade=t=in=st=0=d={duration:.3f}",
                "-c:a", "aac", temp_audio_cross
            ]
            subprocess.run(cmd_fade, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, stdin=subprocess.DEVNULL)
            audio_input = ["-i", temp_audio_cross]
        else:
            audio_input = []
            
        cmd_merge = ["ffmpeg", "-y", "-nostdin", "-i", temp_silent_video]
        if audio_input:
            cmd_merge.extend(audio_input)
            cmd_merge.extend([
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0",
                output_path
            ])
        else:
            cmd_merge.extend([
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                output_path
            ])
            
        subprocess.run(cmd_merge, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, stdin=subprocess.DEVNULL)
        
    except Exception as e:
        logger.error(f"Error processing spatial transition: {e}")
        cmd_fallback = [
            "ffmpeg", "-y", "-nostdin", "-i", temp_silent_video,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            output_path
        ]
        subprocess.run(cmd_fallback, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL)
    finally:
        for p in (temp_silent_video, temp_audio_a, temp_audio_b, temp_audio_cross):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass
