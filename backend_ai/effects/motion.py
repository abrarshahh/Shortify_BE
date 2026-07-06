import math
from typing import List, Tuple, Optional, Dict

# Named speed presets mapped to keyframes (time_fraction, speed_multiplier)
SPEED_PRESETS: Dict[str, List[Tuple[float, float]]] = {
    "constant_fast": [(0.0, 2.0), (1.0, 2.0)],
    "constant_slow": [(0.0, 0.5), (1.0, 0.5)],
    "ramp_up": [(0.0, 1.0), (0.7, 1.0), (1.0, 2.5)],
    "ramp_down": [(0.0, 2.5), (0.3, 1.0), (1.0, 1.0)],
    "speed_bump": [(0.0, 1.0), (0.4, 1.0), (0.5, 3.0), (0.6, 1.0), (1.0, 1.0)],
    "freeze_frame": [(0.0, 1.0), (0.49, 1.0), (0.50, 0.05), (0.51, 1.0), (1.0, 1.0)],
}

def get_average_speed_linear(keyframes: List[Tuple[float, float]]) -> float:
    """
    Computes the exact average speed multiplier of a piecewise linear speed curve.
    Uses calculus (integration of 1/S(t) over interval) to find the total output duration.
    Average speed = Input Duration / Output Duration.
    """
    if not keyframes or len(keyframes) < 2:
        return 1.0
    
    total_output_fraction = 0.0
    for i in range(len(keyframes) - 1):
        x1, s1 = keyframes[i]
        x2, s2 = keyframes[i+1]
        dx = x2 - x1
        if dx <= 0:
            continue
        
        # Clamp speed values slightly above zero to avoid log(0) or division by zero
        s1 = max(1e-4, s1)
        s2 = max(1e-4, s2)
        
        if abs(s2 - s1) < 1e-5:
            # Piecewise constant interval
            total_output_fraction += dx / s1
        else:
            # Piecewise linear interval: integral of 1/(s1 + a*x)
            # which is (dx / (s2 - s1)) * ln(s2 / s1)
            total_output_fraction += (dx / (s2 - s1)) * math.log(s2 / s1)
            
    if total_output_fraction <= 0:
        return 1.0
    return 1.0 / total_output_fraction


def build_ffmpeg_speed_filter(
    duration: float,
    keyframes: Optional[List[Tuple[float, float]]] = None,
    preset: Optional[str] = None,
    reverse: bool = False
) -> Tuple[str, str]:
    """
    Translates speed parameters to FFmpeg video setpts and audio atempo/areverse filters.
    Returns (video_filter_str, audio_filter_str).
    """
    # 1. Resolve keyframes from preset if not provided
    if not keyframes and preset:
        keyframes = SPEED_PRESETS.get(preset)

    if not keyframes or len(keyframes) < 2:
        # Default or no-op speed, check if reversed
        video_filters = ["setpts=PTS-STARTPTS"]
        audio_filters = []
        if reverse:
            video_filters.append("reverse")
            audio_filters.append("areverse")
        
        video_str = ",".join(video_filters)
        audio_str = ",".join(audio_filters) if audio_filters else "copy"
        return video_str, audio_str

    # 2. Build piecewise linear mapping for setpts
    # Let t_i = x_i * duration
    # Let tout_i = output time at input t_i
    t_points = []
    s_points = []
    tout_points = [0.0]

    for x, s in keyframes:
        t_points.append(x * duration)
        s_points.append(max(1e-4, s))

    # Compute tout_points using exact integrals
    for i in range(len(keyframes) - 1):
        t1, s1 = t_points[i], s_points[i]
        t2, s2 = t_points[i+1], s_points[i+1]
        dt = t2 - t1
        
        if dt <= 0:
            tout_points.append(tout_points[-1])
            continue
            
        if abs(s2 - s1) < 1e-5:
            tout = tout_points[-1] + dt / s1
        else:
            tout = tout_points[-1] + (dt / (s2 - s1)) * math.log(s2 / s1)
        tout_points.append(tout)

    # 3. Construct FFmpeg setpts expression recursively as a nested if
    # Output time = if(T < t_1, tout_0 + (T-t_0)/S_0, if(T < t_2, tout_1 + ...))
    # Note: T is input presentation time in seconds
    def build_expr(idx: int) -> str:
        if idx >= len(keyframes) - 1:
            # Last segment fallback
            t_start = t_points[-2]
            tout_start = tout_points[-2]
            s_start = s_points[-2]
            s_end = s_points[-1]
            if abs(s_end - s_start) < 1e-5:
                return f"{tout_start:.4f}+(T-{t_start:.4f})/{s_start:.4f}"
            else:
                a = (s_end - s_start) / (t_points[-1] - t_start)
                return f"{tout_start:.4f}+log(1.0+{a:.6f}*(T-{t_start:.4f})/{s_start:.4f})/{a:.6f}"
        
        t_start = t_points[idx]
        t_end = t_points[idx+1]
        tout_start = tout_points[idx]
        s_start = s_points[idx]
        s_end = s_points[idx+1]
        
        # Current segment expression
        if abs(s_end - s_start) < 1e-5:
            segment_expr = f"{tout_start:.4f}+(T-{t_start:.4f})/{s_start:.4f}"
        else:
            a = (s_end - s_start) / (t_end - t_start)
            segment_expr = f"{tout_start:.4f}+log(1.0+{a:.6f}*(T-{t_start:.4f})/{s_start:.4f})/{a:.6f}"
            
        return f"if(lt(T,{t_end:.4f}),{segment_expr},{build_expr(idx+1)})"

    setpts_expr = build_expr(0)
    
    # Format video filter
    video_filters = [f"setpts='({setpts_expr})/TB'"]
    if reverse:
        video_filters.append("reverse")

    # 4. Format audio filter using average speed
    avg_speed = get_average_speed_linear(keyframes)
    audio_filters = []
    
    # Chained atempo filters for average speed
    current = avg_speed
    while current > 2.0:
        audio_filters.append("atempo=2.0")
        current /= 2.0
    while current < 0.5:
        audio_filters.append("atempo=0.5")
        current /= 0.5
    if abs(current - 1.0) > 1e-4:
        audio_filters.append(f"atempo={current:.4f}")
        
    if reverse:
        audio_filters.append("areverse")
        
    video_str = ",".join(video_filters)
    audio_str = ",".join(audio_filters) if audio_filters else "copy"
    
    return video_str, audio_str
