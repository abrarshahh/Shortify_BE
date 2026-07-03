from typing import Tuple

def build_ffmpeg_overlay_filters(
    duration: float,
    anim_type: str,
    x_resting: str,
    y_resting: str,
    anim_duration: float = 0.5
) -> Tuple[str, str, str]:
    """
    Constructs the video filter (e.g. fade) and overlay coordinate expressions for entrance/exit animations.
    Returns (video_filter_str, overlay_x_expr, overlay_y_expr).
    """
    video_filters = []
    
    # 1. Handle Opacity/Fade Animation
    if anim_type == "fade":
        fade_dur = min(anim_duration, duration / 2)
        if fade_dur > 0:
            video_filters.append(f"fade=in:st=0:d={fade_dur:.3f}")
            video_filters.append(f"fade=out:st={duration - fade_dur:.3f}:d={fade_dur:.3f}")

    video_str = ",".join(video_filters) if video_filters else "copy"

    # 2. Handle Position/Translation Slide Animation (with entrance and exit slides)
    x_expr = x_resting
    y_expr = y_resting
    
    slide_dur = min(anim_duration, duration / 2)
    if slide_dur > 0:
        if anim_type == "slide_up":
            # Entrance: slide up from bottom H. Exit: slide down off bottom H.
            y_expr = (
                f"if(lt(t,{slide_dur:.3f}),H-(H-({y_resting}))*(t/{slide_dur:.3f}),"
                f"if(gt(t,{duration - slide_dur:.3f}),({y_resting})+(H-({y_resting}))*((t-{duration - slide_dur:.3f})/{slide_dur:.3f}),"
                f"{y_resting}))"
            )
        elif anim_type == "slide_down":
            # Entrance: slide down from top -h. Exit: slide up off top -h.
            y_expr = (
                f"if(lt(t,{slide_dur:.3f}),-h+(({y_resting})+h)*(t/{slide_dur:.3f}),"
                f"if(gt(t,{duration - slide_dur:.3f}),({y_resting})-(({y_resting})+h)*((t-{duration - slide_dur:.3f})/{slide_dur:.3f}),"
                f"{y_resting}))"
            )
        elif anim_type == "slide_left":
            # Entrance: slide left from side -w. Exit: slide left off side -w.
            x_expr = (
                f"if(lt(t,{slide_dur:.3f}),-w+(({x_resting})+w)*(t/{slide_dur:.3f}),"
                f"if(gt(t,{duration - slide_dur:.3f}),({x_resting})-(({x_resting})+w)*((t-{duration - slide_dur:.3f})/{slide_dur:.3f}),"
                f"{x_resting}))"
            )
        elif anim_type == "slide_right":
            # Entrance: slide right from side W. Exit: slide right off side W.
            x_expr = (
                f"if(lt(t,{slide_dur:.3f}),W-(W-({x_resting}))*(t/{slide_dur:.3f}),"
                f"if(gt(t,{duration - slide_dur:.3f}),({x_resting})+(W-({x_resting}))*((t-{duration - slide_dur:.3f})/{slide_dur:.3f}),"
                f"{x_resting}))"
            )

    return video_str, x_expr, y_expr
