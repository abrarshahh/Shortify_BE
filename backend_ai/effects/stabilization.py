import os
import subprocess
import logging
import tempfile

logger = logging.getLogger("agents.editor.stabilization")

def stabilize_clip(
    clip_path: str,
    start: float,
    end: float,
    temp_dir: str,
    strength: float = 0.5
) -> str:
    """
    Runs the first-pass analysis of FFmpeg's vidstabdetect filter on the trimmed segment.
    Outputs a unique .trf file in temp_dir and returns its absolute path.
    """
    os.makedirs(temp_dir, exist_ok=True)
    
    # Generate a unique path for the .trf output
    clip_basename = os.path.basename(clip_path)
    trf_filename = f"stabilize_{os.path.splitext(clip_basename)[0]}_{start:.2f}_{end:.2f}.trf"
    trf_path = os.path.join(temp_dir, trf_filename)
    
    # Map strength (0.0 - 1.0) to shakiness (1 - 10)
    shakiness = max(1, min(10, int(strength * 10)))
    accuracy = 15  # Default accuracy for detection
    
    # Use relative path to avoid Windows drive letter colon parser issues in FFmpeg
    rel_trf_path = os.path.relpath(trf_path).replace("\\", "/")
    
    # Pass 1 FFmpeg command
    cmd = [
        "ffmpeg", "-y", "-nostdin",
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-i", clip_path,
        "-vf", f"fps=fps=30,vidstabdetect=shakiness={shakiness}:accuracy={accuracy}:result={rel_trf_path}",
        "-f", "null",
        "-"
    ]
    
    logger.info(f"Running stabilization analysis pass: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, stdin=subprocess.DEVNULL, timeout=180)
    except subprocess.TimeoutExpired as e:
        logger.error(f"Stabilization analysis timed out after 180 seconds for {clip_path}")
        raise RuntimeError(f"FFmpeg stabilization analysis timed out: {e}")
        
    if result.returncode != 0:
        logger.error(f"Stabilization analysis failed: {result.stderr}")
        raise RuntimeError(f"FFmpeg stabilization analysis failed: {result.stderr}")
        
    logger.info(f"Stabilization analysis success: {rel_trf_path}")
    return rel_trf_path

