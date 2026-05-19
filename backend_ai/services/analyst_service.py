import os
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Tuple
from moviepy.video.io.VideoFileClip import VideoFileClip

from backend_ai.core.config_loader import AGENTS_CONFIG

class ProjectAnalystAgent:
    """
    Phase 4: Performs a local, non-API pre-flight sanity check and quality score
    on all incoming video paths. Reorders paths so the highest-quality clips
    are positioned at the front for hook selection.
    """

    SUPPORTED_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    SUPPORTED_PHOTO_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

    def __init__(self):
        config = AGENTS_CONFIG.get("project_analyst", {})
        self.min_video_duration = float(config.get("min_video_duration", 1.0))
        self.frame_sample_count = int(config.get("frame_sample_count", 5))
        self.sharpness_divisor = float(config.get("sharpness_divisor", 500.0))

    def analyze_inputs(self, video_paths: List[str]) -> Dict[str, Any]:
        """
        Main entry point for pre-flight check.
        Validates, grades, and reorders input files.
        """
        print(f"ProjectAnalystAgent: Starting analysis for {len(video_paths)} input files...")
        
        valid_media: List[Dict[str, Any]] = []
        rejected_files: List[Dict[str, Any]] = []
        
        for path in video_paths:
            # 1. Validation Checks
            if not os.path.exists(path):
                print(f"  Warning: File does not exist on disk: {path}")
                rejected_files.append({"path": path, "reason": "file_not_found"})
                continue
                
            file_size = os.path.getsize(path)
            if file_size == 0:
                print(f"  Warning: File is empty (0 bytes): {path}")
                rejected_files.append({"path": path, "reason": "empty_file"})
                continue
                
            ext = os.path.splitext(path)[1].lower()
            
            # Classify media type
            if ext in self.SUPPORTED_VIDEO_EXTS:
                media_type = "video"
            elif ext in self.SUPPORTED_PHOTO_EXTS:
                media_type = "photo"
            else:
                print(f"  Warning: Unsupported file extension '{ext}': {path}")
                rejected_files.append({"path": path, "reason": "unsupported_extension"})
                continue

            # Duration check for videos
            duration = 0.0
            if media_type == "video":
                try:
                    with VideoFileClip(path) as clip:
                        duration = clip.duration
                    
                    if duration < self.min_video_duration:
                        print(f"  Warning: Video clip is too short ({duration:.2f}s): {path}")
                        rejected_files.append({
                            "path": path,
                            "reason": "file_too_short",
                            "duration": duration
                        })
                        continue
                except Exception as e:
                    print(f"  Warning: MoviePy could not read video duration for {path}. Error: {e}")
                    rejected_files.append({"path": path, "reason": "unreadable_video"})
                    continue

            # 2. Quality Scoring
            quality_score = 0.0
            avg_sharpness = 0.0
            avg_brightness = 0.0
            
            try:
                if media_type == "video":
                    quality_score, avg_sharpness, avg_brightness = self._score_video(path)
                else:
                    quality_score, avg_sharpness, avg_brightness = self._score_photo(path)
            except Exception as e:
                print(f"  Warning: Local quality scoring failed for {path}, assigning fallback 0.5. Error: {e}")
                quality_score = 0.5

            valid_media.append({
                "path": path,
                "media_type": media_type,
                "duration": duration,
                "quality_score": round(quality_score, 4),
                "avg_sharpness": round(avg_sharpness, 2),
                "avg_brightness": round(avg_brightness, 2),
            })

        # 3. Sort by Quality Score descending
        valid_media.sort(key=lambda x: x["quality_score"], reverse=True)
        
        recommended_order = [item["path"] for item in valid_media]
        recommended_hook = recommended_order[0] if recommended_order else ""

        report = {
            "total_input_files": len(video_paths),
            "valid_files": len(valid_media),
            "rejected_files": rejected_files,
            "media": valid_media,
            "recommended_hook": recommended_hook,
            "recommended_order": recommended_order
        }
        
        print(f"ProjectAnalystAgent: Analysis complete. Valid: {len(valid_media)}/{len(video_paths)}.")
        if recommended_hook:
            print(f"  Recommended Hook: {recommended_hook} (Score: {valid_media[0]['quality_score']})")
            
        return report

    def _score_video(self, path: str) -> Tuple[float, float, float]:
        """
        Scores video quality by sampling frames and evaluating sharpness
        (Laplacian variance) and brightness balance.
        """
        sharpness_scores = []
        brightness_scores = []
        
        with VideoFileClip(path) as clip:
            duration = clip.duration
            # Compute sampling timestamps spaced evenly across video duration
            timestamps = np.linspace(0.1, max(0.1, duration - 0.1), self.frame_sample_count)
            
            for t in timestamps:
                try:
                    frame = clip.get_frame(t)
                    # frame is HxWx3 numpy array
                    
                    # 1. Brightness
                    gray = frame.mean(axis=2)
                    avg_bright = gray.mean()
                    brightness_scores.append(avg_bright)
                    
                    # 2. Sharpness via Pure NumPy Laplacian convolution
                    # Laplacian kernel:
                    # [ 0,  1,  0]
                    # [ 1, -4,  1]
                    # [ 0,  1,  0]
                    laplacian = (
                        gray[:-2, 1:-1] + gray[2:, 1:-1] +
                        gray[1:-1, :-2] + gray[1:-1, 2:] -
                        4 * gray[1:-1, 1:-1]
                    )
                    sharpness = laplacian.var()
                    sharpness_scores.append(sharpness)
                except Exception as e:
                    # Suppress single frame reading errors
                    continue

        if not sharpness_scores:
            return 0.5, 0.0, 128.0

        avg_sharpness = float(np.mean(sharpness_scores))
        avg_brightness = float(np.mean(brightness_scores))
        
        # Brightness balance penalty: 128 is perfect neutral.
        # Max penalty is 1.0 at pure black (0) or pure white (255).
        brightness_penalty = abs(avg_brightness - 128.0) / 128.0
        
        # Calculate final 0.0 to 1.0 quality score
        quality_score = min(1.0, (avg_sharpness / self.sharpness_divisor) * (1.0 - brightness_penalty * 0.3))
        
        return quality_score, avg_sharpness, avg_brightness

    def _score_photo(self, path: str) -> Tuple[float, float, float]:
        """
        Scores photo quality evaluating sharpness and brightness balance.
        Uses PIL to open and convert image.
        """
        try:
            with Image.open(path) as img:
                # Convert to grayscale
                img_gray = img.convert("L")
                
                # Check brightness
                gray_arr = np.array(img_gray, dtype=np.float32)
                avg_brightness = float(gray_arr.mean())
                
                # To keep sharpness fast and consistent, resize to a standardized smaller thumbnail
                img_small = img_gray.resize((480, 270), Image.Resampling.BILINEAR)
                small_arr = np.array(img_small, dtype=np.float32)
                
                # Compute Laplacian variance
                laplacian = (
                    small_arr[:-2, 1:-1] + small_arr[2:, 1:-1] +
                    small_arr[1:-1, :-2] + small_arr[1:-1, 2:] -
                    4 * small_arr[1:-1, 1:-1]
                )
                sharpness = float(laplacian.var())
                
                brightness_penalty = abs(avg_brightness - 128.0) / 128.0
                quality_score = min(1.0, (sharpness / self.sharpness_divisor) * (1.0 - brightness_penalty * 0.3))
                
                return quality_score, sharpness, avg_brightness
        except Exception:
            return 0.5, 0.0, 128.0


if __name__ == "__main__":
    # Smoke test
    agent = ProjectAnalystAgent()
    print("ProjectAnalystAgent initialized successfully.")
