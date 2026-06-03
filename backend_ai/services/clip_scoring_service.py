import os
import cv2
import numpy as np
import logging
from typing import Dict, Any, List
from moviepy.video.io.VideoFileClip import VideoFileClip

logger = logging.getLogger("agents.clip_scoring")


class ClipScoringAgent:
    """
    Local AI Agent that ranks video segments based on:
    1. Sharpness (Laplacian variance)
    2. Motion Analysis (Grayscale frame differencing to determine static vs. high-motion)
    3. Face Presence (using OpenCV built-in Haar Cascade Face Detection)
    """

    def __init__(self):
        pass

    def score_video_segment(self, file_path: str, start: float, end: float) -> Dict[str, Any]:
        """
        Analyzes a specific segment within a video file.
        Returns a dictionary containing metrics: sharpness, motion_score, motion_type, face_present, local_score.
        """
        if not os.path.exists(file_path):
            return {
                "sharpness": 0.0,
                "motion_score": 0.0,
                "motion_type": "static",
                "face_present": False,
                "local_score": 0.0
            }

        sharpness_list = []
        motion_diff_list = []
        face_detected = False

        try:
            with VideoFileClip(file_path) as clip:
                duration = clip.duration
                t_start = max(0.0, start)
                t_end = min(duration, end)
                
                # Sample 5 frames for sharpness and face presence
                sample_times = np.linspace(t_start, t_end, 5)
                
                # Sample frame pairs (spaced by 0.1s) for motion frame differencing
                motion_times = np.linspace(t_start, max(t_start, t_end - 0.15), 4)
                
                # Sharpness & Face detection loop
                for t in sample_times:
                    try:
                        frame_rgb = clip.get_frame(t)
                        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
                        
                        # Sharpness: Laplacian variance
                        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                        sharpness_list.append(float(laplacian.var()))
                        
                        # Face Presence using OpenCV Haar Cascade
                        if not face_detected:
                            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                            face_cascade = cv2.CascadeClassifier(cascade_path)
                            if not face_cascade.empty():
                                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                                if len(faces) > 0:
                                    face_detected = True
                    except Exception:
                        continue
                        
                # Motion analysis loop
                for t in motion_times:
                    try:
                        f1 = clip.get_frame(t)
                        f2 = clip.get_frame(t + 0.1)
                        
                        gray1 = cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY)
                        gray2 = cv2.cvtColor(f2, cv2.COLOR_RGB2GRAY)
                        
                        # Frame difference
                        diff = cv2.absdiff(gray1, gray2)
                        motion_diff_list.append(float(diff.mean()))
                    except Exception:
                        continue
        except Exception as e:
            logger.warning(f"Error during ClipScoring analysis of {file_path}: {e}")

        avg_sharpness = float(np.mean(sharpness_list)) if sharpness_list else 100.0
        avg_motion = float(np.mean(motion_diff_list)) if motion_diff_list else 0.5
        
        # Categorize Pacing: static vs high-motion
        # Threshold: if average absolute pixel diff is >= 2.0, it is high-motion
        motion_type = "high-motion" if avg_motion >= 2.0 else "static"
        
        # Compute combined score
        norm_sharpness = min(1.0, avg_sharpness / 500.0)
        face_bonus = 0.3 if face_detected else 0.0
        motion_factor = min(0.2, avg_motion / 20.0)
        
        combined_score = min(1.0, max(0.0, norm_sharpness + face_bonus + motion_factor))

        return {
            "sharpness": round(avg_sharpness, 2),
            "motion_score": round(avg_motion, 4),
            "motion_type": motion_type,
            "face_present": face_detected,
            "local_score": round(combined_score, 4)
        }

    def score_visual_data(self, visual_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enriches the Gemini media analyses (visual_data) with local ClipScoringAgent metrics
        for interesting_segments and all_segments.
        """
        logger.info("ClipScoringAgent: Scoring visual segments...")
        for analysis in visual_data:
            metadata = analysis.get("file_metadata", {})
            file_path = metadata.get("path")
            if not file_path or not os.path.exists(file_path):
                file_path = metadata.get("filename")
                
            is_image = metadata.get("media_type") == "photo"
            
            # Score interesting_segments
            for seg in analysis.get("interesting_segments", []):
                start = float(seg.get("start", 0.0))
                end = float(seg.get("end", 3.0))
                
                if is_image:
                    metrics = {
                        "sharpness": 200.0,
                        "motion_score": 0.0,
                        "motion_type": "static",
                        "face_present": False,
                        "local_score": 0.8
                    }
                else:
                    metrics = self.score_video_segment(file_path, start, end)
                    
                seg.update(metrics)

            # Score all_segments
            for seg in analysis.get("all_segments", []):
                start = float(seg.get("start", 0.0))
                end = float(seg.get("end", 3.0))
                
                if is_image:
                    metrics = {
                        "sharpness": 200.0,
                        "motion_score": 0.0,
                        "motion_type": "static",
                        "face_present": False,
                        "local_score": 0.8
                    }
                else:
                    metrics = self.score_video_segment(file_path, start, end)
                    
                seg.update(metrics)

        logger.info("ClipScoringAgent: Visual segments successfully scored!")
        return visual_data
