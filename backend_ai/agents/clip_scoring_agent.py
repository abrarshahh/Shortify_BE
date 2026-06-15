import os
import cv2
import numpy as np
import logging
import json
import hashlib
from typing import Dict, Any, List, Tuple

logger = logging.getLogger("agents.clip_scoring")

class ClipScoringAgent:
    """
    Task 25: Local AI Agent that scores and ranks video files locally.
    Evaluates sharpness, exposure, motion level, and face presence.
    Outputs a style-weighted composite score and motion tier.
    """

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".heic", ".heif"}

    def __init__(self, cache_dir: str = "data/cache/clip_scores"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, file_path: str) -> str:
        """Generates a cache file path based on file metadata."""
        stats = os.stat(file_path)
        fingerprint = f"{os.path.basename(file_path)}_{stats.st_size}_{stats.st_mtime}"
        cache_key = hashlib.md5(fingerprint.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{cache_key}.json")

    def score_file(self, file_path: str, style: str = "cinematic") -> Dict[str, Any]:
        """
        Calculates sharpness, exposure, motion, and face presence, returning
        a cached or calculated score dictionary.
        """
        if not os.path.exists(file_path):
            logger.warning(f"ClipScoringAgent: File not found: {file_path}")
            return self._default_score()

        cache_path = self._get_cache_path(file_path)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    cached = json.load(f)
                logger.info(f"ClipScoringAgent: Loaded cached scores for {os.path.basename(file_path)}")
                # Recalculate composite score in case style changed
                cached["composite_score"] = self._compute_composite_score(cached, style)
                return cached
            except Exception as e:
                logger.warning(f"ClipScoringAgent: Failed to load cache for {file_path}: {e}")

        # Determine type
        ext = os.path.splitext(file_path)[1].lower()
        if ext in self.IMAGE_EXTENSIONS:
            result = self._score_photo(file_path)
        else:
            result = self._score_video(file_path)

        # Cache results (without the style-dependent composite score)
        try:
            with open(cache_path, "w") as f:
                json.dump(result, f, indent=2)
        except Exception as e:
            logger.warning(f"ClipScoringAgent: Failed to write cache for {file_path}: {e}")

        # Append composite score
        result["composite_score"] = self._compute_composite_score(result, style)
        return result

    def _default_score(self) -> Dict[str, Any]:
        return {
            "sharpness": 100.0,
            "exposure_score": 0.8,
            "motion_score": 0.0,
            "motion_tier": "static",
            "face_detected": False,
            "face_anchor_x": 0.5,
            "composite_score": 0.5
        }

    def _score_photo(self, file_path: str) -> Dict[str, Any]:
        """Scores a static image file."""
        frame_bgr = cv2.imread(file_path)
        if frame_bgr is None:
            return self._default_score()

        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Sharpness
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = float(laplacian.var())

        # 2. Exposure
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        v_mean = float(hsv[:, :, 2].mean())
        if v_mean < 30 or v_mean > 220:
            exposure_score = 0.0
        else:
            exposure_score = 1.0 - (abs(v_mean - 125.0) / 95.0)

        # 3. Face Presence
        face_detected, face_anchor_x = self._detect_faces_mediapipe(frame_bgr)

        return {
            "sharpness": round(sharpness, 2),
            "exposure_score": round(exposure_score, 4),
            "motion_score": 0.0,
            "motion_tier": "static",
            "face_detected": face_detected,
            "face_anchor_x": round(face_anchor_x, 4)
        }

    def _score_video(self, file_path: str) -> Dict[str, Any]:
        """Scores a video file by sampling frames and computing metrics."""
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return self._default_score()

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0.0

        if duration <= 0.0:
            cap.release()
            return self._default_score()

        # Sample up to 10 frames across the duration for metrics
        sample_count = min(10, max(3, int(duration / 2.0)))
        sample_times = np.linspace(0.1, duration - 0.1, sample_count)

        sharpness_list = []
        exposure_list = []
        motion_list = []
        face_anchors = []
        face_detected_any = False

        # Accumulator for optical flow vertical columns to detect motion density
        # Divides the width into 5 columns
        col_count = 5
        motion_grid = np.zeros(col_count)

        prev_gray_small = None

        try:
            for idx, t in enumerate(sample_times):
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
                success, frame_bgr = cap.read()
                if not success:
                    continue

                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                h, w = gray.shape

                # 1. Sharpness (Laplacian variance)
                lap = cv2.Laplacian(gray, cv2.CV_64F)
                sharpness_list.append(float(lap.var()))

                # 2. Exposure (HSV V channel mean)
                hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
                v_mean = float(hsv[:, :, 2].mean())
                if v_mean < 30 or v_mean > 220:
                    exp_val = 0.0
                else:
                    exp_val = 1.0 - (abs(v_mean - 125.0) / 95.0)
                exposure_list.append(exp_val)

                # 3. Face Detection via MediaPipe
                has_face, face_x = self._detect_faces_mediapipe(frame_bgr)
                if has_face:
                    face_detected_any = True
                    face_anchors.append(face_x)

                # 4. Motion / Optical Flow (Farneback on downscaled frames)
                # We need consecutive frames, so we read next frame as well
                success_next, frame_next_bgr = cap.read()
                if success_next:
                    gray_next = cv2.cvtColor(frame_next_bgr, cv2.COLOR_BGR2GRAY)
                    
                    small_gray = cv2.resize(gray, (320, 240))
                    small_gray_next = cv2.resize(gray_next, (320, 240))
                    
                    flow = cv2.calcOpticalFlowFarneback(
                        small_gray, small_gray_next, None, 0.5, 3, 15, 3, 5, 1.2, 0
                    )
                    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    flow_mean = float(mag.mean())
                    motion_list.append(flow_mean)

                    # Compute vertical column density
                    small_w = small_gray.shape[1]
                    col_w = small_w // col_count
                    for c in range(col_count):
                        col_flow = mag[:, c * col_w:(c + 1) * col_w]
                        motion_grid[c] += col_flow.mean()

        except Exception as e:
            logger.warning(f"ClipScoringAgent: Error processing video {file_path}: {e}")
        finally:
            cap.release()

        # Compile metrics
        avg_sharpness = float(np.mean(sharpness_list)) if sharpness_list else 100.0
        avg_exposure = float(np.mean(exposure_list)) if exposure_list else 0.8
        avg_motion = float(np.mean(motion_list)) if motion_list else 0.0

        # Motion tier classification
        # Static: motion_score < 0.8
        # Medium: 0.8 <= motion_score < 3.0
        # High: motion_score >= 3.0
        if avg_motion < 0.8:
            motion_tier = "static"
        elif avg_motion < 3.0:
            motion_tier = "medium"
        else:
            motion_tier = "high"

        # Determine face anchor
        if face_detected_any and face_anchors:
            face_anchor_x = float(np.mean(face_anchors))
            logger.info(f"ClipScoringAgent: Face detected in {os.path.basename(file_path)}. Anchor x: {face_anchor_x:.2f}")
        else:
            # Fallback to region with highest motion density
            if motion_list and motion_grid.sum() > 0:
                max_col = int(np.argmax(motion_grid))
                # Map column center to normalized coordinates (0.0 to 1.0)
                face_anchor_x = (max_col + 0.5) / col_count
                logger.info(f"ClipScoringAgent: No face detected. Motion fallback anchor in col {max_col}: {face_anchor_x:.2f}")
            else:
                face_anchor_x = 0.5

        return {
            "sharpness": round(avg_sharpness, 2),
            "exposure_score": round(avg_exposure, 4),
            "motion_score": round(avg_motion, 4),
            "motion_tier": motion_tier,
            "face_detected": face_detected_any,
            "face_anchor_x": round(face_anchor_x, 4)
        }

    def _detect_faces_mediapipe(self, frame_bgr: np.ndarray) -> Tuple[bool, float]:
        """Runs MediaPipe face detection. Returns (detected, face_center_x)."""
        try:
            import mediapipe as mp
            mp_face_detection = mp.solutions.face_detection
            
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            with mp_face_detection.FaceDetection(min_detection_confidence=0.4) as face_detection:
                results = face_detection.process(frame_rgb)
                if results.detections:
                    # Find largest face (biggest bounding box area)
                    largest_face = None
                    max_area = 0.0
                    for detection in results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        area = bbox.width * bbox.height
                        if area > max_area:
                            max_area = area
                            largest_face = bbox
                    
                    if largest_face:
                        center_x = largest_face.xmin + largest_face.width / 2.0
                        return True, float(np.clip(center_x, 0.0, 1.0))
        except Exception as e:
            logger.warning(f"ClipScoringAgent: MediaPipe face detection failed: {e}")
        return False, 0.5

    def _compute_composite_score(self, metrics: Dict[str, Any], style: str) -> float:
        """
        Computes composite score based on:
        Sharpness (40%) + Exposure (25%) + Face presence bonus (20%) + Motion appropriateness (15%).
        """
        # Normalise sharpness (500.0 is target sharp edges value)
        norm_sharp = min(1.0, metrics.get("sharpness", 100.0) / 500.0)
        
        # Exposure
        norm_exp = metrics.get("exposure_score", 0.8)
        
        # Face presence
        face_bonus = 1.0 if metrics.get("face_detected", False) else 0.0
        
        # Motion appropriateness
        motion_score = metrics.get("motion_score", 0.0)
        style_lower = style.lower()
        if any(s in style_lower for s in ["fast", "energy", "ramp", "action", "travel"]):
            # Fast pacing style favors higher motion
            motion_appropriateness = min(1.0, motion_score / 3.5)
        else:
            # Cinematic/slow pacing styles penalise high motion
            motion_appropriateness = max(0.0, 1.0 - (motion_score / 4.0))

        composite = (
            norm_sharp * 0.40 +
            norm_exp * 0.25 +
            face_bonus * 0.20 +
            motion_appropriateness * 0.15
        )
        return round(float(np.clip(composite, 0.0, 1.0)), 4)

    def score_all_clips(self, video_paths: List[str], style: str = "cinematic") -> Dict[str, Dict[str, Any]]:
        """
        Scores a list of video files and returns a dictionary mapping
        each basename filename to its scored details.
        """
        scores = {}
        for path in video_paths:
            filename = os.path.basename(path)
            scores[filename] = self.score_file(path, style)
        return scores

    def score_visual_data(self, visual_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enriches the Gemini media analyses (visual_data) with local ClipScoringAgent metrics.
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
                    metrics = self.score_file(file_path, "cinematic")
                    # Map new metrics to old structure for compatibility
                    metrics = {
                        "sharpness": metrics["sharpness"],
                        "motion_score": metrics["motion_score"],
                        "motion_type": metrics["motion_tier"],
                        "face_present": metrics["face_detected"],
                        "local_score": metrics["composite_score"]
                    }
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
                    metrics = self.score_file(file_path, "cinematic")
                    metrics = {
                        "sharpness": metrics["sharpness"],
                        "motion_score": metrics["motion_score"],
                        "motion_type": metrics["motion_tier"],
                        "face_present": metrics["face_detected"],
                        "local_score": metrics["composite_score"]
                    }
                seg.update(metrics)

        logger.info("ClipScoringAgent: Visual segments successfully scored!")
        return visual_data
