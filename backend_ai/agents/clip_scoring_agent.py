import os
# Suppress TensorFlow and MediaPipe C++ log warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
os.environ['ABSL_LOG_LEVEL'] = '3'

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

    # Confidence threshold for MediaPipe face detection.
    # Lowered from 0.4 to 0.2: outdoor/hiking footage frequently produces
    # confidence scores of 0.15-0.35 for faces that are clearly visible but
    # slightly angled, backlit, or partially in motion. At 0.4 these were all
    # silently rejected as "no face," which was the primary cause of good
    # people-clips being scored as faceless and deprioritized by the director.
    FACE_DETECTION_CONFIDENCE = 0.2

    # How densely to sample frames when face-checking a clip. Previously
    # face detection only ran on the same 3-10 frames used for sharpness/
    # exposure/motion scoring (np.linspace across the whole clip), which
    # meant "no face detected" often just meant "no face at those few
    # exact instants." This interval is independent of that sparse sample
    # set and walks the clip on a fixed cadence instead.
    FACE_SAMPLE_INTERVAL_SEC = 0.5

    def __init__(self, cache_dir: str = "data/cache/clip_scores"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        # Detector is created lazily, once per ClipScoringAgent instance,
        # and reused across every frame / every clip it scores -- this is
        # the fix for root cause #1 (a fresh MediaPipe detector context,
        # plus a fresh Tasks-API model load, was being constructed and
        # torn down on EVERY sampled frame). Recreating it that often was
        # not just slow; it's also why sample_count was kept tiny (3-10)
        # in the first place, since each sample carried full init cost.
        self._mp_legacy_detector = None
        self._mp_tasks_detector = None
        self._detector_backend = None  # "legacy" | "tasks" | "haar" | None (not yet resolved)

    def _get_face_detector(self):
        """
        Resolves and returns a reusable face detector, creating it exactly
        once and caching it on the instance. Tries legacy MediaPipe
        solutions first, then the new Tasks API, then falls back to a
        sentinel meaning "use Haar Cascade" -- same three-tier fallback
        chain as before, but resolved ONCE rather than re-attempted on
        every single frame.

        Returns a tuple (backend_name, detector_obj_or_None). backend_name
        is one of "legacy" / "tasks" / "haar". For "haar", detector_obj is
        None -- the Haar cascade is loaded separately since it's cheap to
        construct and doesn't need the same caching treatment.
        """
        if self._detector_backend is not None:
            # Already resolved on a previous call (previous clip / previous
            # frame). Return the cached result instead of re-probing.
            if self._detector_backend == "legacy":
                return "legacy", self._mp_legacy_detector
            elif self._detector_backend == "tasks":
                return "tasks", self._mp_tasks_detector
            else:
                return "haar", None

        # --- Try Legacy MediaPipe solutions (no external model download) ---
        try:
            import mediapipe as mp
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
                mp_face_detection = mp.solutions.face_detection
                detector = mp_face_detection.FaceDetection(
                    min_detection_confidence=self.FACE_DETECTION_CONFIDENCE
                )
                # Smoke-test it immediately on a tiny blank frame so a
                # detector that *constructs* but is actually broken in
                # this environment fails loudly now, not silently 10
                # frames later with a confusing "no face" result.
                _ = detector.process(np.zeros((64, 64, 3), dtype=np.uint8))
                self._mp_legacy_detector = detector
                self._detector_backend = "legacy"
                logger.info("ClipScoringAgent: Using legacy MediaPipe solutions face detector "
                            f"(min_detection_confidence={self.FACE_DETECTION_CONFIDENCE}).")
                return "legacy", detector
        except Exception as e:
            # Was logger.debug before -- promoted to warning. Silently
            # swallowing this was root cause #2: if the legacy API is
            # broken/deprecated in this environment, every single frame
            # falls through to a second (and possibly third) detector
            # with zero visibility that this was happening at all.
            logger.warning(f"ClipScoringAgent: Legacy MediaPipe face detector unavailable, "
                            f"falling back to Tasks API: {e}")

        # --- Try new MediaPipe Tasks API ---
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            model_dir = "data/models"
            model_path = os.path.join(model_dir, "blaze_face_short_range.tflite")
            if not os.path.exists(model_path):
                os.makedirs(model_dir, exist_ok=True)
                url = ("https://storage.googleapis.com/mediapipe-models/face_detector/"
                       "blaze_face_short_range/float16/1/blaze_face_short_range.tflite")
                logger.info(f"Downloading MediaPipe Face Detector model to {model_path}...")
                import urllib.request
                urllib.request.urlretrieve(url, model_path)
                logger.info("MediaPipe model downloaded successfully.")

            base_options = python.BaseOptions(model_asset_path=model_path)
            # min_detection_confidence is now explicitly set to the SAME
            # value as the legacy path above, instead of silently using
            # whatever the library default is. Before this fix, falling
            # through from legacy -> Tasks API wasn't a clean retry of the
            # same criteria, it was an undocumented change in threshold.
            options = vision.FaceDetectorOptions(
                base_options=base_options,
                min_detection_confidence=self.FACE_DETECTION_CONFIDENCE,
            )
            devnull_fd = os.open(os.devnull, os.O_WRONLY)
            old_stderr_fd = os.dup(2)
            os.dup2(devnull_fd, 2)
            try:
                detector = vision.FaceDetector.create_from_options(options)
            finally:
                os.dup2(old_stderr_fd, 2)
                os.close(old_stderr_fd)
                os.close(devnull_fd)

            self._mp_tasks_detector = detector
            self._detector_backend = "tasks"
            logger.info("ClipScoringAgent: Using new MediaPipe Tasks API face detector "
                        f"(min_detection_confidence={self.FACE_DETECTION_CONFIDENCE}).")
            return "tasks", detector
        except Exception as e:
            logger.warning(f"ClipScoringAgent: MediaPipe Tasks API face detector unavailable, "
                            f"falling back to OpenCV Haar Cascade: {e}")

        # --- Fallback: OpenCV Haar Cascade ---
        # Flagged explicitly (not just silently used): Haar Cascade is a
        # meaningfully weaker detector than either MediaPipe path -- it
        # struggles with angled faces, partial occlusion, small faces in
        # frame, and harsh/backlit outdoor lighting (root cause #3). If
        # you see this log line firing in production, that's worth
        # investigating on its own -- it means BOTH MediaPipe paths
        # failed in this environment, and detection quality for every
        # clip from here on is running on the weakest available method.
        logger.warning("ClipScoringAgent: Both MediaPipe backends unavailable -- "
                        "falling back to OpenCV Haar Cascade for ALL face detection. "
                        "Detection quality will be noticeably lower, especially for "
                        "angled, partially occluded, or backlit faces.")
        self._detector_backend = "haar"
        return "haar", None

    def close(self):
        """
        Releases the cached MediaPipe detector. Call this when done with
        the agent (e.g. at the end of a pipeline run) -- the Tasks API
        detector in particular holds onto a TFLite interpreter that's
        cleaner to close explicitly rather than rely on GC.
        """
        if self._mp_legacy_detector is not None:
            try:
                self._mp_legacy_detector.close()
            except Exception:
                pass
            self._mp_legacy_detector = None
        if self._mp_tasks_detector is not None:
            try:
                self._mp_tasks_detector.close()
            except Exception:
                pass
            self._mp_tasks_detector = None
        self._detector_backend = None

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
        backend, detector = self._get_face_detector()
        face_detected, face_anchor_x = self._detect_faces_mediapipe(frame_bgr, backend, detector)

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

        # Sample up to 10 frames across the duration for sharpness/exposure/
        # motion metrics. This sparse set is fine for those metrics (they're
        # averaged, so a handful of samples gives a reasonable mean) -- it
        # was specifically face detection riding on this SAME sparse set
        # that was the problem (root cause #1: "no face at these few exact
        # instants" was being read as "no face in this clip").
        sample_count = min(10, max(3, int(duration / 2.0)))
        sample_times = np.linspace(0.1, duration - 0.1, sample_count)

        # Face detection now gets its OWN, denser sampling cadence,
        # independent of the metrics sampling above. Walking every
        # FACE_SAMPLE_INTERVAL_SEC seconds gives many more chances to catch
        # a face that's only turned toward camera part of the time, which
        # is exactly the hiking-footage scenario (subject turning, looking
        # around, partially out of frame) that was producing false
        # "no face detected" results.
        face_sample_times = np.arange(0.1, max(0.1, duration - 0.1), self.FACE_SAMPLE_INTERVAL_SEC)
        if len(face_sample_times) == 0:
            face_sample_times = sample_times  # very short clip fallback

        sharpness_list = []
        exposure_list = []
        motion_list = []
        face_anchors = []
        face_detected_any = False

        # Accumulator for optical flow vertical columns to detect motion density
        # Divides the width into 5 columns
        col_count = 5
        motion_grid = np.zeros(col_count)

        # Resolve the face detector ONCE for this whole clip (and in
        # practice, once per ClipScoringAgent instance -- the backend
        # choice is cached across clips too, see _get_face_detector).
        # This is the fix for root cause #1: previously a fresh detector
        # (and, on the Tasks API path, a fresh model load + stderr
        # redirect) was constructed and torn down on every single sampled
        # frame. That cost is why sample_count was kept tiny in the first
        # place -- now that it's paid once, we can afford to sample faces
        # far more densely (see face_sample_times above) at a fraction of
        # the original total cost.
        face_backend, face_detector = self._get_face_detector()

        try:
            # --- Pass 1: sharpness / exposure / motion on the sparse set ---
            for idx, t in enumerate(sample_times):
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
                success, frame_bgr = cap.read()
                if not success:
                    continue

                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

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

                # 3. Motion / Optical Flow (Farneback on downscaled frames)
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

            # --- Pass 2: face detection on its own, denser cadence ---
            # Separated from pass 1 on purpose: face sampling density and
            # metric sampling density are independent tuning knobs now,
            # rather than being accidentally coupled by sharing one loop.
            for t in face_sample_times:
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
                success, frame_bgr = cap.read()
                if not success:
                    continue

                has_face, face_x = self._detect_faces_mediapipe(
                    frame_bgr, face_backend, face_detector
                )
                if has_face:
                    face_detected_any = True
                    face_anchors.append(face_x)

        except Exception as e:
            logger.warning(f"ClipScoringAgent: Error processing video {file_path}: {e}")

        finally:
            cap.release()

        # --- Pass 3: HOG body/person scan (clip-level fallback) ---
        # Face detection requires a visible face. It will always fail for:
        #   - Person facing away from camera (back view)
        #   - Side profile (partially caught by profile cascade, but unreliable)
        #   - Person far from camera (face too small, < ~30px)
        #   - Extreme close-up cropping face out of frame
        # HOG (Histogram of Oriented Gradients) detects the full human body
        # silhouette, catching all of the above cases.
        # We only run it once per clip, after face detection has already
        # had its full dense-sampling pass, to keep overhead low.
        # Cap is released above; _hog_person_scan opens its own capture.
        if not face_detected_any:
            hog_result = self._hog_person_scan(file_path, duration)
            if hog_result[0]:  # person found
                face_detected_any = True
                face_anchors.append(hog_result[1])
                logger.info(
                    f"ClipScoringAgent: HOG body scan found a person in "
                    f"{os.path.basename(file_path)} (face detection had missed). "
                    f"Anchor x={hog_result[1]:.2f}"
                )

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
            logger.info(
                f"ClipScoringAgent: Face detected in {os.path.basename(file_path)}. "
                f"Anchor x: {face_anchor_x:.2f} (hit on {len(face_anchors)}/{len(face_sample_times)} "
                f"sampled frames, backend={face_backend})."
            )
        else:
            # Fallback to region with highest motion density
            if motion_list and motion_grid.sum() > 0:
                max_col = int(np.argmax(motion_grid))
                # Map column center to normalized coordinates (0.0 to 1.0)
                face_anchor_x = (max_col + 0.5) / col_count
                logger.info(
                    f"ClipScoringAgent: No face detected in {os.path.basename(file_path)} "
                    f"across {len(face_sample_times)} sampled frames (backend={face_backend}). "
                    f"Motion fallback anchor in col {max_col}: {face_anchor_x:.2f}"
                )
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

    def _hog_person_scan(self, video_path: str, duration: float) -> Tuple[bool, float]:
        """
        Clip-level person detection using OpenCV's built-in HOG + SVM pedestrian
        detector. This is the fallback when all face-based detectors find nothing.

        HOG detects the full human body silhouette, so it works for:
          - Back-facing persons (most common missed case)
          - Side profiles
          - Persons at moderate distance (body visible even if face is tiny)
          - Partial body crops (upper body, torso)

        Limitations (honest):
          - Very distant persons (< ~40px tall) will still be missed
          - Extreme close-ups where no body outline is visible will be missed
          - Dense crowds can confuse the detector
          - Slower than face detection; runs only once per clip on a few frames

        Returns (person_found: bool, anchor_x: float).
        """
        try:
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        except Exception as e:
            logger.warning(f"ClipScoringAgent: Could not initialise HOG detector: {e}")
            return False, 0.5

        # Sample 5 frames evenly across the clip. Five is enough for a
        # clip-level yes/no decision; more frames would slow renders measurably.
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, 0.5

        try:
            check_times = np.linspace(max(0.05, 0.1), max(0.1, duration - 0.1), min(5, max(1, int(duration))))
            best: Tuple[int, float] = None  # (area, center_x)

            for t in check_times:
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
                ret, frame = cap.read()
                if not ret:
                    continue

                # HOG works best on frames ~640px wide. Downscale if larger;
                # upscale if very small (< 300px wide) to give the 64x128 window
                # room to fit.
                h_f, w_f = frame.shape[:2]
                target_w = 640
                if w_f != target_w:
                    scale_f = target_w / w_f
                    frame = cv2.resize(
                        frame,
                        (target_w, max(1, int(h_f * scale_f))),
                        interpolation=cv2.INTER_AREA if scale_f < 1 else cv2.INTER_LINEAR
                    )

                try:
                    # winStride=(8,8): fine stride for better detection coverage.
                    # padding=(16,16): extra border helps catch persons near edges.
                    # scale=1.05: fine image pyramid catches a wider size range.
                    # finalThreshold=0.3: permissive — we'd rather have a false
                    #   positive (wrongly mark clip as having a person) than miss
                    #   a real person and deprioritize a good clip.
                    found, weights = hog.detectMultiScale(
                        frame,
                        winStride=(8, 8),
                        padding=(16, 16),
                        scale=1.05,
                        finalThreshold=0.3,
                    )
                except Exception:
                    continue

                if len(found) == 0:
                    continue

                # Take the detection with largest area (most prominent person)
                for i, (x, y, wb, hb) in enumerate(found):
                    area = int(wb) * int(hb)
                    cx = (x + wb / 2.0) / frame.shape[1]
                    if best is None or area > best[0]:
                        best = (area, float(cx))

                if best is not None:
                    # Found a person — no need to check more frames
                    break

            if best is not None:
                return True, float(np.clip(best[1], 0.0, 1.0))
            return False, 0.5

        finally:
            cap.release()

    def _detect_faces_mediapipe(
        self, frame_bgr: np.ndarray, backend: str, detector
    ) -> Tuple[bool, float]:
        """
        Runs face detection on a single frame using an ALREADY-RESOLVED
        detector (backend + detector_obj from _get_face_detector()).

        This used to resolve the legacy -> Tasks API -> Haar fallback
        chain from scratch on every single call (i.e. once per sampled
        frame). That meant up to 3 detector constructions/teardowns per
        frame, which was both slow and the reason sample_count was kept
        tiny (3-10 frames/clip) -- a small budget that made "no face
        detected" frequently mean "unlucky sampling," not "no face."

        Now the fallback chain is resolved ONCE per ClipScoringAgent
        instance (see _get_face_detector), and this method just runs a
        single frame through whichever detector was resolved.
        """
        try:
            # Rescale very large frames before detection.
            # On 4K footage (3840×2160) a face that looks "large" to the eye
            # occupies maybe 100×100 pixels — only 2.6% of frame width — and
            # both MediaPipe and Haar struggle at that scale. Resizing to a
            # max dimension of 1280px keeps the face-to-frame ratio reasonable
            # and also speeds up detection significantly.
            MAX_DETECT_WIDTH = 1280
            h_orig, w_orig = frame_bgr.shape[:2]
            if w_orig > MAX_DETECT_WIDTH:
                scale = MAX_DETECT_WIDTH / w_orig
                detect_frame = cv2.resize(
                    frame_bgr,
                    (MAX_DETECT_WIDTH, int(h_orig * scale)),
                    interpolation=cv2.INTER_AREA
                )
            else:
                scale = 1.0
                detect_frame = frame_bgr

            if backend == "legacy":
                frame_rgb = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2RGB)
                results = detector.process(frame_rgb)
                if results.detections:
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
                return False, 0.5

            elif backend == "tasks":
                import mediapipe as mp
                h, w = detect_frame.shape[:2]
                frame_rgb = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                results = detector.detect(mp_image)
                if results and results.detections:
                    largest_face = None
                    max_area = 0.0
                    for detection in results.detections:
                        bbox = detection.bounding_box
                        area = bbox.width * bbox.height
                        if area > max_area:
                            max_area = area
                            largest_face = bbox
                    if largest_face:
                        center_x = (largest_face.origin_x + largest_face.width / 2.0) / w
                        return True, float(np.clip(center_x, 0.0, 1.0))
                return False, 0.5

            else:  # "haar" -- multi-cascade approach
                # Single frontal cascade at strict params catches maybe 40% of
                # real-world outdoor faces. The multi-pass approach below tries:
                #   1. Frontal default + alt + alt2 on the original grayscale
                #   2. Profile cascade (catches side-angle faces)
                #   3. Histogram-equalized version of all the above (fixes backlit/underexposed)
                #   4. Horizontally flipped image for right-profile faces
                # All passes use relaxed params: scaleFactor=1.05 (finer pyramid),
                # minNeighbors=2 (more permissive), minSize=(20,20) (catches small faces).
                gray = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2GRAY)
                dw = detect_frame.shape[1]  # width after potential rescale

                cascade_files = [
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
                    cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml',
                    cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml',
                    cv2.data.haarcascades + 'haarcascade_profileface.xml',
                ]

                # Build list of (image, x_flip) pairs to try per cascade.
                # x_flip=True means x coordinates need to be mirrored back.
                gray_eq = cv2.equalizeHist(gray)
                frames_to_try = [
                    (gray,    False),   # original
                    (gray_eq, False),   # equalized (helps backlit/dark footage)
                    (cv2.flip(gray, 1), True),    # flipped (catches right-profile)
                    (cv2.flip(gray_eq, 1), True), # equalized + flipped
                ]

                best_face = None   # (area, center_x)
                for cascade_path in cascade_files:
                    cascade = cv2.CascadeClassifier(cascade_path)
                    if cascade.empty():
                        continue
                    for img, flipped in frames_to_try:
                        try:
                            faces = cascade.detectMultiScale(
                                img,
                                scaleFactor=1.05,
                                minNeighbors=2,
                                minSize=(20, 20),
                                flags=cv2.CASCADE_SCALE_IMAGE
                            )
                        except Exception:
                            continue
                        if len(faces) > 0:
                            for (x, y, wb, hb) in faces:
                                area = wb * hb
                                raw_cx = (x + wb / 2.0) / dw
                                # If the image was flipped, mirror x back
                                cx = (1.0 - raw_cx) if flipped else raw_cx
                                if best_face is None or area > best_face[0]:
                                    best_face = (area, cx)

                if best_face is not None:
                    return True, float(np.clip(best_face[1], 0.0, 1.0))
                return False, 0.5

        except Exception as e:
            logger.warning(f"ClipScoringAgent: Face detection failed on a frame "
                            f"(backend={backend}): {e}")
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
