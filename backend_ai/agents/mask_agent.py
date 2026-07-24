import os
import cv2
import numpy as np
import logging

logger = logging.getLogger("agents.mask_generator")

class MaskGeneratorAgent:
    def __init__(self, cache_dir: str = "cache/shared/masks"):
        self.cache_dir = os.path.abspath(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)

    def generate_mask(
        self,
        mask_name: str,
        duration: float,
        width: int = 1080,
        height: int = 1920,
        fps: int = 30
    ) -> str:
        """
        Generates a black-and-white transitions video clip of specified duration and resolution.
        The shape starts at 0% size (all black) and grows to cover the screen (all white).
        Returns the absolute path of the generated .mp4 file.
        """
        # Create fingerprint for cache
        fingerprint = f"{mask_name}_{duration:.3f}_{width}_{height}_{fps}"
        import hashlib
        cache_key = hashlib.md5(fingerprint.encode()).hexdigest()
        mask_file = os.path.join(self.cache_dir, f"{cache_key}.mp4")

        if os.path.exists(mask_file):
            logger.info(f"MaskGeneratorAgent: Found cached transition mask: {mask_file}")
            return mask_file

        logger.info(f"MaskGeneratorAgent: Generating transition mask '{mask_name}' | dur={duration}s | resolution={width}x{height}")

        total_frames = max(1, int(duration * fps))
        
        # Use cv2 VideoWriter to create the black-and-white video
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(mask_file, fourcc, fps, (width, height))

        center_x = width // 2
        center_y = height // 2
        max_dimension = int(np.sqrt(center_x**2 + center_y**2))

        for f in range(total_frames):
            # Calculate interpolation factor (0.0 to 1.0)
            t = f / (total_frames - 1) if total_frames > 1 else 1.0
            
            # Start with a black frame
            frame = np.zeros((height, width, 3), dtype=np.uint8)

            # Draw the shape in white
            if mask_name == "circle":
                radius = int(t * max_dimension)
                if radius > 0:
                    cv2.circle(frame, (center_x, center_y), radius, (255, 255, 255), -1)

            elif mask_name == "heart":
                scale = t * max_dimension * 1.1
                if scale > 0:
                    points = []
                    for theta in np.linspace(0, 2 * np.pi, 100):
                        px = int(center_x + 16 * (np.sin(theta)**3) * (scale / 16.0))
                        py = int(center_y - (13 * np.cos(theta) - 5 * np.cos(2*theta) - 2 * np.cos(3*theta) - np.cos(4*theta)) * (scale / 16.0))
                        points.append([px, py])
                    cv2.fillPoly(frame, [np.array(points, dtype=np.int32)], (255, 255, 255))

            elif mask_name == "star":
                scale = t * max_dimension * 1.2
                if scale > 0:
                    points = []
                    for i in range(10):
                        angle = i * np.pi / 5 - np.pi / 2
                        r = scale if i % 2 == 0 else (scale * 0.4)
                        px = int(center_x + r * np.cos(angle))
                        py = int(center_y + r * np.sin(angle))
                        points.append([px, py])
                    cv2.fillPoly(frame, [np.array(points, dtype=np.int32)], (255, 255, 255))

            elif mask_name == "diamond":
                scale = t * max_dimension * 1.2
                if scale > 0:
                    points = [
                        [center_x, int(center_y - scale)],
                        [int(center_x + scale), center_y],
                        [center_x, int(center_y + scale)],
                        [int(center_x - scale), center_y]
                    ]
                    cv2.fillPoly(frame, [np.array(points, dtype=np.int32)], (255, 255, 255))
            else:
                # Fallback: simple rectangle wipe
                rect_h = int(t * height)
                cv2.rectangle(frame, (0, 0), (width, rect_h), (255, 255, 255), -1)

            writer.write(frame)

        writer.release()
        logger.info(f"MaskGeneratorAgent: Generated mask file successfully: {mask_file}")
        return mask_file
