"""
Face detection and face crop helpers.

Primary detector: MTCNN (facenet_pytorch)
Fallback detector: OpenCV Haar Cascade
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
from PIL import Image

from src.utils import setup_logger


_RESAMPLE_LANCZOS = getattr(Image, "Resampling", Image).LANCZOS


@dataclass(slots=True)
class FaceDetection:
    """Single face detection result."""

    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    score: float
    detector: str


class FaceDetector:
    """
    Face detector wrapper that prefers MTCNN and falls back to Haar Cascade.
    """

    def __init__(
        self,
        prefer_mtcnn: bool = True,
        min_confidence: float = 0.90,
        haar_cascade_path: Optional[Path] = None,
        logger=None,
    ) -> None:
        self.logger = logger or setup_logger("face_detection")
        self.prefer_mtcnn = prefer_mtcnn
        self.min_confidence = float(min_confidence)
        self.detector_name = "uninitialized"
        self.mtcnn = None
        self.haar_cascade = None
        self._initialize_detector(haar_cascade_path=haar_cascade_path)

    def _initialize_detector(self, haar_cascade_path: Optional[Path]) -> None:
        """Initialize MTCNN if available, otherwise use Haar Cascade."""
        if self.prefer_mtcnn:
            try:
                from facenet_pytorch import MTCNN

                try:
                    import torch

                    device = "cuda:0" if torch.cuda.is_available() else "cpu"
                except Exception:
                    device = "cpu"

                self.mtcnn = MTCNN(keep_all=True, device=device, post_process=False)
                self.detector_name = "mtcnn"
                self.logger.info("Face detector initialized with MTCNN on %s", device)
                return
            except Exception as exc:
                self.logger.warning(
                    "MTCNN unavailable, falling back to Haar Cascade. Reason: %s",
                    exc,
                )

        if haar_cascade_path is None:
            cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        else:
            cascade_path = Path(haar_cascade_path)

        self.haar_cascade = cv2.CascadeClassifier(str(cascade_path))
        if self.haar_cascade.empty():
            raise RuntimeError(f"Failed to load Haar Cascade file: {cascade_path}")
        self.detector_name = "haar"
        self.logger.info("Face detector initialized with Haar Cascade: %s", cascade_path)

    def detect_largest_face(self, frame_bgr) -> Optional[FaceDetection]:
        """
        Detect faces in a frame and return the largest one.

        Returns:
            FaceDetection or None if no valid face is found.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return None

        if self.detector_name == "mtcnn":
            return self._detect_mtcnn(frame_bgr)
        return self._detect_haar(frame_bgr)

    def _detect_mtcnn(self, frame_bgr) -> Optional[FaceDetection]:
        """Run MTCNN and keep only the largest valid face."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        boxes, probs = self.mtcnn.detect(pil_image)

        if boxes is None or len(boxes) == 0:
            return None

        height, width = frame_bgr.shape[:2]
        candidates = []

        for idx, box in enumerate(boxes):
            if box is None:
                continue

            score = 0.0
            if probs is not None and idx < len(probs) and probs[idx] is not None:
                score = float(probs[idx])

            if score < self.min_confidence:
                continue

            x1, y1, x2, y2 = self._clamp_bbox(box, width=width, height=height)
            if x2 <= x1 or y2 <= y1:
                continue

            area = (x2 - x1) * (y2 - y1)
            candidates.append((area, (x1, y1, x2, y2), score))

        if not candidates:
            return None

        _, bbox, score = max(candidates, key=lambda item: item[0])
        return FaceDetection(bbox=bbox, score=score, detector="mtcnn")

    def _detect_haar(self, frame_bgr) -> Optional[FaceDetection]:
        """Run Haar Cascade and keep only the largest valid face."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.haar_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )

        if faces is None or len(faces) == 0:
            return None

        largest = max(faces, key=lambda rect: int(rect[2]) * int(rect[3]))
        x, y, w, h = [int(v) for v in largest]

        height, width = frame_bgr.shape[:2]
        x1, y1, x2, y2 = self._clamp_bbox((x, y, x + w, y + h), width, height)
        if x2 <= x1 or y2 <= y1:
            return None

        return FaceDetection(bbox=(x1, y1, x2, y2), score=1.0, detector="haar")

    @staticmethod
    def _clamp_bbox(box, width: int, height: int) -> Tuple[int, int, int, int]:
        """Clamp bbox coordinates to image boundaries."""
        x1, y1, x2, y2 = box
        x1 = max(0, min(int(round(x1)), width - 1))
        y1 = max(0, min(int(round(y1)), height - 1))
        x2 = max(0, min(int(round(x2)), width))
        y2 = max(0, min(int(round(y2)), height))
        return x1, y1, x2, y2

    def crop_face_with_margin(
        self,
        frame_bgr,
        bbox: Tuple[int, int, int, int],
        margin_ratio: float = 0.20,
        image_size: Tuple[int, int] = (224, 224),
    ) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
        """
        Crop face with margin padding and resize to target size.

        Returns:
            (PIL.Image, expanded_bbox)
        """
        x1, y1, x2, y2 = bbox
        height, width = frame_bgr.shape[:2]

        face_w = max(1, x2 - x1)
        face_h = max(1, y2 - y1)
        pad_x = int(face_w * margin_ratio)
        pad_y = int(face_h * margin_ratio)

        ex1 = max(0, x1 - pad_x)
        ey1 = max(0, y1 - pad_y)
        ex2 = min(width, x2 + pad_x)
        ey2 = min(height, y2 + pad_y)

        if ex2 <= ex1 or ey2 <= ey1:
            raise ValueError("Expanded face crop has invalid dimensions")

        face_bgr = frame_bgr[ey1:ey2, ex1:ex2]
        if face_bgr is None or face_bgr.size == 0:
            raise ValueError("Face crop is empty")

        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_img = Image.fromarray(face_rgb).resize(image_size, _RESAMPLE_LANCZOS)
        return face_img, (ex1, ey1, ex2, ey2)

    def extract_face(
        self,
        frame_bgr,
        margin_ratio: float = 0.20,
        image_size: Tuple[int, int] = (224, 224),
    ) -> Tuple[Optional[Image.Image], Optional[FaceDetection], Optional[Tuple[int, int, int, int]]]:
        """
        Convenience method: detect largest face and return cropped image.

        Returns:
            (face_image_or_none, detection_or_none, expanded_bbox_or_none)
        """
        detection = self.detect_largest_face(frame_bgr)
        if detection is None:
            return None, None, None

        face_img, expanded_bbox = self.crop_face_with_margin(
            frame_bgr=frame_bgr,
            bbox=detection.bbox,
            margin_ratio=margin_ratio,
            image_size=image_size,
        )
        return face_img, detection, expanded_bbox

