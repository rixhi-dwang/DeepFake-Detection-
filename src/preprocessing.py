"""
Preprocessing module for the Deepfake Detection System.

Handles:
  - Video frame extraction (OpenCV)
  - Face detection and cropping (MTCNN)
  - Image preprocessing for ViT model input
"""

import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import setup_logger

logger = setup_logger("preprocessing")


# ── MTCNN singleton to avoid reloading ────────────────────────────────
_mtcnn_instance = None


def get_mtcnn(device=None):
    """
    Get or create a singleton MTCNN face detector.

    Args:
        device: torch device (auto-detected if None).

    Returns:
        MTCNN detector instance.
    """
    global _mtcnn_instance
    if _mtcnn_instance is None:
        from facenet_pytorch import MTCNN
        from config import Config

        if device is None:
            device = Config.DEVICE

        _mtcnn_instance = MTCNN(
            image_size=Config.FACE_IMAGE_SIZE,
            margin=20,
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            keep_all=False,       # Return only the largest face
            device=device,
            post_process=False,   # Return PIL-compatible values (0-255)
        )
        logger.info(f"MTCNN initialized on {device}")
    return _mtcnn_instance


def extract_frames(
    video_path: str,
    every_n: int = 10,
    max_frames: int = 30,
) -> List[np.ndarray]:
    """
    Extract frames from a video file at regular intervals.

    Args:
        video_path: Path to the video file.
        every_n: Sample every Nth frame.
        max_frames: Maximum number of frames to return.

    Returns:
        List of BGR numpy arrays (OpenCV format).

    Raises:
        FileNotFoundError: If video file doesn't exist.
        RuntimeError: If video cannot be opened.
    """
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = []
    frame_idx = 0

    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % every_n == 0:
            frames.append(frame)

        frame_idx += 1

    cap.release()

    if not frames:
        logger.warning(f"No frames extracted from {video_path}")
    else:
        logger.info(
            f"Extracted {len(frames)} frames from {Path(video_path).name} "
            f"(sampled every {every_n}, total scanned: {frame_idx})"
        )

    return frames


def detect_faces(
    frames: List[np.ndarray],
    confidence_threshold: float = 0.9,
) -> List[Image.Image]:
    """
    Detect and crop faces from a list of video frames using MTCNN.

    Args:
        frames: List of BGR numpy arrays (OpenCV format).
        confidence_threshold: Minimum detection confidence.

    Returns:
        List of cropped face images as PIL Images (RGB).
    """
    mtcnn = get_mtcnn()
    faces = []

    for i, frame in enumerate(frames):
        # Convert BGR (OpenCV) → RGB (PIL)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(rgb_frame)

        try:
            # MTCNN detect: returns boxes and probabilities
            boxes, probs = mtcnn.detect(pil_frame)

            if boxes is not None and probs is not None:
                # Take the highest-confidence face
                best_idx = int(np.argmax(probs))
                best_prob = float(probs[best_idx])

                if best_prob >= confidence_threshold:
                    box = boxes[best_idx].astype(int)
                    # Clamp to image bounds
                    x1 = max(0, box[0])
                    y1 = max(0, box[1])
                    x2 = min(pil_frame.width, box[2])
                    y2 = min(pil_frame.height, box[3])

                    if x2 > x1 and y2 > y1:
                        face_crop = pil_frame.crop((x1, y1, x2, y2))
                        faces.append(face_crop)

        except Exception as e:
            logger.debug(f"Face detection failed on frame {i}: {e}")
            continue

    logger.info(f"Detected {len(faces)} faces from {len(frames)} frames")
    return faces


def preprocess_faces(
    faces: List[Image.Image],
    processor,
    target_size: int = 224,
) -> list:
    """
    Preprocess face images for the ViT model.

    Args:
        faces: List of PIL face images.
        processor: HuggingFace ViTImageProcessor.
        target_size: Target image size (default 224 for ViT).

    Returns:
        List of preprocessed tensors ready for the model.
    """
    if not faces:
        return []

    # Resize all faces to target size
    resized_faces = []
    for face in faces:
        resized = face.resize((target_size, target_size), Image.LANCZOS)
        resized_faces.append(resized)

    # Use the HuggingFace processor for normalization
    inputs = processor(images=resized_faces, return_tensors="pt")
    return inputs


def extract_single_face(frame: np.ndarray) -> Optional[Image.Image]:
    """
    Extract a single face from a BGR frame.

    Convenience function for single-frame processing.

    Args:
        frame: BGR numpy array.

    Returns:
        Cropped face as PIL Image, or None if no face detected.
    """
    result = detect_faces([frame], confidence_threshold=0.8)
    return result[0] if result else None
