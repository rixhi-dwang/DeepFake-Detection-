"""
End-to-end image deepfake pipeline.

Input image -> face crop (if available) -> local ViT prediction.
"""

import time
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
from PIL import Image

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.video_model import load_video_model, predict_single_image
from src.preprocessing import detect_faces
from src.utils import setup_logger

logger = setup_logger("image_pipeline")

_cached_model = None
_cached_processor = None


def _get_model_and_processor():
    global _cached_model, _cached_processor
    if _cached_model is None or _cached_processor is None:
        _cached_model, _cached_processor = load_video_model()
    return _cached_model, _cached_processor


def _center_crop(image: Image.Image, target_size: int = 224) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    cropped = image.crop((left, top, left + side, top + side))
    return cropped.resize((target_size, target_size), Image.LANCZOS)


def run_image_pipeline(
    image: Image.Image,
    face_confidence: Optional[float] = None,
) -> Dict:
    """
    Predict REAL/FAKE for a single image.
    """
    from config import Config

    if face_confidence is None:
        face_confidence = Config.FACE_CONFIDENCE_THRESHOLD

    result = {
        "status": "error",
        "prediction": None,
        "timing": {},
        "face_detected": False,
        "error": None,
    }

    total_start = time.time()

    try:
        pil_image = image.convert("RGB")
        rgb_np = np.array(pil_image)
        bgr_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
    except Exception as exc:
        result["error"] = f"Invalid image input: {exc}"
        return result

    t0 = time.time()
    try:
        faces = detect_faces([bgr_np], confidence_threshold=face_confidence)
        result["timing"]["face_detection"] = round(time.time() - t0, 3)
    except Exception as exc:
        logger.warning(f"Face detection failed on image, using center crop: {exc}")
        faces = []
        result["timing"]["face_detection"] = round(time.time() - t0, 3)

    if faces:
        face_image = faces[0].resize((Config.FACE_IMAGE_SIZE, Config.FACE_IMAGE_SIZE), Image.LANCZOS)
        result["face_detected"] = True
    else:
        face_image = _center_crop(pil_image, target_size=Config.FACE_IMAGE_SIZE)

    t0 = time.time()
    try:
        model, processor = _get_model_and_processor()
        prediction = predict_single_image(model, processor, face_image, device=Config.DEVICE)
        result["timing"]["model_inference"] = round(time.time() - t0, 3)
        result["timing"]["total"] = round(time.time() - total_start, 3)
        result["prediction"] = prediction
        result["status"] = "success"
        return result
    except Exception as exc:
        result["error"] = f"Model inference failed: {exc}"
        result["timing"]["model_inference"] = round(time.time() - t0, 3)
        result["timing"]["total"] = round(time.time() - total_start, 3)
        return result
