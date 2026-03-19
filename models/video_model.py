"""
Video deepfake model loading and inference utilities.

This module is local-only: it never downloads models from remote sources.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import setup_logger, softmax

logger = setup_logger("video_model")


def _iter_model_dirs(config, model_dir: Optional[Union[str, Path]] = None):
    if model_dir is not None:
        yield Path(model_dir)
        return
    for path in config.get_video_model_search_dirs():
        yield Path(path)


def _is_transformers_model_dir(path: Path) -> bool:
    return path.exists() and (path / "model.safetensors").exists() and (path / "config.json").exists()


def load_video_model(config=None, model_dir: Optional[Union[str, Path]] = None) -> Tuple:
    """
    Load the local ViT deepfake model and processor.

    Args:
        config: Config object (uses config.Config when None).
        model_dir: Optional explicit directory path.

    Returns:
        (model, processor)

    Raises:
        RuntimeError: when no valid local model directory can be loaded.
    """
    if config is None:
        from config import Config

        config = Config

    from transformers import AutoImageProcessor, AutoModelForImageClassification

    attempted = []
    last_error = None

    for candidate in _iter_model_dirs(config=config, model_dir=model_dir):
        candidate = candidate.resolve()
        attempted.append(str(candidate))
        if not _is_transformers_model_dir(candidate):
            continue
        try:
            logger.info(f"Loading video model from local path: {candidate}")
            model = AutoModelForImageClassification.from_pretrained(
                str(candidate),
                local_files_only=True,
            )
            processor = AutoImageProcessor.from_pretrained(
                str(candidate),
                local_files_only=True,
            )
            model = model.to(config.DEVICE)
            model.eval()
            logger.info("Video model loaded successfully (local-only)")
            return model, processor
        except Exception as exc:  # pragma: no cover - defensive logging
            last_error = exc
            logger.warning(f"Failed to load local model from {candidate}: {exc}")

    error_lines = [
        "Unable to load local video model.",
        "Checked directories:",
        *[f"  - {path}" for path in attempted],
        "Required files per directory: model.safetensors, config.json, preprocessor_config.json",
    ]
    if last_error is not None:
        error_lines.append(f"Last error: {last_error}")
    raise RuntimeError("\n".join(error_lines))


def predict_single_image(model, processor, image: Image.Image, device=None) -> Dict:
    """
    Predict REAL/FAKE for a single image.
    """
    if device is None:
        from config import Config

        device = Config.DEVICE

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits.detach().cpu().numpy()[0]
    probs = softmax(logits)
    pred_idx = int(np.argmax(probs))
    label = "FAKE" if pred_idx == 1 else "REAL"

    return {
        "label": label,
        "confidence": float(probs[pred_idx]),
        "fake_probability": float(probs[1]),
        "real_probability": float(probs[0]),
    }


def predict_batch(
    model,
    processor,
    images: List[Image.Image],
    device=None,
    batch_size: int = 8,
) -> List[Dict]:
    """
    Predict REAL/FAKE for a batch of images.
    """
    if device is None:
        from config import Config

        device = Config.DEVICE

    if not images:
        return []

    results = []
    for start in range(0, len(images), batch_size):
        batch_images = images[start : start + batch_size]
        inputs = processor(images=batch_images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        batch_logits = outputs.logits.detach().cpu().numpy()
        for logits in batch_logits:
            probs = softmax(logits)
            pred_idx = int(np.argmax(probs))
            label = "FAKE" if pred_idx == 1 else "REAL"
            results.append(
                {
                    "label": label,
                    "confidence": float(probs[pred_idx]),
                    "fake_probability": float(probs[1]),
                    "real_probability": float(probs[0]),
                }
            )

    return results


def aggregate_predictions(predictions: List[Dict]) -> Dict:
    """
    Aggregate per-frame predictions into one video prediction.
    """
    if not predictions:
        return {
            "label": "UNKNOWN",
            "confidence": 0.0,
            "fake_probability": 0.0,
            "real_probability": 0.0,
            "num_frames_analyzed": 0,
            "num_fake_frames": 0,
            "num_real_frames": 0,
            "fake_frame_ratio": 0.0,
        }

    fake_probs = [item["fake_probability"] for item in predictions]
    real_probs = [item["real_probability"] for item in predictions]

    avg_fake = float(np.mean(fake_probs))
    avg_real = float(np.mean(real_probs))
    num_fake = sum(1 for item in predictions if item["label"] == "FAKE")
    num_real = sum(1 for item in predictions if item["label"] == "REAL")

    if avg_fake > avg_real:
        label = "FAKE"
        confidence = avg_fake
    else:
        label = "REAL"
        confidence = avg_real

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "fake_probability": round(avg_fake, 4),
        "real_probability": round(avg_real, 4),
        "num_frames_analyzed": len(predictions),
        "num_fake_frames": num_fake,
        "num_real_frames": num_real,
        "fake_frame_ratio": round(num_fake / len(predictions), 4),
    }
