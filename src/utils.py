"""
Utility functions for the Deepfake Detection System.

Provides logging setup, file validation, and common helper functions
used across all pipeline modules.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def setup_logger(
    name: str = "deepfake",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create a configured logger with console and optional file output.

    Args:
        name: Logger name.
        log_file: Optional path to log file.
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s | %(name)-12s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def validate_video_file(file_path: str) -> bool:
    """
    Check if a file exists and has an allowed video extension.

    Args:
        file_path: Path to the video file.

    Returns:
        True if valid, False otherwise.
    """
    from config import Config

    path = Path(file_path)
    if not path.exists():
        return False
    if path.suffix.lower() not in Config.ALLOWED_EXTENSIONS:
        return False
    if path.stat().st_size == 0:
        return False
    return True


def get_video_info(file_path: str) -> dict:
    """
    Get basic video info using OpenCV.

    Args:
        file_path: Path to the video file.

    Returns:
        Dict with fps, frame_count, width, height, duration.
    """
    import cv2

    cap = cv2.VideoCapture(str(file_path))
    if not cap.isOpened():
        return {"error": f"Cannot open video: {file_path}"}

    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }

    fps = info["fps"]
    if fps > 0:
        info["duration_seconds"] = round(info["frame_count"] / fps, 2)
    else:
        info["duration_seconds"] = 0.0

    cap.release()
    return info


def softmax(logits):
    """
    Compute softmax probabilities from raw logits.

    Args:
        logits: Raw model output logits (numpy array or list).

    Returns:
        Softmax probabilities as numpy array.
    """
    import numpy as np

    logits = np.array(logits, dtype=np.float64)
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / exp_logits.sum()


def safe_cleanup(file_path: str):
    """Silently remove a temporary file if it exists."""
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
    except OSError:
        pass
