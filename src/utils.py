"""
Utility functions for the Deepfake Detection System.

Includes existing project helpers plus dataset extraction helpers:
  - Logging setup
  - File validation
  - Video discovery and label inference
  - Resume helpers
  - Metadata and JSON writers
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

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

    # Avoid adding duplicate handlers.
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(name)-14s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def validate_video_file(
    file_path: str | Path,
    allowed_extensions: Optional[Set[str]] = None,
) -> bool:
    """
    Check if a file exists and has an allowed video extension.

    Args:
        file_path: Path to the video file.
        allowed_extensions: Allowed video suffixes (lowercase).

    Returns:
        True if valid, False otherwise.
    """
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        return False
    if path.stat().st_size == 0:
        return False

    if allowed_extensions is None:
        try:
            from config import Config

            allowed_extensions = {
                ext.lower()
                for ext in getattr(
                    Config,
                    "ALLOWED_EXTENSIONS",
                    {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"},
                )
            }
        except Exception:
            allowed_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}

    if path.suffix.lower() not in allowed_extensions:
        return False
    return True


def get_video_info(file_path: str | Path) -> dict:
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
    info["duration_seconds"] = round(info["frame_count"] / fps, 2) if fps > 0 else 0.0

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


def safe_cleanup(file_path: str | Path) -> None:
    """Silently remove a temporary file if it exists."""
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
    except OSError:
        pass


def ensure_dir(path: str | Path) -> Path:
    """Create directory if needed and return Path."""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def sanitize_name(text: str) -> str:
    """Sanitize string for safe folder/file naming."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "video"


def list_video_files(dataset_root: str | Path, allowed_extensions: Set[str]) -> List[Path]:
    """
    Recursively list valid video files under dataset_root.
    """
    root = Path(dataset_root)
    if not root.exists() or not root.is_dir():
        return []

    exts = {ext.lower() for ext in allowed_extensions}
    videos = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in exts:
            continue
        if path.stat().st_size == 0:
            continue
        videos.append(path)
    videos.sort()
    return videos


def infer_binary_label(
    video_path: str | Path,
    dataset_root: str | Path,
    real_dir_tokens: Set[str],
    fake_dir_tokens: Set[str],
) -> Optional[str]:
    """
    Infer label (real/fake) from path tokens.

    This supports datasets like:
      - root/real/*.mp4, root/fake/*.mp4
      - FaceForensics++ variants where manipulated subsets are fake.
    """
    video = Path(video_path)
    root = Path(dataset_root)

    try:
        rel_parts = [part.lower() for part in video.relative_to(root).parts[:-1]]
    except ValueError:
        rel_parts = [parent.name.lower() for parent in video.parents]

    for token in rel_parts:
        if token in real_dir_tokens:
            return "real"
        if token in fake_dir_tokens:
            return "fake"

    filename_lower = video.name.lower()
    if "real" in filename_lower or "original" in filename_lower:
        return "real"
    if "fake" in filename_lower or "deepfake" in filename_lower:
        return "fake"
    return None


def make_unique_video_output_dir(
    output_root: str | Path,
    label: str,
    video_path: str | Path,
) -> Path:
    """
    Build a deterministic output directory:
      output_root/label/video_stem

    If there is a name collision from different source folders, append a hash.
    """
    output_root = Path(output_root)
    label_dir = ensure_dir(output_root / label)
    video_path = Path(video_path)
    base_name = sanitize_name(video_path.stem)
    primary = label_dir / base_name

    if not primary.exists():
        return primary

    marker_file = primary / ".source_path.txt"
    source_path = str(video_path.resolve())
    if marker_file.exists():
        try:
            if marker_file.read_text(encoding="utf-8").strip() == source_path:
                return primary
        except OSError:
            pass

    digest = hashlib.sha1(source_path.encode("utf-8")).hexdigest()[:8]
    return label_dir / f"{base_name}_{digest}"


def write_source_marker(output_dir: str | Path, video_path: str | Path) -> None:
    """Write source marker for collision-safe resume behavior."""
    output_dir = ensure_dir(output_dir)
    marker_file = output_dir / ".source_path.txt"
    marker_file.write_text(str(Path(video_path).resolve()), encoding="utf-8")


def count_extracted_frames(video_output_dir: str | Path) -> int:
    """Count extracted frame_*.jpg files in a video output directory."""
    output_dir = Path(video_output_dir)
    if not output_dir.exists() or not output_dir.is_dir():
        return 0
    return sum(1 for _ in output_dir.glob("frame_*.jpg"))


def is_video_already_processed(video_output_dir: str | Path) -> bool:
    """
    Check if a video appears processed.

    A video is considered processed if:
      - it has extracted frame files, or
      - it has a summary.json marker.
    """
    output_dir = Path(video_output_dir)
    if not output_dir.exists() or not output_dir.is_dir():
        return False
    if count_extracted_frames(output_dir) > 0:
        return True
    if (output_dir / "summary.json").exists():
        return True
    return False


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def write_json_file(path: str | Path, data: Any) -> None:
    """Write JSON to disk with UTF-8 encoding."""
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    with path_obj.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_metadata_csv(rows: Sequence[Dict[str, Any]], csv_path: str | Path) -> None:
    """
    Save extraction metadata rows to CSV.

    Rows can include custom keys; this function preserves them.
    """
    csv_path = Path(csv_path)
    ensure_dir(csv_path.parent)

    default_columns = [
        "video_path",
        "video_name",
        "label",
        "output_dir",
        "status",
        "detector",
        "total_frames_read",
        "sampled_frames",
        "saved_frames",
        "skipped_no_face",
        "skipped_errors",
        "duration_seconds",
        "start_time_utc",
        "end_time_utc",
        "bbox_file",
        "message",
    ]

    discovered_keys: List[str] = []
    seen = set(default_columns)
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            discovered_keys.append(key)

    fieldnames = default_columns + discovered_keys

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
