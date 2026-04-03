"""
Configuration for converting deepfake videos into a face-cropped image dataset.

This file is intentionally self-contained so the extraction script can run
independently from the rest of the project.
"""

from __future__ import annotations

import argparse
import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Set, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "FaceForensics++_C23"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "frames"


@dataclass(slots=True)
class FrameExtractionConfig:
    """
    Runtime configuration for frame extraction and face cropping.

    Required core parameters requested:
      - dataset_path
      - output_path
      - frame_skip
      - max_frames
      - image_size
    """

    dataset_path: Path = DEFAULT_DATASET_PATH
    output_path: Path = DEFAULT_OUTPUT_PATH
    frame_skip: int = 10
    max_frames: int = 30
    image_size: Tuple[int, int] = (224, 224)

    face_margin: float = 0.20
    min_face_confidence: float = 0.90
    prefer_mtcnn: bool = True
    resume: bool = True
    num_workers: int = 1

    save_metadata: bool = True
    save_bboxes: bool = False
    metadata_filename: str = "metadata.csv"
    bboxes_filename: str = "bboxes.json"

    jpeg_quality: int = 95
    log_level: str = "INFO"

    allowed_extensions: Set[str] = field(
        default_factory=lambda: {
            ".mp4",
            ".avi",
            ".mov",
            ".mkv",
            ".webm",
            ".flv",
            ".mpeg",
            ".mpg",
        }
    )
    real_dir_tokens: Set[str] = field(default_factory=lambda: {"real", "original"})
    fake_dir_tokens: Set[str] = field(
        default_factory=lambda: {
            "fake",
            "deepfake",
            "deepfakes",
            "deepfakedetection",
            "face2face",
            "faceswap",
            "faceshifter",
            "neuraltextures",
            "manipulated",
        }
    )

    def __post_init__(self) -> None:
        self.dataset_path = Path(self.dataset_path)
        self.output_path = Path(self.output_path)

        if isinstance(self.image_size, int):
            size = int(self.image_size)
            self.image_size = (size, size)
        elif len(self.image_size) == 2:
            self.image_size = (int(self.image_size[0]), int(self.image_size[1]))
        else:
            raise ValueError("image_size must be an int or a (width, height) tuple")

        self.frame_skip = int(self.frame_skip)
        self.max_frames = int(self.max_frames)
        self.num_workers = max(1, int(self.num_workers))
        self.face_margin = float(self.face_margin)
        self.min_face_confidence = float(self.min_face_confidence)
        self.jpeg_quality = int(self.jpeg_quality)
        self.log_level = self.log_level.upper()
        self.allowed_extensions = {ext.lower() for ext in self.allowed_extensions}
        self.real_dir_tokens = {token.lower() for token in self.real_dir_tokens}
        self.fake_dir_tokens = {token.lower() for token in self.fake_dir_tokens}

        self.validate()

    def validate(self) -> None:
        """Validate config values and raise ValueError for invalid inputs."""
        if self.frame_skip <= 0:
            raise ValueError("frame_skip must be > 0")
        if self.max_frames <= 0:
            raise ValueError("max_frames must be > 0")
        if self.image_size[0] <= 0 or self.image_size[1] <= 0:
            raise ValueError("image_size values must be > 0")
        if not (0.0 <= self.face_margin < 1.0):
            raise ValueError("face_margin must be in [0.0, 1.0)")
        if not (0.0 <= self.min_face_confidence <= 1.0):
            raise ValueError("min_face_confidence must be in [0.0, 1.0]")
        if not (1 <= self.jpeg_quality <= 100):
            raise ValueError("jpeg_quality must be between 1 and 100")
        if self.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("log_level must be one of DEBUG/INFO/WARNING/ERROR/CRITICAL")

    def ensure_output_dirs(self) -> None:
        """Create required output directories."""
        self.output_path.mkdir(parents=True, exist_ok=True)
        (self.output_path / "real").mkdir(parents=True, exist_ok=True)
        (self.output_path / "fake").mkdir(parents=True, exist_ok=True)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for extraction script."""
    parser = argparse.ArgumentParser(
        description="Convert deepfake videos into a face-cropped image dataset."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"Input dataset root directory (default: {DEFAULT_DATASET_PATH})",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output frames directory (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=10,
        help="Sample every Nth frame (default: 10)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=30,
        help="Maximum number of face frames to save per video (default: 30)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Output image size (square). Example: 224 -> (224, 224)",
    )
    parser.add_argument(
        "--face-margin",
        type=float,
        default=0.20,
        help="Padding margin ratio around detected face (default: 0.20)",
    )
    parser.add_argument(
        "--min-face-confidence",
        type=float,
        default=0.90,
        help="Minimum confidence for MTCNN face detections (default: 0.90)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker threads for video-level parallelism (default: 1)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume mode and reprocess videos even if output exists.",
    )
    parser.add_argument(
        "--save-bboxes",
        action="store_true",
        help="Save per-frame face bounding boxes in JSON for each video.",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Disable metadata.csv export.",
    )
    parser.add_argument(
        "--metadata-filename",
        type=str,
        default="metadata.csv",
        help="Metadata CSV filename under output directory (default: metadata.csv)",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="Saved JPEG quality from 1-100 (default: 95)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--disable-mtcnn",
        action="store_true",
        help="Force Haar Cascade detector instead of trying MTCNN first.",
    )
    return parser


def frame_config_from_args(args: argparse.Namespace) -> FrameExtractionConfig:
    """Create FrameExtractionConfig from parsed CLI args."""
    image_size = (int(args.image_size), int(args.image_size))
    return FrameExtractionConfig(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        frame_skip=args.frame_skip,
        max_frames=args.max_frames,
        image_size=image_size,
        face_margin=args.face_margin,
        min_face_confidence=args.min_face_confidence,
        prefer_mtcnn=not args.disable_mtcnn,
        resume=not args.no_resume,
        num_workers=args.num_workers,
        save_metadata=not args.no_metadata,
        save_bboxes=args.save_bboxes,
        metadata_filename=args.metadata_filename,
        jpeg_quality=args.jpeg_quality,
        log_level=args.log_level,
    )


def _load_root_project_config_class() -> Optional[type]:
    """
    Try to expose the root-level Config class for compatibility.

    This avoids accidental breakage in modules that do `from config import Config`
    when run from different working directories.
    """
    root_config_path = PROJECT_ROOT / "config.py"
    if not root_config_path.exists():
        return None

    try:
        spec = importlib.util.spec_from_file_location(
            "_project_root_config",
            str(root_config_path),
        )
        if spec is None or spec.loader is None:
            return None

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config_cls = getattr(module, "Config", None)
        if isinstance(config_cls, type):
            return config_cls
    except Exception:
        return None
    return None


class _CompatConfig:
    """Fallback minimal Config if root config is unavailable."""

    PROJECT_ROOT = PROJECT_ROOT
    DATASET_ROOT = DEFAULT_DATASET_PATH
    IMAGE_DATASET_ROOT = DEFAULT_OUTPUT_PATH
    FRAME_SAMPLE_RATE = 10
    MAX_FRAMES = 30
    FACE_IMAGE_SIZE = 224
    FACE_CONFIDENCE_THRESHOLD = 0.90
    ALLOWED_EXTENSIONS = {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm",
        ".flv",
        ".mpeg",
        ".mpg",
    }


Config = _load_root_project_config_class() or _CompatConfig

