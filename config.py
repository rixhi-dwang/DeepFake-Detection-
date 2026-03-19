"""
Central configuration for the deepfake detection system.

This project is configured to run fully offline/local by default.
"""

import os
from pathlib import Path

import torch


class Config:
    """Master configuration class."""

    PROJECT_ROOT = Path(__file__).resolve().parent

    # Local-only runtime flags
    LOCAL_ONLY_MODE = os.getenv("DEEPFAKE_LOCAL_ONLY", "1") == "1"
    if LOCAL_ONLY_MODE:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    # Local model paths
    PRETRAINED_DIR = PROJECT_ROOT / "pretrained"
    LOCAL_MODEL_DIR = PRETRAINED_DIR
    LOCAL_CHECKPOINT_DIR = PRETRAINED_DIR / "checkpoint-5252"

    # Audio model paths (optional local audio support)
    AUDIO_MODEL_DIR = PRETRAINED_DIR / "audio_model"
    AUDIO_CLASSIFIER_PATH = AUDIO_MODEL_DIR / "audio_classifier.pt"
    ENABLE_AUDIO = os.getenv("DEEPFAKE_ENABLE_AUDIO", "0") == "1"

    # Dataset paths
    DATA_DIR = PROJECT_ROOT / "data"
    DATASET_ROOT = DATA_DIR / "FaceForensics++_C23"
    IMAGE_DATASET_ROOT = PROJECT_ROOT / "converted_images"
    ORIGINAL_VIDEOS = DATASET_ROOT / "original"
    DEEPFAKE_VIDEOS = DATASET_ROOT / "Deepfakes"
    FACE2FACE_VIDEOS = DATASET_ROOT / "Face2Face"
    FACESWAP_VIDEOS = DATASET_ROOT / "FaceSwap"
    FACESHIFTER_VIDEOS = DATASET_ROOT / "FaceShifter"
    NEURALTEXTURES_VIDEOS = DATASET_ROOT / "NeuralTextures"
    CSV_DIR = DATASET_ROOT / "csv"

    # Output directories
    CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
    TRAINING_RUNS_DIR = PROJECT_ROOT / "training_runs"
    LOGS_DIR = PROJECT_ROOT / "logs"
    TEMP_DIR = PROJECT_ROOT / "runtime_temp"

    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Video/Image pipeline settings
    FRAME_SAMPLE_RATE = 10
    MAX_FRAMES = 30
    FACE_IMAGE_SIZE = 224
    FACE_CONFIDENCE_THRESHOLD = 0.9
    VIDEO_BATCH_SIZE = 8

    # Audio pipeline settings
    AUDIO_SAMPLE_RATE = 16000
    AUDIO_MAX_DURATION = 10
    MEL_N_MELS = 128
    MEL_N_FFT = 2048
    MEL_HOP_LENGTH = 512

    # Training defaults
    TRAIN_EPOCHS = 5
    TRAIN_BATCH_SIZE = 16
    EVAL_BATCH_SIZE = 8
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 0.01
    WARMUP_STEPS = 100
    MAX_GRAD_NORM = 1.0
    TRAIN_RATIO = 0.8
    RANDOM_SEED = 42
    NUM_WORKERS = 0

    # Fusion defaults
    FUSION_VIDEO_WEIGHT = 0.7
    FUSION_AUDIO_WEIGHT = 0.3
    FUSION_THRESHOLD = 0.5

    # API settings
    API_HOST = "127.0.0.1"
    API_PORT = 8000
    MAX_UPLOAD_SIZE_MB = 500
    MAX_IMAGE_UPLOAD_SIZE_MB = 25
    ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
    ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    LOCAL_ALLOWED_HOSTS = {"127.0.0.1", "localhost", "::1"}
    LOCAL_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"

    # Labels
    ID2LABEL = {0: "REAL", 1: "FAKE"}
    LABEL2ID = {"REAL": 0, "FAKE": 1}

    @classmethod
    def get_video_model_search_dirs(cls):
        """Return model directories in priority order."""
        run_candidates = []
        if cls.TRAINING_RUNS_DIR.exists():
            run_candidates.extend(cls.TRAINING_RUNS_DIR.glob("*/best_model"))
            run_candidates.extend(cls.TRAINING_RUNS_DIR.glob("*/final_model"))
            run_candidates = sorted(
                run_candidates,
                key=lambda path: path.stat().st_mtime if path.exists() else 0,
                reverse=True,
            )

        static_candidates = [
            cls.CHECKPOINTS_DIR / "best_model",
            cls.CHECKPOINTS_DIR / "final_model",
            cls.LOCAL_MODEL_DIR,
            cls.LOCAL_CHECKPOINT_DIR,
        ]

        ordered = []
        seen = set()
        for path in list(run_candidates) + static_candidates:
            normalized = str(path.resolve()) if path.exists() else str(path)
            if normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(path)
        return ordered

    @classmethod
    def ensure_dirs(cls):
        """Create required output directories if they do not exist."""
        for directory in [
            cls.CHECKPOINTS_DIR,
            cls.TRAINING_RUNS_DIR,
            cls.LOGS_DIR,
            cls.TEMP_DIR,
            cls.IMAGE_DATASET_ROOT,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _is_valid_transformers_dir(cls, directory: Path) -> bool:
        required = ("model.safetensors", "config.json")
        return directory.exists() and all((directory / name).exists() for name in required)

    @classmethod
    def validate(cls):
        """Validate critical paths and return warnings."""
        warnings = []

        has_video_model = any(
            cls._is_valid_transformers_dir(path) for path in cls.get_video_model_search_dirs()
        )
        if not has_video_model:
            warnings.append(
                "No local video model found. Put model.safetensors + config.json in "
                "pretrained/ or checkpoints/{best_model,final_model}."
            )

        if cls.ENABLE_AUDIO:
            if not cls.AUDIO_MODEL_DIR.exists():
                warnings.append(
                    f"Audio is enabled but missing local audio model dir: {cls.AUDIO_MODEL_DIR}"
                )
            if not cls.AUDIO_CLASSIFIER_PATH.exists():
                warnings.append(
                    f"Audio is enabled but missing classifier weights: {cls.AUDIO_CLASSIFIER_PATH}"
                )

        if not cls.DATASET_ROOT.exists():
            warnings.append(
                f"Dataset not found: {cls.DATASET_ROOT}. "
                "Video-to-image conversion and training will be unavailable."
            )

        if not torch.cuda.is_available():
            warnings.append("CUDA not available - running on CPU (slower)")

        return warnings

    @classmethod
    def summary(cls):
        """Print a concise runtime summary."""
        print("=" * 60)
        print("  DEEPFAKE DETECTION SYSTEM - CONFIGURATION")
        print("=" * 60)
        print(f"  Local Only Mode : {cls.LOCAL_ONLY_MODE}")
        print(f"  Audio Enabled   : {cls.ENABLE_AUDIO}")
        print(f"  Device          : {cls.DEVICE}")
        print(f"  Project Root    : {cls.PROJECT_ROOT}")
        print(f"  Dataset Root    : {cls.DATASET_ROOT}")
        print(f"  Image Dataset   : {cls.IMAGE_DATASET_ROOT}")
        print(f"  Training Runs   : {cls.TRAINING_RUNS_DIR}")
        print(f"  API Bind        : http://{cls.API_HOST}:{cls.API_PORT}")
        print("=" * 60)

        warnings = cls.validate()
        if warnings:
            print("\nWARNINGS:")
            for warning in warnings:
                print(f"  - {warning}")
        else:
            print("\nAll paths validated successfully.")
        print()


Config.ensure_dirs()
