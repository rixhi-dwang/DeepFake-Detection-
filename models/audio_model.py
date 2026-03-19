"""
Audio deepfake model utilities.

Audio is optional and local-only. No remote model downloads are allowed.
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import setup_logger, softmax

logger = setup_logger("audio_model")


class AudioDeepfakeClassifier(nn.Module):
    """Classifier head on top of Wav2Vec2 embeddings."""

    def __init__(self, input_dim: int = 768, num_classes: int = 2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


_audio_feature_extractor = None
_audio_processor = None
_audio_classifier = None


def _has_audio_backbone_files(model_dir: Path) -> bool:
    if not model_dir.exists():
        return False
    if not (model_dir / "config.json").exists():
        return False
    return (model_dir / "model.safetensors").exists() or (model_dir / "pytorch_model.bin").exists()


def load_audio_model(config=None) -> Tuple:
    """
    Load local audio backbone + local classifier weights.
    """
    global _audio_feature_extractor, _audio_processor, _audio_classifier

    if _audio_feature_extractor is not None:
        return _audio_feature_extractor, _audio_processor, _audio_classifier

    if config is None:
        from config import Config

        config = Config

    if not config.ENABLE_AUDIO:
        raise RuntimeError(
            "Audio pipeline is disabled (DEEPFAKE_ENABLE_AUDIO=0)."
        )

    from transformers import Wav2Vec2Model, Wav2Vec2Processor

    model_dir = config.AUDIO_MODEL_DIR
    if not _has_audio_backbone_files(model_dir):
        raise RuntimeError(
            "Local audio backbone not found. Expected config.json and model weights in "
            f"{model_dir}"
        )
    if not config.AUDIO_CLASSIFIER_PATH.exists():
        raise RuntimeError(
            f"Local audio classifier weights not found: {config.AUDIO_CLASSIFIER_PATH}"
        )

    logger.info(f"Loading local audio backbone from: {model_dir}")
    _audio_processor = Wav2Vec2Processor.from_pretrained(
        str(model_dir),
        local_files_only=True,
    )
    _audio_feature_extractor = Wav2Vec2Model.from_pretrained(
        str(model_dir),
        local_files_only=True,
    )
    _audio_feature_extractor = _audio_feature_extractor.to(config.DEVICE)
    _audio_feature_extractor.eval()

    classifier_state = torch.load(
        str(config.AUDIO_CLASSIFIER_PATH),
        map_location=config.DEVICE,
    )
    _audio_classifier = AudioDeepfakeClassifier(input_dim=768, num_classes=2).to(config.DEVICE)
    _audio_classifier.load_state_dict(classifier_state)
    _audio_classifier.eval()

    logger.info("Audio model loaded successfully (local-only)")
    return _audio_feature_extractor, _audio_processor, _audio_classifier


def extract_audio_features(audio_array: np.ndarray, sample_rate: int = 16000, device=None) -> torch.Tensor:
    """Extract mean-pooled Wav2Vec2 features."""
    if device is None:
        from config import Config

        device = Config.DEVICE

    feature_extractor, processor, _ = load_audio_model()

    inputs = processor(
        audio_array,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = feature_extractor(**inputs)
        hidden_states = outputs.last_hidden_state

    return hidden_states.mean(dim=1).squeeze(0)


def predict_audio(audio_array: np.ndarray, sample_rate: int = 16000) -> Dict:
    """Predict REAL/FAKE for an audio signal."""
    if audio_array is None or len(audio_array) == 0:
        return {
            "label": "UNKNOWN",
            "confidence": 0.0,
            "fake_probability": 0.0,
            "real_probability": 0.0,
            "error": "Empty audio input",
        }

    try:
        features = extract_audio_features(audio_array, sample_rate)
        _, _, classifier = load_audio_model()
        with torch.no_grad():
            logits = classifier(features.unsqueeze(0))

        logits_np = logits.detach().cpu().numpy()[0]
        probs = softmax(logits_np)
        pred_idx = int(np.argmax(probs))
        label = "FAKE" if pred_idx == 1 else "REAL"

        return {
            "label": label,
            "confidence": float(probs[pred_idx]),
            "fake_probability": float(probs[1]),
            "real_probability": float(probs[0]),
        }
    except Exception as exc:
        logger.error(f"Audio prediction failed: {exc}")
        return {
            "label": "UNKNOWN",
            "confidence": 0.0,
            "fake_probability": 0.0,
            "real_probability": 0.0,
            "error": str(exc),
        }
