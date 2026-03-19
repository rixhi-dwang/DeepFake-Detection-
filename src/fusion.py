"""
Multimodal Fusion System for Deepfake Detection.

Combines video and audio predictions using:
  1. Rule-based fusion (weighted average)
  2. ML-based fusion (Logistic Regression)

The fusion module takes prediction scores from both modalities
and produces a unified final prediction.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import setup_logger

logger = setup_logger("fusion")


def rule_based_fusion(
    video_score: float,
    audio_score: float,
    video_weight: float = 0.7,
    audio_weight: float = 0.3,
    threshold: float = 0.5,
) -> Dict:
    """
    Rule-based fusion using weighted average of fake probabilities.

    Args:
        video_score: Fake probability from video model (0.0 to 1.0).
        audio_score: Fake probability from audio model (0.0 to 1.0).
        video_weight: Weight for video prediction.
        audio_weight: Weight for audio prediction.
        threshold: Decision threshold for FAKE classification.

    Returns:
        Dict with fused prediction.
    """
    # Normalize weights
    total_weight = video_weight + audio_weight
    v_w = video_weight / total_weight
    a_w = audio_weight / total_weight

    # Weighted average of fake probabilities
    fused_fake_prob = (video_score * v_w) + (audio_score * a_w)
    fused_real_prob = 1.0 - fused_fake_prob

    label = "FAKE" if fused_fake_prob >= threshold else "REAL"
    confidence = max(fused_fake_prob, fused_real_prob)

    return {
        "label": label,
        "confidence": round(float(confidence), 4),
        "fake_probability": round(float(fused_fake_prob), 4),
        "real_probability": round(float(fused_real_prob), 4),
        "method": "rule_based",
        "weights": {"video": round(v_w, 2), "audio": round(a_w, 2)},
    }


class MLFusion:
    """
    ML-based fusion using Logistic Regression.

    Trains on video + audio prediction scores to learn optimal
    combination weights.
    """

    def __init__(self):
        from sklearn.linear_model import LogisticRegression

        self.model = LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=42,
        )
        self.is_trained = False

    def train(
        self,
        video_scores: np.ndarray,
        audio_scores: np.ndarray,
        labels: np.ndarray,
    ):
        """
        Train the fusion model.

        Args:
            video_scores: Array of video fake probabilities.
            audio_scores: Array of audio fake probabilities.
            labels: Array of ground truth labels (0=REAL, 1=FAKE).
        """
        X = np.column_stack([video_scores, audio_scores])
        self.model.fit(X, labels)
        self.is_trained = True

        train_acc = self.model.score(X, labels)
        logger.info(f"ML Fusion model trained — Accuracy: {train_acc:.4f}")

    def predict(self, video_score: float, audio_score: float) -> Dict:
        """
        Predict using the trained fusion model.

        Args:
            video_score: Fake probability from video model.
            audio_score: Fake probability from audio model.

        Returns:
            Dict with fused prediction.
        """
        if not self.is_trained:
            logger.warning(
                "ML Fusion model not trained — falling back to rule-based"
            )
            return rule_based_fusion(video_score, audio_score)

        X = np.array([[video_score, audio_score]])
        probs = self.model.predict_proba(X)[0]
        pred = self.model.predict(X)[0]

        label = "FAKE" if pred == 1 else "REAL"

        return {
            "label": label,
            "confidence": round(float(max(probs)), 4),
            "fake_probability": round(float(probs[1]), 4),
            "real_probability": round(float(probs[0]), 4),
            "method": "ml_logistic_regression",
        }

    def save(self, path: str):
        """Save trained model to disk."""
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        logger.info(f"ML Fusion model saved to {path}")

    def load(self, path: str):
        """Load trained model from disk."""
        if not Path(path).exists():
            raise FileNotFoundError(f"Fusion model not found: {path}")

        with open(path, "rb") as f:
            self.model = pickle.load(f)
        self.is_trained = True
        logger.info(f"ML Fusion model loaded from {path}")


def run_fusion(
    video_result: Dict,
    audio_result: Dict,
    method: str = "rule_based",
    ml_fusion: Optional[MLFusion] = None,
) -> Dict:
    """
    Run fusion on video and audio pipeline results.

    Handles edge cases where one modality may have failed.

    Args:
        video_result: Output from run_video_pipeline().
        audio_result: Output from run_audio_pipeline().
        method: "rule_based" or "ml_based".
        ml_fusion: Pre-trained MLFusion instance (required for ml_based).

    Returns:
        Complete fusion result dict.
    """
    from config import Config

    # Extract fake probabilities from each modality
    video_pred = video_result.get("prediction", {})
    audio_pred = audio_result.get("prediction", {})

    video_fake_prob = video_pred.get("fake_probability", 0.5)
    audio_fake_prob = audio_pred.get("fake_probability", 0.5)

    video_status = video_result.get("status", "error")
    audio_status = audio_result.get("status", "error")

    # ── Handle modality failures ────────────────────────────────────
    if video_status == "error" and audio_status == "error":
        return {
            "label": "UNKNOWN",
            "confidence": 0.0,
            "fake_probability": 0.0,
            "real_probability": 0.0,
            "method": "none",
            "note": "Both video and audio analysis failed",
            "video_status": video_status,
            "audio_status": audio_status,
        }

    if video_status == "error":
        # Only audio available — use audio prediction directly
        return {
            **audio_pred,
            "method": "audio_only",
            "note": "Video analysis failed — using audio prediction only",
            "video_status": video_status,
            "audio_status": audio_status,
        }

    if audio_status in ("error", "partial"):
        # Only video available or audio is uncertain
        if audio_status == "partial":
            # Audio had no track — rely primarily on video
            fused = rule_based_fusion(
                video_fake_prob, 0.5,
                video_weight=0.95, audio_weight=0.05,
            )
            fused["note"] = "Audio unavailable — primarily using video prediction"
        else:
            fused = {
                **video_pred,
                "method": "video_only",
                "note": "Audio analysis failed — using video prediction only",
            }
        fused["video_status"] = video_status
        fused["audio_status"] = audio_status
        return fused

    # ── Both modalities available ───────────────────────────────────
    if method == "ml_based" and ml_fusion is not None and ml_fusion.is_trained:
        fused = ml_fusion.predict(video_fake_prob, audio_fake_prob)
    else:
        fused = rule_based_fusion(
            video_fake_prob,
            audio_fake_prob,
            video_weight=Config.FUSION_VIDEO_WEIGHT,
            audio_weight=Config.FUSION_AUDIO_WEIGHT,
            threshold=Config.FUSION_THRESHOLD,
        )

    fused["video_status"] = video_status
    fused["audio_status"] = audio_status
    fused["video_prediction"] = video_pred
    fused["audio_prediction"] = audio_pred

    logger.info(
        f"Fusion result ({fused.get('method', 'unknown')}): "
        f"{fused['label']} (confidence: {fused['confidence']:.4f})"
    )

    return fused
