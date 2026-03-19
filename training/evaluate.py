"""
Evaluation module for the Deepfake Detection System.

Computes:
  - Accuracy, Precision, Recall, F1-score
  - Confusion Matrix
  - Classification Report
  - Per-class statistics

Can evaluate on:
  1. A set of video files with known labels
  2. Pre-computed predictions vs ground truth labels
"""

import os
import json
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import setup_logger

logger = setup_logger("evaluator")


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Compute classification metrics.

    Args:
        y_true: Ground truth labels (0=REAL, 1=FAKE).
        y_pred: Predicted labels.
        class_names: Names for each class.

    Returns:
        Dict with accuracy, precision, recall, f1, and confusion matrix.
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        classification_report,
    )

    if class_names is None:
        class_names = ["REAL", "FAKE"]

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "total_samples": len(y_true),
        "correct_predictions": int((y_true == y_pred).sum()),
    }

    return metrics


def print_evaluation_report(metrics: Dict):
    """Print a formatted evaluation report."""
    print("\n" + "=" * 60)
    print("  DEEPFAKE DETECTION — EVALUATION REPORT")
    print("=" * 60)
    print(f"  Total Samples    : {metrics['total_samples']}")
    print(f"  Correct          : {metrics['correct_predictions']}")
    print(f"  Accuracy         : {metrics['accuracy']:.4f}")
    print(f"  Precision (FAKE) : {metrics['precision']:.4f}")
    print(f"  Recall (FAKE)    : {metrics['recall']:.4f}")
    print(f"  F1 Score (FAKE)  : {metrics['f1_score']:.4f}")
    print()

    cm = metrics["confusion_matrix"]
    print("  Confusion Matrix:")
    print(f"                 Predicted REAL  Predicted FAKE")
    print(f"  Actual REAL      {cm[0][0]:>8}        {cm[0][1]:>8}")
    print(f"  Actual FAKE      {cm[1][0]:>8}        {cm[1][1]:>8}")
    print()

    report = metrics.get("classification_report", {})
    if report:
        print("  Per-Class Report:")
        for cls_name in ["REAL", "FAKE"]:
            cls_data = report.get(cls_name, {})
            print(
                f"    {cls_name:>8}: "
                f"precision={cls_data.get('precision', 0):.4f}  "
                f"recall={cls_data.get('recall', 0):.4f}  "
                f"f1={cls_data.get('f1-score', 0):.4f}  "
                f"support={cls_data.get('support', 0)}"
            )

    print("=" * 60)


def evaluate_on_videos(
    video_paths: List[str],
    labels: List[int],
    batch_size: int = 8,
) -> Dict:
    """
    Evaluate the video pipeline on a set of videos with known labels.

    Args:
        video_paths: List of video file paths.
        labels: Ground truth labels (0=REAL, 1=FAKE).
        batch_size: Not used directly here (per-video processing).

    Returns:
        Evaluation metrics dict.
    """
    from src.video_pipeline import run_video_pipeline

    predictions = []
    errors = 0

    logger.info(f"Evaluating on {len(video_paths)} videos...")

    for i, (video_path, true_label) in enumerate(zip(video_paths, labels)):
        try:
            result = run_video_pipeline(video_path)

            if result["status"] == "success":
                pred_label = result["prediction"]["label"]
                pred_int = 1 if pred_label == "FAKE" else 0
                predictions.append(pred_int)
            else:
                # Default to REAL if pipeline fails
                predictions.append(0)
                errors += 1
                logger.warning(f"Pipeline error on {video_path}: {result.get('error')}")

        except Exception as e:
            predictions.append(0)
            errors += 1
            logger.error(f"Exception on {video_path}: {e}")

        if (i + 1) % 10 == 0:
            logger.info(f"Evaluated {i + 1}/{len(video_paths)} videos...")

    logger.info(f"Evaluation complete. Errors: {errors}/{len(video_paths)}")

    metrics = compute_metrics(labels, predictions)
    metrics["pipeline_errors"] = errors

    return metrics


def save_evaluation_results(metrics: Dict, output_path: str):
    """Save evaluation results to a JSON file."""
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable = json.loads(json.dumps(metrics, default=convert))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)

    logger.info(f"Evaluation results saved to {output_path}")


def plot_confusion_matrix(
    metrics: Dict,
    output_path: Optional[str] = None,
    class_names: Optional[List[str]] = None,
):
    """
    Plot and optionally save a confusion matrix visualization.

    Args:
        metrics: Metrics dict with confusion_matrix key.
        output_path: Path to save the plot image.
        class_names: Class names for axis labels.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
        import seaborn as sns

        if class_names is None:
            class_names = ["REAL", "FAKE"]

        cm = np.array(metrics["confusion_matrix"])

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_title("Deepfake Detection — Confusion Matrix", fontsize=14)

        plt.tight_layout()

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150)
            logger.info(f"Confusion matrix plot saved to {output_path}")
        else:
            plt.show()

        plt.close()

    except ImportError:
        logger.warning("matplotlib/seaborn not available — skipping plot")


if __name__ == "__main__":
    """Quick evaluation on a small sample of FF++ videos."""
    from config import Config
    from training.dataset import build_dataset

    Config.summary()

    # Build a small evaluation set
    _, _, val_paths, val_labels = build_dataset(
        config=Config,
        fake_methods=["Deepfakes"],
        max_videos_per_class=10,  # Small sample for quick test
    )

    print(f"\nEvaluating on {len(val_paths)} validation videos...\n")

    metrics = evaluate_on_videos(val_paths, val_labels)
    print_evaluation_report(metrics)

    # Save results
    output_path = str(Config.LOGS_DIR / "evaluation_results.json")
    save_evaluation_results(metrics, output_path)

    # Plot confusion matrix
    plot_path = str(Config.LOGS_DIR / "confusion_matrix.png")
    plot_confusion_matrix(metrics, output_path=plot_path)
