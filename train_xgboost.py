from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from xgboost import XGBClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost on ViT-extracted features.")
    parser.add_argument("--x-path", type=str, default="features/X.npy", help="Path to X.npy features.")
    parser.add_argument("--y-path", type=str, default="features/y.npy", help="Path to y.npy labels.")
    parser.add_argument("--output-model", type=str, default="xgb_model.json", help="Output model path.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def compute_scale_pos_weight(y: np.ndarray) -> float:
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    if positives == 0:
        return 1.0
    return float(negatives / positives)


def main() -> None:
    args = parse_args()

    x_path = Path(args.x_path)
    y_path = Path(args.y_path)
    if not x_path.exists():
        raise FileNotFoundError(f"Features not found: {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Labels not found: {y_path}")

    X = np.load(x_path)
    y = np.load(y_path).astype(np.int64)

    if X.ndim != 2:
        raise ValueError(f"Expected 2D feature array, got shape={X.shape}")
    if y.ndim != 1:
        raise ValueError(f"Expected 1D label array, got shape={y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Sample mismatch: X has {X.shape[0]} rows, y has {y.shape[0]} rows")

    scale_pos_weight = compute_scale_pos_weight(y)

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=int(args.seed),
        n_jobs=-1,
    )
    model.fit(X, y)

    output_path = Path(args.output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output_path))

    y_pred = model.predict(X).astype(np.int64)
    acc = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    cm = confusion_matrix(y, y_pred, labels=[0, 1])

    print(f"Saved XGBoost model to: {output_path}")
    print(f"scale_pos_weight: {scale_pos_weight:.6f}")
    print(f"accuracy : {acc:.4f}")
    print(f"precision: {precision:.4f}")
    print(f"recall   : {recall:.4f}")
    print(f"f1-score : {f1:.4f}")
    print("confusion_matrix:")
    print(cm)


if __name__ == "__main__":
    main()

