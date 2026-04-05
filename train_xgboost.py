from __future__ import annotations

import argparse
from pathlib import Path

import model
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping
from sklearn.model_selection import train_test_split




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

    print("Loading features...")
    X = np.load(x_path)
    y = np.load(y_path).astype(np.int64)

    if X.ndim != 2:
        raise ValueError(f"Expected 2D feature array, got shape={X.shape}")
    if y.ndim != 1:
        raise ValueError(f"Expected 1D label array, got shape={y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Sample mismatch: X has {X.shape[0]} rows, y has {y.shape[0]} rows")

    # ✅ Normalize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # ✅ Train-test split (VERY IMPORTANT)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )

    # ✅ Handle class imbalance
    scale_pos_weight = compute_scale_pos_weight(y_train) *1.3

    # ✅ GPU check
    tree_method = "hist"
    try:
        import torch
        if torch.cuda.is_available():
            tree_method = "hist"
    except:
        pass

    print(f"Using tree_method: {tree_method}")

    # ✅ Model
    model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    n_estimators=600,
    max_depth=8,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    gamma=1.0,
    min_child_weight=3,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    tree_method=tree_method,
    early_stopping_rounds=50   # ✅ HERE (IMPORTANT)
)
    

    print("Training XGBoost...")
    # train
    # Train model
    model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    verbose=True
)

    # ✅ Predict on TEST (not train)
    #y_pred = model.predict(X_test).astype(np.int64)
# increasing presision 

    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs > 0.45).astype(int) # try 0.3–0.45
    # ✅ Metrics
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    # ✅ Save model
    output_path = Path(args.output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output_path))

    print("\n📊 RESULTS (REAL PERFORMANCE):")
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

