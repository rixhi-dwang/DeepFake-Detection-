from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import load_vit_checkpoint
from src.utils import setup_logger
from training.train_vit import FrameDataset, build_frame_manifest, build_transforms, split_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ViT + XGBoost ensemble inference/evaluation.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained ViT checkpoint.")
    parser.add_argument("--xgb-model", type=str, required=True, help="Path to trained XGBoost model JSON.")

    parser.add_argument("--frames-root", type=str, default=None, help="Frame root for split evaluation.")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--image-paths", nargs="*", default=None, help="Optional image path(s) for direct inference.")

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--max-images-per-video", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--xgb-weight", type=float, default=0.6)
    parser.add_argument("--vit-weight", type=float, default=0.4)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available.")
    return device


def load_xgb_model(model_path: str | Path) -> XGBClassifier:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"XGBoost model not found: {model_path}")
    model = XGBClassifier()
    model.load_model(str(model_path))
    return model


def normalize_weights(xgb_weight: float, vit_weight: float) -> Tuple[float, float]:
    total = float(xgb_weight + vit_weight)
    if total <= 0:
        raise ValueError("xgb_weight + vit_weight must be > 0")
    return float(xgb_weight / total), float(vit_weight / total)


def predict_ensemble_batch(
    vit_model,
    xgb_model: XGBClassifier,
    images: torch.Tensor,
    xgb_weight: float,
    vit_weight: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with torch.no_grad():
        logits = vit_model(images)
        vit_probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        features = vit_model.get_features(images).detach().cpu().numpy()

    xgb_probs = xgb_model.predict_proba(features)[:, 1]
    final_probs = (xgb_weight * xgb_probs) + (vit_weight * vit_probs)
    return final_probs, vit_probs, xgb_probs


def collect_split_samples(args: argparse.Namespace, logger) -> Sequence:
    if args.frames_root is None:
        raise ValueError("--frames-root is required for split evaluation.")

    manifest, _ = build_frame_manifest(
        frames_root=Path(args.frames_root),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
        max_images_per_video=args.max_images_per_video,
        seed=int(args.seed),
        logger=logger,
    )
    split_map = split_samples(manifest)
    if args.split == "all":
        samples = split_map["train"] + split_map["val"] + split_map["test"]
    else:
        samples = split_map[args.split]
    if not samples:
        raise RuntimeError(f"No samples found for split='{args.split}'.")
    return samples


def build_eval_loader(samples: Sequence, args: argparse.Namespace) -> DataLoader:
    _, eval_tf = build_transforms(image_size=int(args.image_size))
    dataset = FrameDataset(samples=samples, image_size=int(args.image_size), transform=eval_tf)
    return DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(args.num_workers) > 0,
    )


def evaluate_ensemble(
    vit_model,
    xgb_model: XGBClassifier,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    xgb_weight: float,
    vit_weight: float,
) -> dict:
    all_true = []
    all_pred = []
    all_final_prob = []

    vit_model.eval()
    for images, labels in tqdm(loader, desc="Evaluating ensemble"):
        images = images.to(device, non_blocking=True)
        labels_np = labels.detach().cpu().numpy().astype(np.int64)
        final_probs, _, _ = predict_ensemble_batch(
            vit_model=vit_model,
            xgb_model=xgb_model,
            images=images,
            xgb_weight=xgb_weight,
            vit_weight=vit_weight,
        )
        preds = (final_probs >= threshold).astype(np.int64)

        all_true.append(labels_np)
        all_pred.append(preds)
        all_final_prob.append(final_probs.astype(np.float32))

    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)
    y_prob = np.concatenate(all_final_prob, axis=0)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]),
        "num_samples": int(y_true.shape[0]),
        "mean_probability": float(np.mean(y_prob)),
    }
    return metrics


def print_metrics(metrics: dict) -> None:
    print(f"num_samples: {metrics['num_samples']}")
    print(f"accuracy : {metrics['accuracy']:.4f}")
    print(f"precision: {metrics['precision']:.4f}")
    print(f"recall   : {metrics['recall']:.4f}")
    print(f"f1-score : {metrics['f1']:.4f}")
    print("confusion_matrix:")
    print(metrics["confusion_matrix"])


def predict_images(
    vit_model,
    xgb_model: XGBClassifier,
    image_paths: Sequence[str],
    device: torch.device,
    image_size: int,
    batch_size: int,
    threshold: float,
    xgb_weight: float,
    vit_weight: float,
) -> List[dict]:
    _, eval_tf = build_transforms(image_size=image_size)

    valid_paths: List[str] = []
    tensors: List[torch.Tensor] = []
    for image_path in image_paths:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        with Image.open(path) as img:
            tensor = eval_tf(img.convert("RGB"))
        valid_paths.append(str(path))
        tensors.append(tensor)

    results: List[dict] = []
    for start in range(0, len(tensors), batch_size):
        batch_tensors = tensors[start : start + batch_size]
        batch_paths = valid_paths[start : start + batch_size]
        images = torch.stack(batch_tensors, dim=0).to(device, non_blocking=True)

        final_probs, vit_probs, xgb_probs = predict_ensemble_batch(
            vit_model=vit_model,
            xgb_model=xgb_model,
            images=images,
            xgb_weight=xgb_weight,
            vit_weight=vit_weight,
        )

        for index, image_path in enumerate(batch_paths):
            pred = int(final_probs[index] >= threshold)
            results.append(
                {
                    "image_path": image_path,
                    "prediction": pred,
                    "label": "FAKE" if pred == 1 else "REAL",
                    "final_probability_fake": float(final_probs[index]),
                    "vit_probability_fake": float(vit_probs[index]),
                    "xgb_probability_fake": float(xgb_probs[index]),
                }
            )

    return results


def main() -> None:
    args = parse_args()
    logger = setup_logger("ensemble")

    device = resolve_device(args.device)
    xgb_weight, vit_weight = normalize_weights(args.xgb_weight, args.vit_weight)

    vit_model, _ = load_vit_checkpoint(checkpoint_path=args.checkpoint, device=device, num_classes=2)
    xgb_model = load_xgb_model(args.xgb_model)

    has_eval_data = args.frames_root is not None
    has_images = bool(args.image_paths)
    if not has_eval_data and not has_images:
        raise ValueError("Provide either --frames-root for evaluation or --image-paths for direct inference.")

    if has_eval_data:
        samples = collect_split_samples(args=args, logger=logger)
        loader = build_eval_loader(samples=samples, args=args)
        metrics = evaluate_ensemble(
            vit_model=vit_model,
            xgb_model=xgb_model,
            loader=loader,
            device=device,
            threshold=float(args.threshold),
            xgb_weight=xgb_weight,
            vit_weight=vit_weight,
        )
        print_metrics(metrics)

    if has_images:
        predictions = predict_images(
            vit_model=vit_model,
            xgb_model=xgb_model,
            image_paths=args.image_paths or [],
            device=device,
            image_size=int(args.image_size),
            batch_size=int(args.batch_size),
            threshold=float(args.threshold),
            xgb_weight=xgb_weight,
            vit_weight=vit_weight,
        )
        for item in predictions:
            print(
                f"{item['image_path']} | {item['label']} | "
                f"final={item['final_probability_fake']:.4f} "
                f"(xgb={item['xgb_probability_fake']:.4f}, vit={item['vit_probability_fake']:.4f})"
            )


if __name__ == "__main__":
    main()

