"""
Train a Vision Transformer (ViT) on extracted deepfake frame images.

Expected input structure:
  <frames_root>/
    real/
      <video_id_1>/frame_000.jpg
      <video_id_2>/frame_000.jpg
    fake/
      <video_id_1>/frame_000.jpg
      <video_id_2>/frame_000.jpg

Key features:
  - Video-level split (prevents leakage from the same video across splits)
  - Optional per-video frame cap
  - Class balancing options (sampler and/or class-weighted loss)
  - Mixed precision training (AMP)
  - Checkpointing, resume support, and early stopping
  - Train/val/test metrics and artifact export
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import ensure_dir, setup_logger


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_TO_ID = {"real": 0, "fake": 1}
ID_TO_LABEL = {0: "real", 1: "fake"}


@dataclass(frozen=True)
class FrameSample:
    """One training sample."""

    image_path: str
    label: int
    label_name: str
    video_id: str
    split: str


class FrameDataset(Dataset):
    """Dataset over image file paths."""

    def __init__(
        self,
        samples: Sequence[FrameSample],
        image_size: int,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.samples = list(samples)
        self.image_size = image_size
        self.transform = transform
        self._to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        try:
            with Image.open(sample.image_path) as img:
                image = img.convert("RGB")
            if self.transform is not None:
                tensor = self.transform(image)
            else:
                tensor = self._to_tensor(image)
        except Exception:
            # Failsafe sample to avoid crashing full training for one bad image.
            image = Image.new("RGB", (self.image_size, self.image_size), color=(0, 0, 0))
            tensor = self.transform(image) if self.transform is not None else self._to_tensor(image)

        target = torch.tensor(sample.label, dtype=torch.long)
        return tensor, target


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_int_seed(text: str, base_seed: int) -> int:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return (int(digest, 16) + base_seed) % (2**32)


def resolve_label_dir(frames_root: Path, label_name: str) -> Optional[Path]:
    candidates = [
        frames_root / label_name,
        frames_root / label_name.upper(),
        frames_root / label_name.capitalize(),
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def list_image_files(video_dir: Path) -> List[Path]:
    files = [path for path in video_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    files.sort()
    return files


def compute_split_counts(total: int, train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[int, int, int]:
    if total <= 0:
        return 0, 0, 0
    if total == 1:
        return 1, 0, 0
    if total == 2:
        return 1, 1, 0

    ratios = np.array([train_ratio, val_ratio, test_ratio], dtype=np.float64)
    ratio_sum = float(ratios.sum())
    if ratio_sum <= 0:
        raise ValueError("train/val/test ratios must sum to a positive value.")
    ratios = ratios / ratio_sum

    raw = ratios * total
    counts = np.floor(raw).astype(int)
    remainder = int(total - counts.sum())

    if remainder > 0:
        frac = raw - counts
        order = np.argsort(frac)[::-1]
        for idx in order[:remainder]:
            counts[idx] += 1

    # Ensure each split gets at least one video when total >= 3.
    for split_idx in range(3):
        if counts[split_idx] == 0:
            donor = int(np.argmax(counts))
            if counts[donor] > 1:
                counts[donor] -= 1
                counts[split_idx] += 1

    return int(counts[0]), int(counts[1]), int(counts[2])


def split_video_dirs(
    video_dirs: Sequence[Path],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[Path]]:
    video_dirs = list(video_dirs)
    rng = random.Random(seed)
    rng.shuffle(video_dirs)

    n_train, n_val, n_test = compute_split_counts(
        total=len(video_dirs),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    train_videos = video_dirs[:n_train]
    val_videos = video_dirs[n_train : n_train + n_val]
    test_videos = video_dirs[n_train + n_val : n_train + n_val + n_test]

    return {"train": train_videos, "val": val_videos, "test": test_videos}


def sample_images_from_video(
    images: Sequence[Path],
    max_images_per_video: Optional[int],
    seed: int,
) -> List[Path]:
    images = list(images)
    if max_images_per_video is None or max_images_per_video <= 0:
        return images
    if len(images) <= max_images_per_video:
        return images

    rng = random.Random(seed)
    picked = rng.sample(images, k=max_images_per_video)
    picked.sort()
    return picked


def build_frame_manifest(
    frames_root: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    max_images_per_video: Optional[int],
    seed: int,
    logger,
) -> Tuple[List[FrameSample], Dict[str, Dict[str, int]]]:
    if not frames_root.exists():
        raise FileNotFoundError(f"Frames root not found: {frames_root}")

    manifest: List[FrameSample] = []
    summary: Dict[str, Dict[str, int]] = {
        "videos": {"train": 0, "val": 0, "test": 0},
        "images_real": {"train": 0, "val": 0, "test": 0},
        "images_fake": {"train": 0, "val": 0, "test": 0},
        "images_total": {"train": 0, "val": 0, "test": 0},
    }

    for label_name, label_id in LABEL_TO_ID.items():
        label_dir = resolve_label_dir(frames_root, label_name)
        if label_dir is None:
            logger.warning("Missing label directory for '%s' under %s", label_name, frames_root)
            continue

        video_dirs = sorted([path for path in label_dir.iterdir() if path.is_dir()])
        if not video_dirs:
            logger.warning("No video folders found for class '%s' in %s", label_name, label_dir)
            continue

        split_seed = stable_int_seed(str(label_dir), base_seed=seed)
        split_map = split_video_dirs(
            video_dirs=video_dirs,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=split_seed,
        )

        for split_name, split_video_dirs_list in split_map.items():
            summary["videos"][split_name] += len(split_video_dirs_list)

            for video_dir in split_video_dirs_list:
                images = list_image_files(video_dir)
                if not images:
                    continue

                sample_seed = stable_int_seed(str(video_dir), base_seed=seed)
                selected_images = sample_images_from_video(
                    images=images,
                    max_images_per_video=max_images_per_video,
                    seed=sample_seed,
                )

                for image_path in selected_images:
                    manifest.append(
                        FrameSample(
                            image_path=str(image_path),
                            label=label_id,
                            label_name=label_name,
                            video_id=video_dir.name,
                            split=split_name,
                        )
                    )

                if label_name == "real":
                    summary["images_real"][split_name] += len(selected_images)
                else:
                    summary["images_fake"][split_name] += len(selected_images)
                summary["images_total"][split_name] += len(selected_images)

    logger.info(
        "Manifest built. Train=%d Val=%d Test=%d images.",
        summary["images_total"]["train"],
        summary["images_total"]["val"],
        summary["images_total"]["test"],
    )
    return manifest, summary


def split_samples(manifest: Sequence[FrameSample]) -> Dict[str, List[FrameSample]]:
    split_map = {"train": [], "val": [], "test": []}
    for sample in manifest:
        if sample.split in split_map:
            split_map[sample.split].append(sample)
    return split_map


def build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.80, 1.00)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.04),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_tf, eval_tf


def create_dataloaders(
    split_map: Dict[str, List[FrameSample]],
    image_size: int,
    batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    balanced_sampling: bool,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_tf, eval_tf = build_transforms(image_size=image_size)

    train_dataset = FrameDataset(split_map["train"], image_size=image_size, transform=train_tf)
    val_dataset = FrameDataset(split_map["val"], image_size=image_size, transform=eval_tf)
    test_dataset = FrameDataset(split_map["test"], image_size=image_size, transform=eval_tf)

    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0

    sampler = None
    shuffle = True
    if balanced_sampling and len(split_map["train"]) > 0:
        label_counts = np.bincount([sample.label for sample in split_map["train"]], minlength=2)
        label_counts = np.maximum(label_counts, 1)
        sample_weights = [1.0 / label_counts[sample.label] for sample in split_map["train"]]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    return train_loader, val_loader, test_loader


def build_vit_model(pretrained: bool, dropout: float, logger) -> nn.Module:
    weights = None
    if pretrained:
        try:
            weights = models.ViT_B_16_Weights.IMAGENET1K_V1
        except Exception:
            try:
                weights = models.ViT_B_16_Weights.DEFAULT
            except Exception:
                weights = None

    try:
        model = models.vit_b_16(weights=weights)
    except Exception as exc:
        logger.warning(
            "Could not load pretrained ViT weights (%s). Falling back to random init.",
            exc,
        )
        model = models.vit_b_16(weights=None)

    in_features = model.heads.head.in_features
    if dropout > 0:
        model.heads.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 2),
        )
    else:
        model.heads.head = nn.Linear(in_features, 2)
    return model


def maybe_freeze_backbone(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if name.startswith("heads.head"):
            param.requires_grad = True
        else:
            param.requires_grad = False


def compute_class_weights(samples: Sequence[FrameSample]) -> torch.Tensor:
    counts = np.bincount([sample.label for sample in samples], minlength=2).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    total = float(counts.sum())
    weights = total / (2.0 * counts)
    return torch.tensor(weights, dtype=torch.float32)


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if y_true.size == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "specificity": 0.0,
            "balanced_accuracy": 0.0,
        }

    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    total = max(tp + tn + fp + fn, 1)
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    balanced_accuracy = 0.5 * (recall + specificity)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
    train: bool,
    grad_clip: float,
    use_amp: bool,
    epoch: int,
    phase_name: str,
) -> Dict[str, float]:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_items = 0
    all_targets: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []

    progress = tqdm(loader, desc=f"Epoch {epoch} [{phase_name}]", leave=False)
    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)

            if train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    if grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

        batch_size = int(targets.size(0))
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size

        preds = torch.argmax(logits, dim=1)
        all_targets.append(targets.detach().cpu().numpy())
        all_preds.append(preds.detach().cpu().numpy())

        running_loss = total_loss / max(total_items, 1)
        progress.set_postfix(loss=f"{running_loss:.4f}")

    if total_items == 0:
        metrics = binary_metrics(np.array([], dtype=np.int64), np.array([], dtype=np.int64))
        metrics["loss"] = 0.0
        return metrics

    y_true = np.concatenate(all_targets, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    metrics = binary_metrics(y_true=y_true, y_pred=y_pred)
    metrics["loss"] = float(total_loss / total_items)
    return metrics


def save_manifest_csv(samples: Sequence[FrameSample], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "label", "label_name", "video_id", "image_path"],
        )
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "split": sample.split,
                    "label": sample.label,
                    "label_name": sample.label_name,
                    "video_id": sample.video_id,
                    "image_path": sample.image_path,
                }
            )


def save_history_csv(history: Sequence[Dict[str, float]], path: Path) -> None:
    ensure_dir(path.parent)
    if not history:
        return
    fieldnames = list(history[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def is_better(metric_name: str, current: float, best: float) -> bool:
    if metric_name == "loss":
        return current < best
    return current > best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ViT on extracted deepfake frame images with video-level split."
    )
    parser.add_argument("--frames-root", type=str, default="data/frames")
    parser.add_argument("--output-dir", type=str, default="training_runs")
    parser.add_argument("--run-name", type=str, default=None)

    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--scheduler", choices=["cosine", "none"], default="cosine")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--max-images-per-video", type=int, default=None)

    parser.add_argument("--balanced-sampling", action="store_true")
    parser.add_argument("--class-weighted-loss", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")

    parser.add_argument("--selection-metric", choices=["f1", "accuracy", "balanced_accuracy", "loss"], default="f1")
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=1)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help="e.g. cuda, cuda:0, cpu")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision when running on CUDA")
    parser.add_argument("--resume-checkpoint", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Build splits/manifests and exit before training")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"vit_frames_{timestamp}"
    run_dir = Path(args.output_dir) / run_name
    checkpoints_dir = run_dir / "checkpoints"
    ensure_dir(checkpoints_dir)

    logger = setup_logger(
        name="train_vit",
        log_file=str(run_dir / "train.log"),
    )

    frames_root = Path(args.frames_root)
    logger.info("Frames root: %s", frames_root)
    logger.info("Run dir    : %s", run_dir)

    manifest, split_summary = build_frame_manifest(
        frames_root=frames_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        max_images_per_video=args.max_images_per_video,
        seed=args.seed,
        logger=logger,
    )
    if not manifest:
        raise RuntimeError("No samples found. Check --frames-root and extracted image folders.")

    split_map = split_samples(manifest)
    if not split_map["train"] or not split_map["val"]:
        raise RuntimeError("Train/val split is empty. Add more videos or adjust split ratios.")

    with (run_dir / "split_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(split_summary, handle, indent=2)
    save_manifest_csv(manifest, run_dir / "samples_manifest.csv")

    config_dump = vars(args).copy()
    config_dump["timestamp"] = timestamp
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config_dump, handle, indent=2)

    logger.info(
        "Dataset summary | train=%d val=%d test=%d images",
        len(split_map["train"]),
        len(split_map["val"]),
        len(split_map["test"]),
    )

    if args.dry_run:
        logger.info("Dry run complete. Artifacts written to %s", run_dir)
        return

    train_loader, val_loader, test_loader = create_dataloaders(
        split_map=split_map,
        image_size=args.image_size,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=max(0, int(args.num_workers)),
        balanced_sampling=args.balanced_sampling,
    )

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training device: %s", device)

    model = build_vit_model(pretrained=not args.no_pretrained, dropout=args.dropout, logger=logger)
    if args.freeze_backbone:
        maybe_freeze_backbone(model)
        logger.info("Backbone frozen: only head parameters are trainable.")
    model.to(device)

    class_weights = None
    if args.class_weighted_loss:
        class_weights = compute_class_weights(split_map["train"]).to(device)
        logger.info("Using class-weighted loss: %s", class_weights.tolist())

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=float(args.label_smoothing),
    )
    optimizer = AdamW(
        params=[param for param in model.parameters() if param.requires_grad],
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, int(args.epochs)), eta_min=float(args.min_lr))

    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch = 1
    history: List[Dict[str, float]] = []
    metric_name = args.selection_metric
    best_metric = float("inf") if metric_name == "loss" else float("-inf")
    best_epoch = 0
    best_checkpoint_path = checkpoints_dir / "best.pt"
    patience_counter = 0

    if args.resume_checkpoint:
        checkpoint_path = Path(args.resume_checkpoint)
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            if scheduler is not None and checkpoint.get("scheduler_state") is not None:
                scheduler.load_state_dict(checkpoint["scheduler_state"])
            if checkpoint.get("scaler_state") is not None and use_amp:
                scaler.load_state_dict(checkpoint["scaler_state"])
            start_epoch = int(checkpoint.get("epoch", 0)) + 1
            best_metric = float(checkpoint.get("best_metric", best_metric))
            best_epoch = int(checkpoint.get("best_epoch", best_epoch))
            logger.info("Resumed from checkpoint: %s (next epoch=%d)", checkpoint_path, start_epoch)
        else:
            logger.warning("Resume checkpoint not found: %s", checkpoint_path)

    logger.info("Starting training for %d epochs", int(args.epochs))
    for epoch in range(start_epoch, int(args.epochs) + 1):
        epoch_start = time.time()

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler if use_amp else None,
            device=device,
            train=True,
            grad_clip=float(args.grad_clip),
            use_amp=use_amp,
            epoch=epoch,
            phase_name="train",
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            scaler=None,
            device=device,
            train=False,
            grad_clip=0.0,
            use_amp=use_amp,
            epoch=epoch,
            phase_name="val",
        )

        if scheduler is not None:
            scheduler.step()

        elapsed = round(time.time() - epoch_start, 2)
        lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "lr": float(lr),
            "seconds": elapsed,
            "train_loss": float(train_metrics["loss"]),
            "train_accuracy": float(train_metrics["accuracy"]),
            "train_precision": float(train_metrics["precision"]),
            "train_recall": float(train_metrics["recall"]),
            "train_f1": float(train_metrics["f1"]),
            "val_loss": float(val_metrics["loss"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_precision": float(val_metrics["precision"]),
            "val_recall": float(val_metrics["recall"]),
            "val_f1": float(val_metrics["f1"]),
            "val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
        }
        history.append(row)

        logger.info(
            "Epoch %d/%d | %.2fs | train_loss=%.4f train_f1=%.4f | val_loss=%.4f val_f1=%.4f val_acc=%.4f",
            epoch,
            int(args.epochs),
            elapsed,
            row["train_loss"],
            row["train_f1"],
            row["val_loss"],
            row["val_f1"],
            row["val_accuracy"],
        )

        current_metric = float(row[f"val_{metric_name}"] if metric_name != "loss" else row["val_loss"])
        if is_better(metric_name, current_metric, best_metric):
            best_metric = current_metric
            best_epoch = epoch
            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                "scaler_state": scaler.state_dict() if use_amp else None,
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "label_to_id": LABEL_TO_ID,
                "id_to_label": ID_TO_LABEL,
                "args": vars(args),
            }
            torch.save(checkpoint, best_checkpoint_path)
            logger.info("New best checkpoint at epoch %d (%s=%.6f)", best_epoch, metric_name, best_metric)
            patience_counter = 0
        else:
            patience_counter += 1

        if int(args.save_every) > 0 and epoch % int(args.save_every) == 0:
            epoch_ckpt = checkpoints_dir / f"epoch_{epoch:03d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                    "scaler_state": scaler.state_dict() if use_amp else None,
                    "best_metric": best_metric,
                    "best_epoch": best_epoch,
                },
                epoch_ckpt,
            )

        if int(args.early_stopping_patience) > 0 and patience_counter >= int(args.early_stopping_patience):
            logger.info(
                "Early stopping triggered after %d epochs without improvement.",
                patience_counter,
            )
            break

    final_checkpoint_path = checkpoints_dir / "final.pt"
    torch.save(
        {
            "epoch": history[-1]["epoch"] if history else 0,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state": scaler.state_dict() if use_amp else None,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "label_to_id": LABEL_TO_ID,
            "id_to_label": ID_TO_LABEL,
            "args": vars(args),
        },
        final_checkpoint_path,
    )

    # Load best checkpoint before test evaluation if available.
    if best_checkpoint_path.exists():
        best_state = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(best_state["model_state"])

    test_metrics = run_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        optimizer=None,
        scaler=None,
        device=device,
        train=False,
        grad_clip=0.0,
        use_amp=use_amp,
        epoch=best_epoch if best_epoch > 0 else 0,
        phase_name="test",
    )

    with (run_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    save_history_csv(history, run_dir / "history.csv")

    final_summary = {
        "best_epoch": best_epoch,
        "selection_metric": metric_name,
        "best_metric_value": best_metric,
        "num_epochs_ran": len(history),
        "train_images": len(split_map["train"]),
        "val_images": len(split_map["val"]),
        "test_images": len(split_map["test"]),
        "split_summary": split_summary,
        "test_metrics": test_metrics,
        "paths": {
            "run_dir": str(run_dir),
            "best_checkpoint": str(best_checkpoint_path),
            "final_checkpoint": str(final_checkpoint_path),
            "manifest_csv": str(run_dir / "samples_manifest.csv"),
            "history_json": str(run_dir / "history.json"),
            "history_csv": str(run_dir / "history.csv"),
            "train_log": str(run_dir / "train.log"),
        },
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(final_summary, handle, indent=2)

    logger.info("Training complete.")
    logger.info("Best epoch: %d | Best %s: %.6f", best_epoch, metric_name, best_metric)
    logger.info(
        "Test metrics | loss=%.4f acc=%.4f precision=%.4f recall=%.4f f1=%.4f",
        test_metrics["loss"],
        test_metrics["accuracy"],
        test_metrics["precision"],
        test_metrics["recall"],
        test_metrics["f1"],
    )
    logger.info("Artifacts written to: %s", run_dir)


if __name__ == "__main__":
    parsed_args = parse_args()
    main(parsed_args)
