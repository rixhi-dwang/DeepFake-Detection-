"""
Training entry point for local deepfake model fine-tuning.

Supports two data modes:
  1) video  - on-the-fly frame/face extraction from FF++ videos
  2) images - pre-converted image folders (from prepare_image_dataset.py)
"""

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import setup_logger

logger = setup_logger("trainer")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    epoch: int,
    max_grad_norm: float,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_index, batch in enumerate(loader):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += float(loss.item())
        preds = outputs.logits.argmax(dim=-1)
        correct += int((preds == labels).sum().item())
        total += int(labels.size(0))

        if (batch_index + 1) % 10 == 0:
            avg_loss = total_loss / (batch_index + 1)
            acc = correct / max(total, 1)
            logger.info(
                f"Epoch {epoch} | Batch {batch_index + 1}/{len(loader)} | Loss {avg_loss:.4f} | Acc {acc:.4f}"
            )

    return {
        "loss": round(total_loss / max(len(loader), 1), 4),
        "accuracy": round(correct / max(total, 1), 4),
    }


@torch.no_grad()
def validate(model, loader, device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(pixel_values=pixel_values, labels=labels)
        total_loss += float(outputs.loss.item())
        preds = outputs.logits.argmax(dim=-1)
        correct += int((preds == labels).sum().item())
        total += int(labels.size(0))

    return {
        "loss": round(total_loss / max(len(loader), 1), 4),
        "accuracy": round(correct / max(total, 1), 4),
    }


def maybe_freeze_backbone(model):
    frozen = 0
    trainable = 0

    if hasattr(model, "vit"):
        for param in model.vit.parameters():
            param.requires_grad = False
    elif hasattr(model, "base_model"):
        for param in model.base_model.parameters():
            param.requires_grad = False

    for param in model.parameters():
        if param.requires_grad:
            trainable += param.numel()
        else:
            frozen += param.numel()

    logger.info(f"Backbone frozen. Frozen params={frozen}, trainable params={trainable}")


def create_loaders(args, processor):
    from config import Config

    if args.data_mode == "video":
        from training.dataset import create_dataloaders

        return create_dataloaders(
            config=Config,
            processor=processor,
            fake_methods=args.methods,
            max_videos=args.max_videos,
            train_ratio=args.train_ratio,
            seed=args.seed,
            frames_per_video=args.frames_per_video,
            frame_sample_rate=args.frame_sample_rate,
            face_confidence=args.face_confidence,
            train_batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
        )

    from training.image_dataset import create_image_dataloaders

    image_root = Path(args.image_dataset_dir)
    if not image_root.exists():
        raise FileNotFoundError(
            f"Image dataset path not found: {image_root}. "
            "Run training/prepare_image_dataset.py first."
        )

    return create_image_dataloaders(
        image_root=image_root,
        processor=processor,
        train_batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        max_images_per_class=args.max_images_per_class,
        seed=args.seed,
        target_size=args.image_size,
        use_augmentation=not args.no_augment,
    )


def run_training(args):
    from config import Config
    from models.video_model import load_video_model

    set_seed(args.seed)
    Config.summary()

    logger.info("=" * 70)
    logger.info("LOCAL TRAINING START")
    logger.info("=" * 70)
    logger.info(f"Data mode           : {args.data_mode}")
    logger.info(f"Epochs              : {args.epochs}")
    logger.info(f"Batch size          : {args.batch_size}")
    logger.info(f"Eval batch size     : {args.eval_batch_size}")
    logger.info(f"Learning rate       : {args.lr}")
    logger.info(f"Weight decay        : {args.weight_decay}")
    logger.info(f"Gradient clip       : {args.max_grad_norm}")
    logger.info(f"Device              : {Config.DEVICE}")
    logger.info(f"Output root         : {args.output_dir}")
    logger.info("=" * 70)

    model, processor = load_video_model(model_dir=args.model_dir)
    if args.freeze_backbone:
        maybe_freeze_backbone(model)

    train_loader, val_loader = create_loaders(args=args, processor=processor)
    if len(train_loader.dataset) == 0 or len(val_loader.dataset) == 0:
        raise RuntimeError("Empty train or val dataset. Check paths and conversion output.")

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    run_name = args.run_name or f"{args.data_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.output_dir) / run_name
    best_dir = run_dir / "best_model"
    final_dir = run_dir / "final_model"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    config_dump = vars(args).copy()
    config_dump["device"] = str(Config.DEVICE)
    with open(run_dir / "training_config.json", "w", encoding="utf-8") as handle:
        json.dump(config_dump, handle, indent=2)

    best_val_acc = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=Config.DEVICE,
            epoch=epoch,
            max_grad_norm=args.max_grad_norm,
        )
        val_metrics = validate(
            model=model,
            loader=val_loader,
            device=Config.DEVICE,
        )
        if scheduler is not None:
            scheduler.step()

        elapsed = round(time.time() - start, 2)
        lr = optimizer.param_groups[0]["lr"]

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "learning_rate": lr,
            "epoch_seconds": elapsed,
        }
        history.append(record)

        logger.info(
            f"Epoch {epoch}/{args.epochs} ({elapsed}s) | "
            f"Train loss {train_metrics['loss']:.4f} acc {train_metrics['accuracy']:.4f} | "
            f"Val loss {val_metrics['loss']:.4f} acc {val_metrics['accuracy']:.4f} | "
            f"LR {lr:.7f}"
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            model.save_pretrained(str(best_dir))
            processor.save_pretrained(str(best_dir))
            logger.info(f"Saved new best model to {best_dir} (val_acc={best_val_acc:.4f})")

    model.save_pretrained(str(final_dir))
    processor.save_pretrained(str(final_dir))

    with open(run_dir / "training_history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    logger.info("=" * 70)
    logger.info(f"Training complete. Best val accuracy: {best_val_acc:.4f}")
    logger.info(f"Run artifacts: {run_dir}")
    logger.info("=" * 70)

    return {
        "best_val_accuracy": best_val_acc,
        "run_dir": str(run_dir),
        "history": history,
    }


def parse_args():
    from config import Config

    parser = argparse.ArgumentParser(description="Train local deepfake detection model")

    # Core mode/datasets
    parser.add_argument(
        "--data_mode",
        choices=["video", "images"],
        default="video",
        help="Training source: FF++ videos or converted images",
    )
    parser.add_argument(
        "--image_dataset_dir",
        type=str,
        default=str(Config.IMAGE_DATASET_ROOT),
        help="Root folder for converted images (train/val/REAL|FAKE)",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Optional local model directory to start from",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Config.TRAINING_RUNS_DIR),
        help="Root output directory for run artifacts",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional run name (default: auto timestamp)",
    )

    # Optimization
    parser.add_argument("--epochs", type=int, default=Config.TRAIN_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=Config.TRAIN_BATCH_SIZE)
    parser.add_argument("--eval_batch_size", type=int, default=Config.EVAL_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=Config.LEARNING_RATE)
    parser.add_argument("--min_lr", type=float, default=1e-7)
    parser.add_argument("--weight_decay", type=float, default=Config.WEIGHT_DECAY)
    parser.add_argument("--max_grad_norm", type=float, default=Config.MAX_GRAD_NORM)
    parser.add_argument("--scheduler", choices=["cosine", "none"], default="cosine")
    parser.add_argument("--freeze_backbone", action="store_true")

    # Data controls
    parser.add_argument("--seed", type=int, default=Config.RANDOM_SEED)
    parser.add_argument("--num_workers", type=int, default=Config.NUM_WORKERS)
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--image_size", type=int, default=Config.FACE_IMAGE_SIZE)

    # Video-mode controls
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["Deepfakes", "Face2Face", "FaceSwap", "FaceShifter", "NeuralTextures"],
        help="Fake methods used in FF++ video mode",
    )
    parser.add_argument("--max_videos", type=int, default=None, help="Cap videos per class")
    parser.add_argument("--train_ratio", type=float, default=Config.TRAIN_RATIO)
    parser.add_argument("--frames_per_video", type=int, default=3)
    parser.add_argument("--frame_sample_rate", type=int, default=Config.FRAME_SAMPLE_RATE)
    parser.add_argument("--face_confidence", type=float, default=0.8)

    # Image-mode controls
    parser.add_argument(
        "--max_images_per_class",
        type=int,
        default=None,
        help="Cap images per class for image mode",
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_training(arguments)
