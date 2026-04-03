from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import load_vit_checkpoint
from src.utils import ensure_dir, setup_logger
from training.train_vit import FrameDataset, build_frame_manifest, build_transforms, split_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract ViT features and labels to .npy files.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained ViT checkpoint.")
    parser.add_argument("--frames-root", type=str, required=True, help="Root folder with real/fake frame folders.")
    parser.add_argument("--output-dir", type=str, default="features", help="Directory to save X.npy and y.npy.")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="train")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--max-images-per-video", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available.")
    return device


def collect_samples(args: argparse.Namespace, logger) -> List:
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


def create_loader(samples: List, args: argparse.Namespace) -> DataLoader:
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


def extract_and_save_features(
    model,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    split: str,
) -> None:
    all_features = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Extracting features"):
            images = images.to(device, non_blocking=True)
            features = model.get_features(images)
            all_features.append(features.detach().cpu().numpy().astype(np.float32))
            all_labels.append(labels.detach().cpu().numpy().astype(np.int64))

    if not all_features:
        raise RuntimeError("No features extracted.")

    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)

    ensure_dir(output_dir)
    x_path = output_dir / "X.npy"
    y_path = output_dir / "y.npy"
    meta_path = output_dir / "meta.json"

    np.save(x_path, X)
    np.save(y_path, y)
    meta_path.write_text(
        json.dumps(
            {
                "split": split,
                "num_samples": int(y.shape[0]),
                "feature_dim": int(X.shape[1]),
                "x_path": str(x_path),
                "y_path": str(y_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved features: {x_path} | shape={X.shape}")
    print(f"Saved labels  : {y_path} | shape={y.shape}")


def main() -> None:
    args = parse_args()
    logger = setup_logger("extract_features")
    device = resolve_device(args.device)

    vit_model, _ = load_vit_checkpoint(checkpoint_path=args.checkpoint, device=device, num_classes=2)
    samples = collect_samples(args=args, logger=logger)
    loader = create_loader(samples=samples, args=args)

    extract_and_save_features(
        model=vit_model,
        loader=loader,
        device=device,
        output_dir=Path(args.output_dir),
        split=args.split,
    )


if __name__ == "__main__":
    main()

