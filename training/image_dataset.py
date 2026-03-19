"""
Image dataset utilities for training from extracted FaceForensics images.

Expected structure:
  <image_root>/
    train/
      REAL/*.jpg
      FAKE/*.jpg
    val/
      REAL/*.jpg
      FAKE/*.jpg
"""

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import setup_logger

logger = setup_logger("image_dataset")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _scan_class_dir(class_dir: Path) -> List[str]:
    if not class_dir.exists():
        return []
    files: List[str] = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(str(path) for path in class_dir.rglob(f"*{ext}"))
    return sorted(files)


def collect_image_paths(
    image_root: Path,
    split: str,
    max_images_per_class: Optional[int] = None,
    seed: int = 42,
) -> Tuple[List[str], List[int]]:
    """
    Collect image paths/labels from one split.
    """
    split_dir = image_root / split
    real_images = _scan_class_dir(split_dir / "REAL")
    fake_images = _scan_class_dir(split_dir / "FAKE")

    rng = random.Random(seed)
    if max_images_per_class is not None:
        rng.shuffle(real_images)
        rng.shuffle(fake_images)
        real_images = real_images[:max_images_per_class]
        fake_images = fake_images[:max_images_per_class]

    paths = real_images + fake_images
    labels = [0] * len(real_images) + [1] * len(fake_images)
    logger.info(
        f"{split} image set: REAL={len(real_images)}, FAKE={len(fake_images)}, TOTAL={len(paths)}"
    )
    return paths, labels


class DeepfakeImageDataset(Dataset):
    """
    Dataset over pre-extracted images.
    """

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        processor=None,
        augment: bool = False,
        target_size: int = 224,
    ):
        if len(image_paths) != len(labels):
            raise ValueError(
                f"Mismatch: {len(image_paths)} images vs {len(labels)} labels"
            )
        self.image_paths = image_paths
        self.labels = labels
        self.processor = processor
        self.augment = augment
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        path = self.image_paths[index]
        label = self.labels[index]

        try:
            image = Image.open(path).convert("RGB")
            image = image.resize((self.target_size, self.target_size), Image.LANCZOS)
            if self.augment:
                image = self._apply_augmentation(image)

            if self.processor is not None:
                inputs = self.processor(images=image, return_tensors="pt")
                pixel_values = inputs["pixel_values"].squeeze(0)
            else:
                pixel_values = self._manual_to_tensor(image)

            return {
                "pixel_values": pixel_values,
                "labels": torch.tensor(label, dtype=torch.long),
            }
        except Exception as exc:
            logger.debug(f"Image sample failed for {path}: {exc}")
            return {
                "pixel_values": torch.zeros(3, self.target_size, self.target_size),
                "labels": torch.tensor(label, dtype=torch.long),
            }

    @staticmethod
    def _apply_augmentation(image: Image.Image) -> Image.Image:
        from torchvision import transforms

        aug = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.15,
                    hue=0.05,
                ),
            ]
        )
        return aug(image)

    @staticmethod
    def _manual_to_tensor(image: Image.Image) -> torch.Tensor:
        arr = np.array(image, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        return torch.from_numpy(arr).permute(2, 0, 1)


def create_image_dataloaders(
    image_root: Path,
    processor=None,
    train_batch_size: int = 16,
    eval_batch_size: int = 8,
    num_workers: int = 0,
    max_images_per_class: Optional[int] = None,
    seed: int = 42,
    target_size: int = 224,
    use_augmentation: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders from converted image dataset.
    """
    train_paths, train_labels = collect_image_paths(
        image_root=image_root,
        split="train",
        max_images_per_class=max_images_per_class,
        seed=seed,
    )
    val_paths, val_labels = collect_image_paths(
        image_root=image_root,
        split="val",
        max_images_per_class=max_images_per_class,
        seed=seed,
    )

    train_dataset = DeepfakeImageDataset(
        image_paths=train_paths,
        labels=train_labels,
        processor=processor,
        augment=use_augmentation,
        target_size=target_size,
    )
    val_dataset = DeepfakeImageDataset(
        image_paths=val_paths,
        labels=val_labels,
        processor=processor,
        augment=False,
        target_size=target_size,
    )

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader
