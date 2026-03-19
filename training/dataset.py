"""
Video dataset utilities for FaceForensics++ training.
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import detect_faces, extract_frames
from src.utils import setup_logger

logger = setup_logger("dataset")

FAKE_METHOD_TO_DIR_ATTR = {
    "Deepfakes": "DEEPFAKE_VIDEOS",
    "Face2Face": "FACE2FACE_VIDEOS",
    "FaceSwap": "FACESWAP_VIDEOS",
    "FaceShifter": "FACESHIFTER_VIDEOS",
    "NeuralTextures": "NEURALTEXTURES_VIDEOS",
}


def _list_mp4_files(directory: Path) -> List[str]:
    if not directory.exists():
        return []
    return sorted(str(path) for path in directory.glob("*.mp4"))


def collect_video_paths(
    config=None,
    fake_methods: Optional[List[str]] = None,
    max_videos_per_class: Optional[int] = None,
    seed: int = 42,
) -> Tuple[List[str], List[int]]:
    """
    Collect FF++ video paths and labels.
    """
    if config is None:
        from config import Config

        config = Config

    if fake_methods is None:
        fake_methods = ["Deepfakes"]

    real_videos = _list_mp4_files(config.ORIGINAL_VIDEOS)
    fake_videos: List[str] = []

    for method in fake_methods:
        attr = FAKE_METHOD_TO_DIR_ATTR.get(method)
        if attr is None:
            logger.warning(f"Unknown fake method skipped: {method}")
            continue
        method_dir = getattr(config, attr)
        method_videos = _list_mp4_files(method_dir)
        if not method_videos:
            logger.warning(f"No videos found for method '{method}' in {method_dir}")
            continue
        logger.info(f"{method}: {len(method_videos)} videos")
        fake_videos.extend(method_videos)

    rng = random.Random(seed)
    if max_videos_per_class is not None:
        rng.shuffle(real_videos)
        rng.shuffle(fake_videos)
        real_videos = real_videos[:max_videos_per_class]
        fake_videos = fake_videos[:max_videos_per_class]

    all_paths = real_videos + fake_videos
    all_labels = [0] * len(real_videos) + [1] * len(fake_videos)
    logger.info(
        f"Collected videos: REAL={len(real_videos)}, FAKE={len(fake_videos)}, TOTAL={len(all_paths)}"
    )
    return all_paths, all_labels


def split_video_paths(
    paths: List[str],
    labels: List[int],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[List[str], List[int], List[str], List[int]]:
    """
    Deterministically split paths and labels into train/val.
    """
    assert len(paths) == len(labels), "paths/labels length mismatch"

    combined = list(zip(paths, labels))
    rng = random.Random(seed)
    rng.shuffle(combined)

    if not combined:
        return [], [], [], []

    shuffled_paths, shuffled_labels = zip(*combined)
    shuffled_paths = list(shuffled_paths)
    shuffled_labels = list(shuffled_labels)

    split_index = int(len(shuffled_paths) * train_ratio)
    train_paths = shuffled_paths[:split_index]
    train_labels = shuffled_labels[:split_index]
    val_paths = shuffled_paths[split_index:]
    val_labels = shuffled_labels[split_index:]

    logger.info(f"Split complete: train={len(train_paths)}, val={len(val_paths)}")
    return train_paths, train_labels, val_paths, val_labels


def build_dataset(
    config=None,
    fake_methods: Optional[List[str]] = None,
    train_ratio: float = 0.8,
    max_videos_per_class: Optional[int] = None,
    seed: int = 42,
) -> Tuple[List[str], List[int], List[str], List[int]]:
    """
    Build train/val video splits from FaceForensics++.
    """
    all_paths, all_labels = collect_video_paths(
        config=config,
        fake_methods=fake_methods,
        max_videos_per_class=max_videos_per_class,
        seed=seed,
    )
    return split_video_paths(
        paths=all_paths,
        labels=all_labels,
        train_ratio=train_ratio,
        seed=seed,
    )


class DeepfakeVideoDataset(Dataset):
    """
    On-the-fly face extraction dataset from videos.
    """

    def __init__(
        self,
        video_paths: List[str],
        labels: List[int],
        processor=None,
        frames_per_video: int = 5,
        frame_sample_rate: int = 10,
        augment: bool = False,
        target_size: int = 224,
        face_confidence: float = 0.8,
    ):
        self.video_paths = video_paths
        self.labels = labels
        self.processor = processor
        self.frames_per_video = frames_per_video
        self.frame_sample_rate = frame_sample_rate
        self.augment = augment
        self.target_size = target_size
        self.face_confidence = face_confidence

        if len(video_paths) != len(labels):
            raise ValueError(
                f"Mismatch: {len(video_paths)} videos vs {len(labels)} labels"
            )

        logger.info(
            f"Video dataset: {len(video_paths)} samples (REAL={labels.count(0)}, FAKE={labels.count(1)})"
        )

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        video_path = self.video_paths[index]
        label = self.labels[index]

        try:
            frames = extract_frames(
                video_path,
                every_n=self.frame_sample_rate,
                max_frames=max(self.frames_per_video * 3, self.frames_per_video),
            )
            if not frames:
                return self._blank_sample(label)

            try:
                faces = detect_faces(frames, confidence_threshold=self.face_confidence)
            except Exception as exc:
                logger.debug(f"Face detection failed for {video_path}: {exc}")
                faces = []
            if not faces:
                faces = self._center_crop_frames(frames)
            if not faces:
                return self._blank_sample(label)

            face = random.choice(faces[: self.frames_per_video])
            face = face.resize((self.target_size, self.target_size), Image.LANCZOS)
            if self.augment:
                face = self._apply_augmentation(face)

            if self.processor is not None:
                inputs = self.processor(images=face, return_tensors="pt")
                pixel_values = inputs["pixel_values"].squeeze(0)
            else:
                pixel_values = self._manual_to_tensor(face)

            return {
                "pixel_values": pixel_values,
                "labels": torch.tensor(label, dtype=torch.long),
            }
        except Exception as exc:
            logger.debug(f"Video sample failed for {video_path}: {exc}")
            return self._blank_sample(label)

    def _blank_sample(self, label: int) -> Dict[str, torch.Tensor]:
        pixel_values = torch.zeros(3, self.target_size, self.target_size)
        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(label, dtype=torch.long),
        }

    def _center_crop_frames(self, frames: List[np.ndarray]) -> List[Image.Image]:
        results: List[Image.Image] = []
        for frame in frames[: self.frames_per_video]:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            width, height = pil_image.size
            side = min(width, height)
            left = (width - side) // 2
            top = (height - side) // 2
            results.append(pil_image.crop((left, top, left + side, top + side)))
        return results

    @staticmethod
    def _apply_augmentation(image: Image.Image) -> Image.Image:
        from torchvision import transforms

        aug = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                ),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.05, 0.05),
                    scale=(0.95, 1.05),
                ),
            ]
        )
        return aug(image)

    @staticmethod
    def _manual_to_tensor(image: Image.Image) -> torch.Tensor:
        arr = np.array(image, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        return torch.from_numpy(arr).permute(2, 0, 1)


def create_dataloaders(
    config=None,
    processor=None,
    fake_methods: Optional[List[str]] = None,
    max_videos: Optional[int] = None,
    train_ratio: float = 0.8,
    seed: int = 42,
    frames_per_video: int = 3,
    frame_sample_rate: int = 10,
    face_confidence: float = 0.8,
    train_batch_size: Optional[int] = None,
    eval_batch_size: Optional[int] = None,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train/val DataLoaders for video training mode.
    """
    if config is None:
        from config import Config

        config = Config

    if train_batch_size is None:
        train_batch_size = config.TRAIN_BATCH_SIZE
    if eval_batch_size is None:
        eval_batch_size = config.EVAL_BATCH_SIZE

    train_paths, train_labels, val_paths, val_labels = build_dataset(
        config=config,
        fake_methods=fake_methods,
        train_ratio=train_ratio,
        max_videos_per_class=max_videos,
        seed=seed,
    )

    train_dataset = DeepfakeVideoDataset(
        video_paths=train_paths,
        labels=train_labels,
        processor=processor,
        frames_per_video=frames_per_video,
        frame_sample_rate=frame_sample_rate,
        augment=True,
        target_size=config.FACE_IMAGE_SIZE,
        face_confidence=face_confidence,
    )
    val_dataset = DeepfakeVideoDataset(
        video_paths=val_paths,
        labels=val_labels,
        processor=processor,
        frames_per_video=frames_per_video,
        frame_sample_rate=frame_sample_rate,
        augment=False,
        target_size=config.FACE_IMAGE_SIZE,
        face_confidence=face_confidence,
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
