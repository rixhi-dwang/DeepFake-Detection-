"""
Convert FaceForensics++ videos into an image dataset for training.

This script extracts face crops (or center crops) from REAL/FAKE videos and
writes a train/val image folder structure.
"""

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import List

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config
from src.preprocessing import detect_faces, extract_frames
from src.utils import setup_logger
from training.dataset import build_dataset

logger = setup_logger("prepare_images")


def center_crop_frame(frame: np.ndarray, size: int) -> Image.Image:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    width, height = pil_image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    crop = pil_image.crop((left, top, left + side, top + side))
    return crop.resize((size, size), Image.LANCZOS)


def extract_images_from_video(
    video_path: str,
    frames_per_video: int,
    frame_sample_rate: int,
    face_confidence: float,
    image_size: int,
    use_face_detection: bool,
) -> List[Image.Image]:
    frames = extract_frames(
        video_path=video_path,
        every_n=frame_sample_rate,
        max_frames=max(frames_per_video * 4, frames_per_video),
    )
    if not frames:
        return []

    if use_face_detection:
        try:
            faces = detect_faces(frames, confidence_threshold=face_confidence)
        except Exception as exc:
            logger.debug(f"Face detection failed for {video_path}: {exc}")
            faces = []
    else:
        faces = []

    if faces:
        return [
            face.resize((image_size, image_size), Image.LANCZOS)
            for face in faces[:frames_per_video]
        ]

    # Fallback to center-cropped frames
    center_crops = [center_crop_frame(frame, size=image_size) for frame in frames[:frames_per_video]]
    return center_crops


def save_split(
    split_name: str,
    video_paths: List[str],
    labels: List[int],
    output_root: Path,
    frames_per_video: int,
    frame_sample_rate: int,
    face_confidence: float,
    image_size: int,
    use_face_detection: bool,
) -> List[dict]:
    metadata: List[dict] = []
    split_dir = output_root / split_name

    for video_path, label in tqdm(
        list(zip(video_paths, labels)),
        desc=f"{split_name} videos",
        unit="video",
    ):
        label_name = "FAKE" if label == 1 else "REAL"
        class_dir = split_dir / label_name
        class_dir.mkdir(parents=True, exist_ok=True)

        try:
            images = extract_images_from_video(
                video_path=video_path,
                frames_per_video=frames_per_video,
                frame_sample_rate=frame_sample_rate,
                face_confidence=face_confidence,
                image_size=image_size,
                use_face_detection=use_face_detection,
            )
            if not images:
                continue

            stem = Path(video_path).stem
            for idx, image in enumerate(images):
                filename = f"{stem}_{idx:03d}.jpg"
                save_path = class_dir / filename
                image.save(save_path, format="JPEG", quality=95)
                metadata.append(
                    {
                        "split": split_name,
                        "label": label_name,
                        "source_video": video_path,
                        "image_path": str(save_path),
                    }
                )
        except Exception as exc:
            logger.warning(f"Failed to process {video_path}: {exc}")

    return metadata


def write_metadata(metadata: List[dict], output_root: Path):
    csv_path = output_root / "metadata.csv"
    json_path = output_root / "metadata.json"
    summary_path = output_root / "summary.json"

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["split", "label", "source_video", "image_path"]
        )
        writer.writeheader()
        writer.writerows(metadata)

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    summary = {
        "train_real": sum(1 for row in metadata if row["split"] == "train" and row["label"] == "REAL"),
        "train_fake": sum(1 for row in metadata if row["split"] == "train" and row["label"] == "FAKE"),
        "val_real": sum(1 for row in metadata if row["split"] == "val" and row["label"] == "REAL"),
        "val_fake": sum(1 for row in metadata if row["split"] == "val" and row["label"] == "FAKE"),
        "total_images": len(metadata),
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    logger.info(f"Wrote metadata CSV: {csv_path}")
    logger.info(f"Wrote metadata JSON: {json_path}")
    logger.info(f"Wrote summary JSON: {summary_path}")


def run_conversion(
    output_dir: Path,
    methods: List[str],
    frames_per_video: int,
    frame_sample_rate: int,
    face_confidence: float,
    train_ratio: float,
    max_videos_per_class: int,
    seed: int,
    image_size: int,
    use_face_detection: bool,
    force: bool,
):
    if output_dir.exists() and force:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_paths, train_labels, val_paths, val_labels = build_dataset(
        config=Config,
        fake_methods=methods,
        train_ratio=train_ratio,
        max_videos_per_class=max_videos_per_class,
        seed=seed,
    )
    logger.info(
        f"Video split ready: train={len(train_paths)}, val={len(val_paths)}"
    )

    metadata = []
    metadata.extend(
        save_split(
            split_name="train",
            video_paths=train_paths,
            labels=train_labels,
            output_root=output_dir,
            frames_per_video=frames_per_video,
            frame_sample_rate=frame_sample_rate,
            face_confidence=face_confidence,
            image_size=image_size,
            use_face_detection=use_face_detection,
        )
    )
    metadata.extend(
        save_split(
            split_name="val",
            video_paths=val_paths,
            labels=val_labels,
            output_root=output_dir,
            frames_per_video=frames_per_video,
            frame_sample_rate=frame_sample_rate,
            face_confidence=face_confidence,
            image_size=image_size,
            use_face_detection=use_face_detection,
        )
    )

    write_metadata(metadata, output_dir)

    conversion_config = {
        "output_dir": str(output_dir),
        "methods": methods,
        "frames_per_video": frames_per_video,
        "frame_sample_rate": frame_sample_rate,
        "face_confidence": face_confidence,
        "train_ratio": train_ratio,
        "max_videos_per_class": max_videos_per_class,
        "seed": seed,
        "image_size": image_size,
        "use_face_detection": use_face_detection,
    }
    with open(output_dir / "conversion_config.json", "w", encoding="utf-8") as handle:
        json.dump(conversion_config, handle, indent=2)
    logger.info(f"Saved conversion config to {output_dir / 'conversion_config.json'}")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert FaceForensics++ videos into image dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Config.IMAGE_DATASET_ROOT),
        help="Output root directory for train/val image folders",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["Deepfakes", "Face2Face", "FaceSwap", "FaceShifter", "NeuralTextures"],
        help="Fake methods to include",
    )
    parser.add_argument(
        "--frames_per_video",
        type=int,
        default=5,
        help="Images to extract per video",
    )
    parser.add_argument(
        "--frame_sample_rate",
        type=int,
        default=10,
        help="Sample every Nth frame from each video",
    )
    parser.add_argument(
        "--face_confidence",
        type=float,
        default=0.8,
        help="MTCNN confidence threshold",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=Config.TRAIN_RATIO,
        help="Train split ratio (0-1)",
    )
    parser.add_argument(
        "--max_videos_per_class",
        type=int,
        default=None,
        help="Optional cap for debugging",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=Config.RANDOM_SEED,
        help="Random seed for split/sampling",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=Config.FACE_IMAGE_SIZE,
        help="Output image size",
    )
    parser.add_argument(
        "--no_face_detection",
        action="store_true",
        help="Disable face detection and use center crop only",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete output_dir before conversion if it exists",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_conversion(
        output_dir=Path(args.output_dir),
        methods=args.methods,
        frames_per_video=args.frames_per_video,
        frame_sample_rate=args.frame_sample_rate,
        face_confidence=args.face_confidence,
        train_ratio=args.train_ratio,
        max_videos_per_class=args.max_videos_per_class,
        seed=args.seed,
        image_size=args.image_size,
        use_face_detection=not args.no_face_detection,
        force=args.force,
    )
