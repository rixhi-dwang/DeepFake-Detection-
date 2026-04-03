"""
Main script to convert deepfake videos into a structured face image dataset.

Features:
  - OpenCV streaming video read (no full video in memory)
  - Configurable frame sampling
  - MTCNN-first face detection with Haar fallback
  - Largest-face selection
  - Margin-aware face crop + resize
  - Resume processing
  - Optional multithreading
  - Metadata CSV and optional bbox JSON
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
from tqdm import tqdm

# Allow both:
#   python src/extract_frames.py
#   python -m src.extract_frames
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FrameExtractionConfig, build_arg_parser, frame_config_from_args
from src.face_detection import FaceDetector
from src.utils import (
    count_extracted_frames,
    infer_binary_label,
    is_video_already_processed,
    list_video_files,
    make_unique_video_output_dir,
    setup_logger,
    utc_now_iso,
    write_json_file,
    write_metadata_csv,
    write_source_marker,
)


_THREAD_LOCAL = threading.local()


def _get_worker_detector(config: FrameExtractionConfig, logger: logging.Logger) -> FaceDetector:
    """
    Keep one detector per worker thread for better performance.
    """
    detector = getattr(_THREAD_LOCAL, "detector", None)
    detector_key = getattr(_THREAD_LOCAL, "detector_key", None)
    current_key = (config.prefer_mtcnn, round(config.min_face_confidence, 4))

    if detector is None or detector_key != current_key:
        detector = FaceDetector(
            prefer_mtcnn=config.prefer_mtcnn,
            min_confidence=config.min_face_confidence,
            logger=logger,
        )
        _THREAD_LOCAL.detector = detector
        _THREAD_LOCAL.detector_key = current_key

    return detector


def discover_labeled_videos(
    config: FrameExtractionConfig,
    logger: logging.Logger,
) -> List[Tuple[Path, str]]:
    """
    Discover and label videos from dataset directory.
    """
    videos = list_video_files(config.dataset_path, config.allowed_extensions)
    if not videos:
        logger.warning("No valid videos found under: %s", config.dataset_path)
        return []

    labeled: List[Tuple[Path, str]] = []
    skipped_unknown = 0

    for video_path in videos:
        label = infer_binary_label(
            video_path=video_path,
            dataset_root=config.dataset_path,
            real_dir_tokens=config.real_dir_tokens,
            fake_dir_tokens=config.fake_dir_tokens,
        )
        if label is None:
            skipped_unknown += 1
            logger.debug("Skipping video with unknown label mapping: %s", video_path)
            continue
        labeled.append((video_path, label))

    if skipped_unknown > 0:
        logger.warning(
            "Skipped %d videos due to unknown label mapping. "
            "Add folder tokens in config if needed.",
            skipped_unknown,
        )

    return labeled


def process_video(
    video_path: Path,
    label: str,
    config: FrameExtractionConfig,
    logger: logging.Logger,
) -> Dict[str, object]:
    """
    Process one video:
      - sample frames
      - detect largest face
      - crop + resize + save
    """
    start_time = time.time()
    start_time_utc = utc_now_iso()

    output_dir = make_unique_video_output_dir(config.output_path, label, video_path)
    summary_file = output_dir / "summary.json"
    existing_frames = count_extracted_frames(output_dir)

    if config.resume and is_video_already_processed(output_dir):
        return {
            "video_path": str(video_path),
            "video_name": video_path.name,
            "label": label,
            "output_dir": str(output_dir),
            "status": "skipped_resume",
            "detector": "",
            "total_frames_read": 0,
            "sampled_frames": 0,
            "saved_frames": existing_frames,
            "skipped_no_face": 0,
            "skipped_errors": 0,
            "duration_seconds": round(time.time() - start_time, 3),
            "start_time_utc": start_time_utc,
            "end_time_utc": utc_now_iso(),
            "bbox_file": "",
            "message": "Skipped because output already exists and resume mode is enabled.",
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_source_marker(output_dir, video_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Cannot open video: %s", video_path)
        error_row = {
            "video_path": str(video_path),
            "video_name": video_path.name,
            "label": label,
            "output_dir": str(output_dir),
            "status": "error_open_video",
            "detector": "",
            "total_frames_read": 0,
            "sampled_frames": 0,
            "saved_frames": 0,
            "skipped_no_face": 0,
            "skipped_errors": 1,
            "duration_seconds": round(time.time() - start_time, 3),
            "start_time_utc": start_time_utc,
            "end_time_utc": utc_now_iso(),
            "bbox_file": "",
            "message": "OpenCV failed to open this file (possibly corrupted).",
        }
        write_json_file(summary_file, error_row)
        return error_row

    total_frames_read = 0
    sampled_frames = 0
    saved_frames = 0
    skipped_no_face = 0
    skipped_errors = 0
    no_face_frame_indices: List[int] = []
    error_examples: List[str] = []
    bbox_records: List[dict] = []
    detector_name = ""

    detector = _get_worker_detector(config, logger)
    detector_name = detector.detector_name

    frame_index = 0
    try:
        while saved_frames < config.max_frames:
            ok, frame = cap.read()
            if not ok:
                break

            total_frames_read += 1

            if frame_index % config.frame_skip != 0:
                frame_index += 1
                continue

            sampled_frames += 1
            try:
                detection = detector.detect_largest_face(frame)
            except Exception as exc:
                skipped_errors += 1
                if len(error_examples) < 5:
                    error_examples.append(f"frame {frame_index}: detection error: {exc}")
                logger.warning("Face detection failed for %s frame %d: %s", video_path.name, frame_index, exc)
                frame_index += 1
                continue

            if detection is None:
                skipped_no_face += 1
                if len(no_face_frame_indices) < 20:
                    no_face_frame_indices.append(frame_index)
                logger.debug("No face found in %s frame %d", video_path.name, frame_index)
                frame_index += 1
                continue

            try:
                face_img, expanded_bbox = detector.crop_face_with_margin(
                    frame_bgr=frame,
                    bbox=detection.bbox,
                    margin_ratio=config.face_margin,
                    image_size=config.image_size,
                )
                frame_name = f"frame_{saved_frames:03d}.jpg"
                output_file = output_dir / frame_name
                face_img.save(output_file, format="JPEG", quality=config.jpeg_quality, optimize=True)

                if config.save_bboxes:
                    bbox_records.append(
                        {
                            "frame_index": frame_index,
                            "saved_frame_name": frame_name,
                            "detector": detection.detector,
                            "score": round(float(detection.score), 6),
                            "bbox": list(map(int, detection.bbox)),
                            "bbox_with_margin": list(map(int, expanded_bbox)),
                        }
                    )

                saved_frames += 1
            except Exception as exc:
                skipped_errors += 1
                if len(error_examples) < 5:
                    error_examples.append(f"frame {frame_index}: crop/save error: {exc}")
                logger.warning("Crop/save failed for %s frame %d: %s", video_path.name, frame_index, exc)

            frame_index += 1
    finally:
        cap.release()

    bbox_file = ""
    if config.save_bboxes and bbox_records:
        bbox_path = output_dir / config.bboxes_filename
        write_json_file(bbox_path, bbox_records)
        bbox_file = str(bbox_path)

    if saved_frames > 0:
        status = "success"
        message = f"Saved {saved_frames} face crops."
    elif skipped_errors > 0:
        status = "failed_no_output"
        message = "No images saved due to repeated processing errors."
    else:
        status = "no_face_frames"
        message = "No detectable faces in sampled frames."

    end_time_utc = utc_now_iso()
    duration = round(time.time() - start_time, 3)
    row = {
        "video_path": str(video_path),
        "video_name": video_path.name,
        "label": label,
        "output_dir": str(output_dir),
        "status": status,
        "detector": detector_name,
        "total_frames_read": total_frames_read,
        "sampled_frames": sampled_frames,
        "saved_frames": saved_frames,
        "skipped_no_face": skipped_no_face,
        "skipped_errors": skipped_errors,
        "duration_seconds": duration,
        "start_time_utc": start_time_utc,
        "end_time_utc": end_time_utc,
        "bbox_file": bbox_file,
        "no_face_frame_indices": ",".join(map(str, no_face_frame_indices)),
        "error_examples": " | ".join(error_examples),
        "message": message,
    }

    write_json_file(summary_file, row)
    return row


def run_pipeline(config: FrameExtractionConfig) -> List[Dict[str, object]]:
    """
    Run complete extraction pipeline and return metadata rows.
    """
    try:
        config.ensure_output_dirs()
    except OSError as exc:
        bootstrap_logger = setup_logger(
            name="extract_frames_bootstrap",
            level=getattr(logging, config.log_level, logging.INFO),
        )
        bootstrap_logger.error(
            "Cannot create output directories at %s: %s",
            config.output_path,
            exc,
        )
        bootstrap_logger.error(
            "Use --output-path with a writable directory and run again."
        )
        return []

    log_file = config.output_path / "extraction.log"
    try:
        logger = setup_logger(
            name="extract_frames",
            log_file=str(log_file),
            level=getattr(logging, config.log_level, logging.INFO),
        )
    except OSError as exc:
        logger = setup_logger(
            name="extract_frames",
            level=getattr(logging, config.log_level, logging.INFO),
        )
        logger.warning(
            "Could not create log file at %s (%s). Continuing with console logging only.",
            log_file,
            exc,
        )

    logger.info("Starting extraction pipeline")
    logger.info("Dataset path: %s", config.dataset_path)
    logger.info("Output path : %s", config.output_path)
    logger.info(
        "Settings: frame_skip=%d max_frames=%d image_size=%s margin=%.2f workers=%d resume=%s",
        config.frame_skip,
        config.max_frames,
        config.image_size,
        config.face_margin,
        config.num_workers,
        config.resume,
    )

    if not config.dataset_path.exists():
        logger.error("Dataset path does not exist: %s", config.dataset_path)
        return []

    jobs = discover_labeled_videos(config, logger)
    if not jobs:
        logger.warning("No processable videos found. Exiting.")
        return []

    logger.info("Total labeled videos discovered: %d", len(jobs))
    results: List[Dict[str, object]] = []

    if config.num_workers > 1:
        logger.info("Using ThreadPoolExecutor with %d workers", config.num_workers)
        with ThreadPoolExecutor(max_workers=config.num_workers) as executor:
            future_to_job = {
                executor.submit(process_video, video_path, label, config, logger): (video_path, label)
                for video_path, label in jobs
            }
            for future in tqdm(
                as_completed(future_to_job),
                total=len(future_to_job),
                desc="Videos",
                unit="video",
            ):
                video_path, label = future_to_job[future]
                try:
                    row = future.result()
                except Exception as exc:
                    logger.exception("Unhandled failure while processing %s: %s", video_path, exc)
                    row = {
                        "video_path": str(video_path),
                        "video_name": video_path.name,
                        "label": label,
                        "output_dir": "",
                        "status": "exception",
                        "detector": "",
                        "total_frames_read": 0,
                        "sampled_frames": 0,
                        "saved_frames": 0,
                        "skipped_no_face": 0,
                        "skipped_errors": 1,
                        "duration_seconds": 0.0,
                        "start_time_utc": utc_now_iso(),
                        "end_time_utc": utc_now_iso(),
                        "bbox_file": "",
                        "message": str(exc),
                    }
                results.append(row)
                logger.info(
                    "[%s] %s -> status=%s saved=%s sampled=%s",
                    label,
                    video_path.name,
                    row.get("status", ""),
                    row.get("saved_frames", 0),
                    row.get("sampled_frames", 0),
                )
    else:
        for video_path, label in tqdm(jobs, desc="Videos", unit="video"):
            row = process_video(video_path, label, config, logger)
            results.append(row)
            logger.info(
                "[%s] %s -> status=%s saved=%s sampled=%s",
                label,
                video_path.name,
                row.get("status", ""),
                row.get("saved_frames", 0),
                row.get("sampled_frames", 0),
            )

    results.sort(key=lambda item: (item.get("label", ""), item.get("video_path", "")))

    if config.save_metadata:
        metadata_path = config.output_path / config.metadata_filename
        write_metadata_csv(results, metadata_path)
        logger.info("Metadata saved: %s", metadata_path)

    status_counts: Dict[str, int] = {}
    total_saved = 0
    for row in results:
        status = str(row.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
        total_saved += int(row.get("saved_frames", 0))

    logger.info("Pipeline complete")
    logger.info("Videos processed: %d", len(results))
    logger.info("Total face images saved: %d", total_saved)
    logger.info("Status breakdown: %s", status_counts)

    return results


def main(cli_args: argparse.Namespace | None = None) -> None:
    """CLI entrypoint."""
    parser = build_arg_parser()
    args = cli_args if cli_args is not None else parser.parse_args()
    config = frame_config_from_args(args)
    run_pipeline(config)


if __name__ == "__main__":
    main()
