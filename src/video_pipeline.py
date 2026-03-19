"""
End-to-end Video Deepfake Detection Pipeline.

Takes a video file path and returns a complete analysis:
  Video → Frame Extraction → Face Detection → ViT Prediction → Aggregation
"""

import time
from pathlib import Path
from typing import Dict, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import setup_logger, validate_video_file, get_video_info
from src.preprocessing import extract_frames, detect_faces
from models.video_model import (
    load_video_model,
    predict_batch,
    aggregate_predictions,
)

logger = setup_logger("video_pipeline")

# ── Module-level model cache ──────────────────────────────────────────
_cached_model = None
_cached_processor = None


def _get_model_and_processor():
    """Load model once and cache it for subsequent calls."""
    global _cached_model, _cached_processor
    if _cached_model is None or _cached_processor is None:
        _cached_model, _cached_processor = load_video_model()
    return _cached_model, _cached_processor


def run_video_pipeline(
    video_path: str,
    frame_sample_rate: Optional[int] = None,
    max_frames: Optional[int] = None,
    batch_size: Optional[int] = None,
    face_confidence: Optional[float] = None,
) -> Dict:
    """
    Run the complete video deepfake detection pipeline.

    Steps:
      1. Validate input video
      2. Extract frames at regular intervals
      3. Detect and crop faces using MTCNN
      4. Run ViT model on each face
      5. Aggregate predictions

    Args:
        video_path: Path to the input video file.
        frame_sample_rate: Extract every Nth frame (default from config).
        max_frames: Maximum frames to process (default from config).
        batch_size: Inference batch size (default from config).
        face_confidence: MTCNN confidence threshold (default from config).

    Returns:
        Dict containing:
          - prediction: aggregated label and confidence
          - video_info: metadata about the input video
          - timing: processing time breakdown
          - status: "success" or "error"
          - error: error message if any
    """
    from config import Config

    # Apply defaults from config
    if frame_sample_rate is None:
        frame_sample_rate = Config.FRAME_SAMPLE_RATE
    if max_frames is None:
        max_frames = Config.MAX_FRAMES
    if batch_size is None:
        batch_size = Config.VIDEO_BATCH_SIZE
    if face_confidence is None:
        face_confidence = Config.FACE_CONFIDENCE_THRESHOLD

    result = {
        "status": "error",
        "video_path": str(video_path),
        "prediction": None,
        "video_info": None,
        "timing": {},
        "error": None,
    }

    total_start = time.time()

    # ── Step 1: Validate ────────────────────────────────────────────
    try:
        if not validate_video_file(video_path):
            result["error"] = (
                f"Invalid video file: {video_path}. "
                f"File may not exist, be empty, or have an unsupported extension."
            )
            return result

        result["video_info"] = get_video_info(video_path)
        logger.info(f"Processing video: {Path(video_path).name}")
    except Exception as e:
        result["error"] = f"Validation failed: {str(e)}"
        return result

    # ── Step 2: Extract Frames ──────────────────────────────────────
    t0 = time.time()
    try:
        frames = extract_frames(
            video_path,
            every_n=frame_sample_rate,
            max_frames=max_frames,
        )
        result["timing"]["frame_extraction"] = round(time.time() - t0, 3)

        if not frames:
            result["error"] = "No frames could be extracted from the video."
            return result

    except Exception as e:
        result["error"] = f"Frame extraction failed: {str(e)}"
        return result

    # ── Step 3: Detect Faces ────────────────────────────────────────
    t0 = time.time()
    try:
        faces = detect_faces(frames, confidence_threshold=face_confidence)
        result["timing"]["face_detection"] = round(time.time() - t0, 3)
    except Exception as e:
        logger.warning(f"Face detection unavailable, using center-crop fallback: {e}")
        faces = []
        result["timing"]["face_detection"] = round(time.time() - t0, 3)

    if not faces:
        # Fallback: use raw frames if no faces detected
        logger.warning(
            "No faces detected. Using raw center-cropped frames as fallback."
        )
        from PIL import Image
        import cv2

        fallback_faces = []
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            # Center crop to square
            w, h = pil_img.size
            side = min(w, h)
            left = (w - side) // 2
            top = (h - side) // 2
            cropped = pil_img.crop((left, top, left + side, top + side))
            cropped = cropped.resize((224, 224), Image.LANCZOS)
            fallback_faces.append(cropped)
        faces = fallback_faces
        logger.info(f"Using {len(faces)} center-cropped frames as fallback")

    # ── Step 4: Model Prediction ────────────────────────────────────
    t0 = time.time()
    try:
        model, processor = _get_model_and_processor()
        predictions = predict_batch(
            model, processor, faces, batch_size=batch_size
        )
        result["timing"]["model_inference"] = round(time.time() - t0, 3)

    except Exception as e:
        result["error"] = f"Model inference failed: {str(e)}"
        return result

    # ── Step 5: Aggregate ───────────────────────────────────────────
    try:
        aggregated = aggregate_predictions(predictions)
        result["prediction"] = aggregated
        result["status"] = "success"
        result["timing"]["total"] = round(time.time() - total_start, 3)

        logger.info(
            f"Video result: {aggregated['label']} "
            f"(confidence: {aggregated['confidence']:.4f}, "
            f"frames: {aggregated['num_frames_analyzed']})"
        )

    except Exception as e:
        result["error"] = f"Aggregation failed: {str(e)}"

    return result


if __name__ == "__main__":
    """Quick test: run pipeline on a sample video."""
    import json

    from config import Config

    Config.summary()

    # Test with first original video
    test_video = str(Config.ORIGINAL_VIDEOS / "000.mp4")
    print(f"\nTesting with: {test_video}\n")

    result = run_video_pipeline(test_video)
    print(json.dumps(result, indent=2))
