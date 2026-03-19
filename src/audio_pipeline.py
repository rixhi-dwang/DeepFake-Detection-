"""
End-to-end Audio Deepfake Detection Pipeline.

Takes a video file path and:
  1. Extracts audio track using subprocess + ffmpeg
  2. Loads audio with librosa
  3. Generates Mel spectrogram features
  4. Runs audio deepfake classifier
  5. Returns prediction
"""

import os
import subprocess
import time
import numpy as np
from pathlib import Path
from typing import Dict, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import setup_logger, safe_cleanup

logger = setup_logger("audio_pipeline")


def extract_audio_from_video(
    video_path: str,
    output_dir: Optional[str] = None,
    sample_rate: int = 16000,
) -> Optional[str]:
    """
    Extract audio from a video file using ffmpeg.

    Args:
        video_path: Path to input video.
        output_dir: Directory for temporary audio file.
        sample_rate: Target sample rate in Hz.

    Returns:
        Path to extracted WAV file, or None on failure.
    """
    from config import Config

    if output_dir is None:
        output_dir = str(Config.TEMP_DIR)
    os.makedirs(output_dir, exist_ok=True)

    video_name = Path(video_path).stem
    audio_path = os.path.join(output_dir, f"{video_name}_audio.wav")

    try:
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vn",                  # No video
            "-acodec", "pcm_s16le", # PCM 16-bit
            "-ar", str(sample_rate),
            "-ac", "1",             # Mono
            "-y",                   # Overwrite
            str(audio_path),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            logger.warning(
                f"ffmpeg failed (code {result.returncode}): {result.stderr[:200]}"
            )
            return None

        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            logger.warning("ffmpeg produced empty audio file")
            return None

        logger.info(f"Audio extracted: {audio_path}")
        return audio_path

    except FileNotFoundError:
        logger.error(
            "ffmpeg not found. Please install ffmpeg and add it to PATH."
        )
        return None
    except subprocess.TimeoutExpired:
        logger.error("Audio extraction timed out (>60s)")
        return None
    except Exception as e:
        logger.error(f"Audio extraction error: {e}")
        return None


def load_audio(
    audio_path: str,
    sample_rate: int = 16000,
    max_duration: float = 10.0,
) -> Optional[np.ndarray]:
    """
    Load an audio file using librosa.

    Args:
        audio_path: Path to WAV file.
        sample_rate: Target sample rate.
        max_duration: Maximum duration in seconds to load.

    Returns:
        1D numpy array of audio samples, or None on failure.
    """
    try:
        import librosa

        audio, sr = librosa.load(
            audio_path,
            sr=sample_rate,
            mono=True,
            duration=max_duration,
        )

        if audio is None or len(audio) == 0:
            logger.warning("Loaded audio is empty")
            return None

        logger.info(
            f"Audio loaded: {len(audio)} samples, "
            f"{len(audio)/sr:.2f}s at {sr}Hz"
        )
        return audio

    except Exception as e:
        logger.error(f"Failed to load audio: {e}")
        return None


def compute_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int = 16000,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> Optional[np.ndarray]:
    """
    Compute log-Mel spectrogram from audio signal.

    Args:
        audio: 1D audio signal.
        sample_rate: Sample rate in Hz.
        n_mels: Number of Mel bands.
        n_fft: FFT window size.
        hop_length: Hop length between frames.

    Returns:
        Log-Mel spectrogram as 2D numpy array (n_mels, time).
    """
    try:
        import librosa

        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
        )

        # Convert to log scale
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        logger.info(f"Mel spectrogram computed: shape={log_mel.shape}")
        return log_mel

    except Exception as e:
        logger.error(f"Mel spectrogram computation failed: {e}")
        return None


def run_audio_pipeline(video_path: str) -> Dict:
    """
    Run the complete audio deepfake detection pipeline.

    Steps:
      1. Extract audio from video (ffmpeg)
      2. Load audio signal (librosa)
      3. Compute Mel spectrogram
      4. Run audio deepfake classifier
      5. Clean up temporary files

    Args:
        video_path: Path to the input video file.

    Returns:
        Dict with prediction, timing, and status.
    """
    from config import Config
    from models.audio_model import predict_audio

    result = {
        "status": "error",
        "video_path": str(video_path),
        "prediction": None,
        "mel_spectrogram_shape": None,
        "timing": {},
        "error": None,
    }

    total_start = time.time()
    temp_audio_path = None

    if not Config.ENABLE_AUDIO:
        result["status"] = "partial"
        result["error"] = "Audio model is disabled in local-only configuration"
        result["prediction"] = {
            "label": "UNKNOWN",
            "confidence": 0.0,
            "fake_probability": 0.5,
            "real_probability": 0.5,
            "note": "Audio disabled (set DEEPFAKE_ENABLE_AUDIO=1 and provide local audio model)",
        }
        return result

    try:
        # ── Step 1: Extract audio ───────────────────────────────────
        t0 = time.time()
        temp_audio_path = extract_audio_from_video(
            video_path,
            sample_rate=Config.AUDIO_SAMPLE_RATE,
        )
        result["timing"]["audio_extraction"] = round(time.time() - t0, 3)

        if temp_audio_path is None:
            result["error"] = (
                "Could not extract audio from video. "
                "The video may not contain an audio track, or ffmpeg is not installed."
            )
            # Return a neutral prediction instead of failing completely
            result["prediction"] = {
                "label": "UNKNOWN",
                "confidence": 0.0,
                "fake_probability": 0.5,
                "real_probability": 0.5,
                "note": "No audio available — audio analysis skipped",
            }
            result["status"] = "partial"
            return result

        # ── Step 2: Load audio ──────────────────────────────────────
        t0 = time.time()
        audio = load_audio(
            temp_audio_path,
            sample_rate=Config.AUDIO_SAMPLE_RATE,
            max_duration=Config.AUDIO_MAX_DURATION,
        )
        result["timing"]["audio_loading"] = round(time.time() - t0, 3)

        if audio is None:
            result["error"] = "Failed to load audio signal"
            return result

        # ── Step 3: Mel Spectrogram ─────────────────────────────────
        t0 = time.time()
        mel_spec = compute_mel_spectrogram(
            audio,
            sample_rate=Config.AUDIO_SAMPLE_RATE,
            n_mels=Config.MEL_N_MELS,
            n_fft=Config.MEL_N_FFT,
            hop_length=Config.MEL_HOP_LENGTH,
        )
        result["timing"]["spectrogram"] = round(time.time() - t0, 3)

        if mel_spec is not None:
            result["mel_spectrogram_shape"] = list(mel_spec.shape)

        # ── Step 4: Audio Prediction ────────────────────────────────
        t0 = time.time()
        prediction = predict_audio(audio, sample_rate=Config.AUDIO_SAMPLE_RATE)
        result["timing"]["model_inference"] = round(time.time() - t0, 3)

        result["prediction"] = prediction
        result["status"] = "success"
        result["timing"]["total"] = round(time.time() - total_start, 3)

        logger.info(
            f"Audio result: {prediction.get('label', 'UNKNOWN')} "
            f"(confidence: {prediction.get('confidence', 0):.4f})"
        )

    except Exception as e:
        result["error"] = f"Audio pipeline error: {str(e)}"
        logger.error(f"Audio pipeline error: {e}")

    finally:
        # Clean up temporary audio file
        if temp_audio_path:
            safe_cleanup(temp_audio_path)

    return result


if __name__ == "__main__":
    """Quick test: run audio pipeline on a sample video."""
    import json
    from config import Config

    Config.summary()

    test_video = str(Config.ORIGINAL_VIDEOS / "000.mp4")
    print(f"\nTesting audio pipeline with: {test_video}\n")

    result = run_audio_pipeline(test_video)
    print(json.dumps(result, indent=2))
