"""
FastAPI backend for local deepfake detection.

Local-only by design:
  - no remote model calls
  - localhost clients only
  - dashboard served from local static file
"""

import time
import uuid
import ipaddress
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import safe_cleanup, setup_logger

logger = setup_logger("api")

from config import Config

LOCAL_CLIENTS = {"127.0.0.1", "localhost", "::1", "testclient"}


def _is_loopback_host(host: Optional[str]) -> bool:
    if not host:
        return False
    if host in LOCAL_CLIENTS:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False

app = FastAPI(
    title="Deepfake Detection API (Local)",
    description="Local-only deepfake detection for video and image inputs.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=Config.LOCAL_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def localhost_only_middleware(request: Request, call_next):
    client_host = request.client.host if request.client else None
    if not _is_loopback_host(client_host):
        return PlainTextResponse("Access denied: localhost only.", status_code=403)
    return await call_next(request)


def _load_dashboard_html() -> str:
    dashboard_path = Path(__file__).resolve().parent / "dashboard.html"
    if not dashboard_path.exists():
        return (
            "<html><body><h3>Dashboard not found</h3>"
            "<p>Create app/dashboard.html to enable the local dashboard.</p>"
            "</body></html>"
        )
    return dashboard_path.read_text(encoding="utf-8")


@app.get("/", include_in_schema=False)
async def root():
    return {
        "name": "Local Deepfake Detection API",
        "dashboard": "/dashboard",
        "docs": "/docs",
    }


@app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def dashboard():
    return HTMLResponse(content=_load_dashboard_html())


@app.get("/health")
async def health_check():
    warnings = Config.validate()
    video_model_candidates = [str(path) for path in Config.get_video_model_search_dirs()]

    return {
        "status": "healthy",
        "local_only_mode": Config.LOCAL_ONLY_MODE,
        "device": str(Config.DEVICE),
        "audio_enabled": Config.ENABLE_AUDIO,
        "video_model_candidates": video_model_candidates,
        "dataset_path": str(Config.DATASET_ROOT),
        "image_dataset_path": str(Config.IMAGE_DATASET_ROOT),
        "warnings": warnings,
    }


@app.get("/config")
async def get_config():
    return {
        "api_host": Config.API_HOST,
        "api_port": Config.API_PORT,
        "local_only_mode": Config.LOCAL_ONLY_MODE,
        "frame_sample_rate": Config.FRAME_SAMPLE_RATE,
        "max_frames": Config.MAX_FRAMES,
        "face_image_size": Config.FACE_IMAGE_SIZE,
        "face_confidence_threshold": Config.FACE_CONFIDENCE_THRESHOLD,
        "audio_enabled": Config.ENABLE_AUDIO,
        "audio_sample_rate": Config.AUDIO_SAMPLE_RATE,
        "audio_max_duration": Config.AUDIO_MAX_DURATION,
        "fusion_video_weight": Config.FUSION_VIDEO_WEIGHT,
        "fusion_audio_weight": Config.FUSION_AUDIO_WEIGHT,
        "fusion_threshold": Config.FUSION_THRESHOLD,
        "allowed_video_extensions": sorted(list(Config.ALLOWED_EXTENSIONS)),
        "allowed_image_extensions": sorted(list(Config.ALLOWED_IMAGE_EXTENSIONS)),
        "max_upload_size_mb": Config.MAX_UPLOAD_SIZE_MB,
        "max_image_upload_size_mb": Config.MAX_IMAGE_UPLOAD_SIZE_MB,
    }


async def _save_upload_file(upload: UploadFile, max_size_mb: int, temp_suffix: str) -> str:
    if not upload.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    unique_id = uuid.uuid4().hex[:12]
    temp_name = f"{unique_id}_{temp_suffix}_{Path(upload.filename).name}"
    temp_path = Config.TEMP_DIR / temp_name
    Config.TEMP_DIR.mkdir(parents=True, exist_ok=True)

    content = await upload.read()
    max_bytes = max_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File too large. Max size is {max_size_mb}MB")

    with open(temp_path, "wb") as handle:
        handle.write(content)
    return str(temp_path)


@app.post("/predict")
async def predict_video_alias(
    video: UploadFile = File(..., description="Video file to analyze"),
    fusion_method: str = Query(default="rule_based"),
    skip_audio: bool = Query(default=False),
):
    return await predict_video(video=video, fusion_method=fusion_method, skip_audio=skip_audio)


@app.post("/predict/video")
async def predict_video(
    video: UploadFile = File(..., description="Video file to analyze"),
    fusion_method: str = Query(default="rule_based"),
    skip_audio: bool = Query(default=False),
):
    from src.audio_pipeline import run_audio_pipeline
    from src.fusion import run_fusion
    from src.video_pipeline import run_video_pipeline

    if fusion_method not in {"rule_based", "ml_based"}:
        raise HTTPException(status_code=400, detail="fusion_method must be 'rule_based' or 'ml_based'")

    extension = Path(video.filename or "").suffix.lower()
    if extension not in Config.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format: {extension}. Allowed: {sorted(list(Config.ALLOWED_EXTENSIONS))}",
        )

    total_start = time.time()
    temp_path = None
    try:
        temp_path = await _save_upload_file(
            upload=video,
            max_size_mb=Config.MAX_UPLOAD_SIZE_MB,
            temp_suffix="video",
        )
        logger.info(f"Received video: {video.filename}")

        video_result = run_video_pipeline(temp_path)
        if skip_audio:
            audio_result = {
                "status": "skipped",
                "prediction": {
                    "label": "UNKNOWN",
                    "confidence": 0.0,
                    "fake_probability": 0.5,
                    "real_probability": 0.5,
                    "note": "Audio skipped by request",
                },
            }
        else:
            audio_result = run_audio_pipeline(temp_path)

        final_result = run_fusion(
            video_result=video_result,
            audio_result=audio_result,
            method=fusion_method,
        )

        response = {
            "filename": video.filename,
            "video": {
                "status": video_result.get("status"),
                "prediction": video_result.get("prediction"),
                "video_info": video_result.get("video_info"),
                "timing": video_result.get("timing"),
                "error": video_result.get("error"),
            },
            "audio": {
                "status": audio_result.get("status"),
                "prediction": audio_result.get("prediction"),
                "timing": audio_result.get("timing"),
                "error": audio_result.get("error"),
            },
            "final": {
                "label": final_result.get("label"),
                "confidence": final_result.get("confidence"),
                "fake_probability": final_result.get("fake_probability"),
                "real_probability": final_result.get("real_probability"),
                "fusion_method": final_result.get("method"),
                "note": final_result.get("note"),
            },
            "total_processing_time": round(time.time() - total_start, 3),
        }
        return JSONResponse(content=response)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Video prediction error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {exc}")
    finally:
        safe_cleanup(temp_path)


@app.post("/predict/image")
async def predict_image(
    image: UploadFile = File(..., description="Image file to analyze"),
    face_confidence: Optional[float] = Query(default=None),
):
    from PIL import Image

    from src.image_pipeline import run_image_pipeline

    extension = Path(image.filename or "").suffix.lower()
    if extension not in Config.ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image format: {extension}. Allowed: {sorted(list(Config.ALLOWED_IMAGE_EXTENSIONS))}",
        )

    total_start = time.time()
    temp_path = None
    try:
        temp_path = await _save_upload_file(
            upload=image,
            max_size_mb=Config.MAX_IMAGE_UPLOAD_SIZE_MB,
            temp_suffix="image",
        )

        pil_image = Image.open(temp_path).convert("RGB")
        image_result = run_image_pipeline(
            image=pil_image,
            face_confidence=face_confidence,
        )

        response = {
            "filename": image.filename,
            "image": {
                "status": image_result.get("status"),
                "face_detected": image_result.get("face_detected"),
                "prediction": image_result.get("prediction"),
                "timing": image_result.get("timing"),
                "error": image_result.get("error"),
            },
            "total_processing_time": round(time.time() - total_start, 3),
        }
        return JSONResponse(content=response)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Image prediction error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {exc}")
    finally:
        safe_cleanup(temp_path)
