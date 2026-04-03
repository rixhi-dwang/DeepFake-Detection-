# Local Multimodal Deepfake Detection

Local-first deepfake detection system with:
- Video analysis (frame extraction, face detection, ViT inference)
- Image analysis (single image face-first pipeline)
- Optional audio analysis (Wav2Vec2 + classifier head)
- Multimodal fusion (rule-based by default, optional ML-based fusion)
- FastAPI + local dashboard for interactive testing
- Training pipelines for both video-native and image-converted workflows

The project is designed to run fully offline by default.

## Table Of Contents
- [1) Key Features](#1-key-features)
- [2) Local-Only Design](#2-local-only-design)
- [3) System Architecture](#3-system-architecture)
- [4) Repository Structure](#4-repository-structure)
- [5) Prerequisites](#5-prerequisites)
- [6) Installation](#6-installation)
- [7) Dataset Preparation](#7-dataset-preparation)
- [8) Training Process](#8-training-process)
- [9) Inference And Serving](#9-inference-and-serving)
- [10) API Reference](#10-api-reference)
- [11) Model Discovery And Artifacts](#11-model-discovery-and-artifacts)
- [12) Configuration](#12-configuration)
- [13) Evaluation](#13-evaluation)
- [14) Troubleshooting](#14-troubleshooting)
- [15) Notes And Limitations](#15-notes-and-limitations)

## 1) Key Features

- Local-only by default (no remote model downloads in runtime path).
- End-to-end video pipeline with timing breakdown:
  - frame extraction
  - face detection
  - model inference
  - aggregation
- Robust fallback behavior:
  - no face detected -> center crop fallback
  - missing audio track -> partial audio result with neutral probability
- Two training tracks:
  - `training/train.py` for Hugging Face image classifier fine-tuning (recommended for API compatibility).
  - `training/train_vit.py` for advanced torchvision ViT experiments with richer metrics/checkpoint controls.
- Dashboard at `/dashboard` and OpenAPI docs at `/docs`.

## 2) Local-Only Design

The runtime is intentionally constrained for offline/local operation:

- `DEEPFAKE_LOCAL_ONLY=1` by default.
- Offline environment flags are set automatically:
  - `HF_HUB_OFFLINE=1`
  - `TRANSFORMERS_OFFLINE=1`
  - `HF_DATASETS_OFFLINE=1`
- API host defaults to `127.0.0.1`.
- Middleware denies non-loopback clients.
- CORS allows localhost origins only.

This means you must place model files locally before inference/training.

## 3) System Architecture

### 3.1 Inference Architecture

```text
Client (Dashboard / API caller)
        |
        v
FastAPI (app/api.py)
  |-- /predict/image -> src/image_pipeline.py
  |-- /predict/video -> src/video_pipeline.py
  |                     + optional src/audio_pipeline.py
  |                     + src/fusion.py
  '-- /health, /config, /dashboard
```

### 3.2 Video Pipeline (`src/video_pipeline.py`)

1. Validate file and extension.
2. Extract sampled frames with OpenCV (`extract_frames`).
3. Detect faces with MTCNN (`detect_faces`).
4. Fallback to center-crop frames if no faces are found.
5. Run ViT inference in batches (`models/video_model.py`).
6. Aggregate frame-level probabilities into one video-level prediction.

Output includes:
- label/confidence/probabilities
- frame statistics (`num_frames_analyzed`, `fake_frame_ratio`, etc.)
- timing per stage

### 3.3 Image Pipeline (`src/image_pipeline.py`)

1. Convert input to RGB.
2. Run face detection on the image.
3. If face found: resize detected face.
4. If no face: center-crop fallback.
5. Run single-image ViT inference.

### 3.4 Audio Pipeline (`src/audio_pipeline.py`)

Audio is optional and disabled by default.

1. Extract audio from video via `ffmpeg`.
2. Load waveform using `librosa`.
3. Compute log-Mel spectrogram (for diagnostics/timing).
4. Extract Wav2Vec2 features and classify via local classifier head.

If audio is disabled or unavailable, the pipeline returns a neutral/partial result instead of crashing full video inference.

### 3.5 Fusion (`src/fusion.py`)

Supports:
- Rule-based weighted average (default):
  - `fused_fake = video_w * video_fake + audio_w * audio_fake`
- Optional ML fusion (logistic regression) when a trained fusion model is supplied.

Edge-case handling:
- both failed -> `UNKNOWN`
- video failed -> audio-only output
- audio failed/partial -> primarily video result

### 3.6 Training Architecture

There are two complementary training workflows:

- Workflow A: `training/train.py`
  - Uses local Hugging Face image-classification model directories.
  - Saves `best_model/` and `final_model/` in HF format.
  - These artifacts are directly compatible with runtime model loader.

- Workflow B: `training/train_vit.py`
  - Uses torchvision ViT-B/16 classifier checkpoints (`.pt`).
  - Advanced experiment controls (balanced sampling, early stopping, selection metric).
  - Great for research iteration; requires conversion/adaptation if you want direct API loading.

## 4) Repository Structure

```text
app/
  api.py                  # FastAPI routes and local-only middleware
  dashboard.html          # Local web UI

models/
  video_model.py          # ViT load + image/frame inference + aggregation
  audio_model.py          # Wav2Vec2 + classifier head for audio

src/
  video_pipeline.py       # End-to-end video inference
  image_pipeline.py       # End-to-end image inference
  audio_pipeline.py       # End-to-end audio inference
  fusion.py               # Multimodal fusion logic
  preprocessing.py        # frame extraction + MTCNN face detection
  extract_frames.py       # dataset frame extraction utility
  face_detection.py       # detector abstraction (MTCNN + Haar fallback)
  utils.py                # logging, file helpers, metadata helpers
  config.py               # frame extraction specific config (separate from root)

training/
  prepare_image_dataset.py # convert videos -> image dataset (train/val)
  dataset.py               # video-mode dataset and loaders
  image_dataset.py         # image-mode dataset and loaders
  train.py                 # local fine-tuning entrypoint (HF model flow)
  train_vit.py             # advanced ViT training on extracted frames
  evaluate.py              # evaluation/reporting helpers

config.py                 # global runtime/training config
run_api.py                # API launcher
requirements.txt
```

## 5) Prerequisites

- Python 3.10+ recommended
- OS: Windows/Linux/macOS
- Optional but recommended: NVIDIA GPU + CUDA for faster training/inference
- External tool for audio extraction:
  - `ffmpeg` must be installed and available in `PATH`

## 6) Installation

```bash
git clone <your-repo-url>
cd "project exhibition 2"
python -m venv venv
```

Windows PowerShell:
```bash
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux/macOS:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

Depending on your environment, you may also need:
- `fastapi`
- `uvicorn`
- `facenet-pytorch`
- `librosa`
- `scikit-learn`
- `seaborn`

## 7) Dataset Preparation

### 7.1 Expected FaceForensics++ Layout

`config.py` expects:

```text
data/
  FaceForensics++_C23/
    original/
    Deepfakes/
    Face2Face/
    FaceSwap/
    FaceShifter/
    NeuralTextures/
```

### 7.2 Convert Videos To Image Dataset (Recommended For `train.py --data_mode images`)

```bash
python training/prepare_image_dataset.py ^
  --methods Deepfakes Face2Face FaceSwap FaceShifter NeuralTextures ^
  --frames_per_video 5 ^
  --frame_sample_rate 10 ^
  --face_confidence 0.8 ^
  --train_ratio 0.8 ^
  --output_dir converted_images
```

Default output:

```text
converted_images/
  train/
    REAL/
    FAKE/
  val/
    REAL/
    FAKE/
  metadata.csv
  metadata.json
  summary.json
  conversion_config.json
```

### 7.3 Alternative Frame Extraction Pipeline (`src/extract_frames.py`)

For frame-folder workflows (used by `train_vit.py`):

```bash
python -m src.extract_frames ^
  --dataset-path data/FaceForensics++_C23 ^
  --output-path data/frames ^
  --frame-skip 10 ^
  --max-frames 30 ^
  --image-size 224 ^
  --num-workers 4
```

Produces class folders under `data/frames/real` and `data/frames/fake` with metadata logging.

## 8) Training Process

### 8.1 Workflow A: Train HF-Compatible Model (`training/train.py`)

This workflow is the most direct path to API deployment.

### Image-mode training (recommended)

```bash
python training/train.py ^
  --data_mode images ^
  --image_dataset_dir converted_images ^
  --epochs 10 ^
  --batch_size 16 ^
  --eval_batch_size 8 ^
  --lr 1e-5 ^
  --weight_decay 0.01 ^
  --scheduler cosine ^
  --seed 42
```

### Video-mode training (on-the-fly extraction)

```bash
python training/train.py ^
  --data_mode video ^
  --methods Deepfakes Face2Face FaceSwap ^
  --max_videos 200 ^
  --frames_per_video 3 ^
  --frame_sample_rate 10 ^
  --face_confidence 0.8
```

### What this script does internally

1. Load base model locally using `models/video_model.load_video_model`.
2. Build train/val loaders from either:
  - `training/image_dataset.py` (images mode), or
  - `training/dataset.py` (video mode).
3. Train with AdamW (+ optional cosine schedule).
4. Save best checkpoint by validation accuracy.
5. Save final checkpoint and training history.

### Artifacts

```text
training_runs/<run_name>/
  best_model/             # HF model dir (API-compatible)
  final_model/            # HF model dir (API-compatible)
  training_config.json
  training_history.json
```

### 8.2 Workflow B: Advanced ViT Experiments (`training/train_vit.py`)

Use this when you need richer experiment controls and reporting.

Example:

```bash
python training/train_vit.py ^
  --frames-root data/frames ^
  --run-name vit_stage1 ^
  --epochs 20 ^
  --batch-size 16 ^
  --eval-batch-size 32 ^
  --lr 1.5e-4 ^
  --balanced-sampling ^
  --class-weighted-loss ^
  --amp
```

Key capabilities:
- Video-level split (avoids leakage)
- Balanced sampling (WeightedRandomSampler)
- Class-weighted loss
- AMP mixed precision
- Early stopping by selected metric
- Resume from checkpoint

Artifacts:

```text
training_runs/<run_name>/
  checkpoints/
    best.pt
    final.pt
    epoch_XXX.pt
  config.json
  split_summary.json
  samples_manifest.csv
  history.json
  history.csv
  summary.json
  train.log
```

### 8.3 Recommended Practical Training Loop

1. Convert FF++ videos -> image dataset (`prepare_image_dataset.py`).
2. Run `training/train.py --data_mode images`.
3. Validate with API `/predict/image` and `/predict/video`.
4. Iterate hyperparameters:
  - `--lr`
  - `--batch_size`
  - `--max_images_per_class`
  - `--freeze_backbone`
5. Promote best run to production model path (`checkpoints/best_model`).

## 9) Inference And Serving

Start API:

```bash
python run_api.py
```

Local endpoints:
- Dashboard: `http://127.0.0.1:8000/dashboard`
- Docs: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/health`

To enable reload in development:

```bash
python run_api.py --reload
```

## 10) API Reference

### 10.1 `GET /health`

Returns status, device, warnings, and discovered model candidate directories.

### 10.2 `GET /config`

Returns runtime config values (upload limits, thresholds, sample rates, etc.).

### 10.3 `POST /predict/video`

Form field:
- `video`: uploaded video file

Query params:
- `fusion_method`: `rule_based` or `ml_based`
- `skip_audio`: `true|false`

Also available as alias: `POST /predict`

### 10.4 `POST /predict/image`

Form field:
- `image`: uploaded image file

Query params:
- `face_confidence` (optional float)

### 10.5 Example `curl`

Video:
```bash
curl -X POST "http://127.0.0.1:8000/predict/video?fusion_method=rule_based&skip_audio=true" ^
  -F "video=@path\\to\\sample.mp4"
```

Image:
```bash
curl -X POST "http://127.0.0.1:8000/predict/image?face_confidence=0.85" ^
  -F "image=@path\\to\\sample.jpg"
```

## 11) Model Discovery And Artifacts

Runtime model search order is dynamic and prefers newest training runs first.

High-level priority:
1. `training_runs/*/best_model`
2. `training_runs/*/final_model`
3. `checkpoints/best_model`
4. `checkpoints/final_model`
5. `pretrained/`
6. `pretrained/checkpoint-5252/`

Each candidate must contain at least:
- `model.safetensors`
- `config.json`
- `preprocessor_config.json` (expected by processor loading)

### 11.1 Audio model files (optional)

When `DEEPFAKE_ENABLE_AUDIO=1`, the following must exist:

```text
pretrained/audio_model/
  config.json
  model.safetensors (or pytorch_model.bin)
  audio_classifier.pt
```

Enable audio:

Windows PowerShell:
```bash
$env:DEEPFAKE_ENABLE_AUDIO=1
python run_api.py
```

Linux/macOS:
```bash
export DEEPFAKE_ENABLE_AUDIO=1
python run_api.py
```

## 12) Configuration

Main settings live in root `config.py`.

Important values:
- Data:
  - `DATASET_ROOT`
  - `IMAGE_DATASET_ROOT`
- Pipeline:
  - `FRAME_SAMPLE_RATE`
  - `MAX_FRAMES`
  - `FACE_CONFIDENCE_THRESHOLD`
- Fusion:
  - `FUSION_VIDEO_WEIGHT`
  - `FUSION_AUDIO_WEIGHT`
  - `FUSION_THRESHOLD`
- API:
  - `API_HOST`
  - `API_PORT`
  - `MAX_UPLOAD_SIZE_MB`
  - `MAX_IMAGE_UPLOAD_SIZE_MB`

Useful environment variables:
- `DEEPFAKE_LOCAL_ONLY` (default `1`)
- `DEEPFAKE_ENABLE_AUDIO` (default `0`)

## 13) Evaluation

`training/evaluate.py` provides:
- Accuracy, precision, recall, F1
- Confusion matrix
- Classification report
- Optional confusion matrix plotting

Quick run (module has a built-in sample flow):

```bash
python training/evaluate.py
```

## 14) Troubleshooting

### "Unable to load local video model"
- Ensure a candidate directory contains:
  - `model.safetensors`
  - `config.json`
  - `preprocessor_config.json`
- Check `GET /health` for discovered candidate paths and warnings.

### "Access denied: localhost only"
- API is intentionally loopback-only.
- Use `127.0.0.1` or `localhost` from the same machine.

### Audio always disabled
- Set `DEEPFAKE_ENABLE_AUDIO=1`
- Install `ffmpeg` and verify audio model files exist.

### Face detection returns no faces
- Lower `face_confidence` on image endpoint.
- The pipeline will still run with center-crop fallback.

### Training crashes with empty dataset
- Confirm dataset directories exist and are populated.
- For image mode, verify `converted_images/train/{REAL,FAKE}` and `converted_images/val/{REAL,FAKE}`.

## 15) Notes And Limitations

- This system is optimized for local experimentation, not hardened production deployment.
- Detection quality depends heavily on:
  - training data quality/balance
  - domain shift between train and inference media
  - face visibility and compression artifacts
- Audio branch is optional and can be disabled without breaking video/image inference.
- `training/train_vit.py` checkpoints are not directly loaded by runtime API without adaptation.

---

If you want, this README can be split next into:
- `docs/architecture.md` (deep technical detail)
- `docs/training.md` (reproducible experiments)
- `docs/api.md` (clean endpoint contract for frontend/integration work)
