# Multimodal Deepfake Detection (Local-Only)

This project now runs in **local-only mode** by default:
- No remote model downloads.
- API binds to localhost.
- Dashboard uses same-origin localhost endpoints only.

## Project Layout

```
app/
  api.py
  dashboard.html
config.py
models/
  video_model.py
  audio_model.py
run_api.py
src/
  video_pipeline.py
  image_pipeline.py
  audio_pipeline.py
training/
  dataset.py
  image_dataset.py
  prepare_image_dataset.py
  train.py
```

## Start API (localhost only)

```bash
python run_api.py
```

Routes:
- `GET /dashboard`
- `POST /predict/video` (or `/predict`)
- `POST /predict/image`
- `GET /health`

## Local Model Requirements

At least one local video model directory must exist with:
- `model.safetensors`
- `config.json`
- `preprocessor_config.json`

Checked in this order:
1. `checkpoints/best_model/`
2. `checkpoints/final_model/`
3. `pretrained/`
4. `pretrained/checkpoint-5252/`

## Convert FaceForensics++ Videos to Images

```bash
python training/prepare_image_dataset.py ^
  --methods Deepfakes Face2Face FaceSwap FaceShifter NeuralTextures ^
  --frames_per_video 5 ^
  --frame_sample_rate 10 ^
  --train_ratio 0.8
```

Output (default):
```
converted_images/
  train/REAL
  train/FAKE
  val/REAL
  val/FAKE
  metadata.csv
  summary.json
```

## Train with Full CLI Control

### Image-mode training

```bash
python training/train.py ^
  --data_mode images ^
  --image_dataset_dir converted_images ^
  --epochs 10 ^
  --batch_size 16 ^
  --eval_batch_size 8 ^
  --lr 1e-5 ^
  --weight_decay 0.01 ^
  --max_grad_norm 1.0 ^
  --scheduler cosine ^
  --seed 42
```

Default training artifacts are written under `training_runs/`.

### Video-mode training

```bash
python training/train.py ^
  --data_mode video ^
  --methods Deepfakes Face2Face ^
  --max_videos 200 ^
  --frames_per_video 3 ^
  --frame_sample_rate 10 ^
  --face_confidence 0.8
```

Useful options:
- `--model_dir` local checkpoint to start from
- `--output_dir` training artifact root
- `--run_name` explicit run folder name
- `--freeze_backbone`
- `--num_workers`
- `--max_images_per_class`
- `--no_augment`

## Audio Notes

Audio is disabled by default (`DEEPFAKE_ENABLE_AUDIO=0`) in local-only mode.

To enable audio locally, provide:
- `pretrained/audio_model/` (local Wav2Vec2 files)
- `pretrained/audio_model/audio_classifier.pt`

Then run with:
```bash
set DEEPFAKE_ENABLE_AUDIO=1
python run_api.py
```
