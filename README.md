# RVC Inference Server

Minimal, production-ready voice conversion inference service using RVC (Retrieval-based Voice Conversion) models.

**No training. No UI. Just inference.**

## Features

- ✅ Load RVC models from `.pth` and `.index` files
- ✅ RESTful API with FastAPI
- ✅ Support multiple models simultaneously
- ✅ Pitch adjustment (semitones)
- ✅ Multiple F0 extraction methods
- ✅ CPU and GPU support (auto-detect)
- ✅ Docker-ready with minimal image size

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set models directory
export MODELS_DIR=/path/to/your/models

# Run server
python rvc_inference_server.py
```

### Docker

```bash
# Build image
docker build -t rvc-inference .

# Run container
docker run -p 8003:8003 -v /path/to/models:/models rvc-inference
```

## API Endpoints

### Health Check

```http
GET /health
```

Returns server status and loaded models.

**Response:**

```json
{
  "status": "healthy",
  "loaded_models": ["model1", "model2"],
  "available_models": ["model1", "model2", "model3"],
  "models_dir": "/models"
}
```

### List Voices

```http
GET /voices
```

Returns all available voice models.

**Response:**

```json
{
  "voices": [
    { "name": "model1", "loaded": true, "has_index": true },
    { "name": "model2", "loaded": false, "has_index": false }
  ],
  "total": 2
}
```

### Convert Audio

```http
POST /convert
Content-Type: multipart/form-data
```

**Parameters:**

| Name         | Type   | Required | Default | Description                         |
| ------------ | ------ | -------- | ------- | ----------------------------------- |
| `audio`      | file   | ✅       | -       | Audio file (WAV/MP3/OGG)            |
| `voice_name` | string | ✅       | -       | Model name to use                   |
| `f0_up_key`  | int    | ❌       | 0       | Pitch shift (-24 to +24 semitones)  |
| `f0_method`  | string | ❌       | harvest | Pitch extraction: `harvest` or `pm` |

**Response:** WAV audio file with headers:

- `X-Processing-Time`: Processing time in seconds
- `X-Input-Duration`: Input audio duration
- `X-Output-Sample-Rate`: Output sample rate
- `X-Model-Name`: Model used

**Example (curl):**

```bash
curl -X POST http://localhost:8003/convert \
  -F "audio=@input.wav" \
  -F "voice_name=my_model" \
  -F "f0_up_key=2" \
  -F "f0_method=harvest" \
  --output converted.wav
```

**Example (Python):**

```python
import requests

with open("input.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8003/convert",
        files={"audio": f},
        data={
            "voice_name": "my_model",
            "f0_up_key": 2,
            "f0_method": "harvest"
        }
    )

with open("output.wav", "wb") as f:
    f.write(response.content)
```

### Load/Unload Models

```http
POST /load
POST /unload
```

Manually load or unload models to manage memory.

## Model Setup

### Directory Structure

Place your models in the models directory:

```
/models/
├── voice1.pth
├── voice1.index      # Optional but recommended
├── voice2/
│   ├── model.pth
│   └── model.index
└── voice3.pth
```

### Model Requirements

- **`.pth` file**: PyTorch checkpoint with model weights
- **`.index` file** (optional): FAISS index for speaker embeddings

## Configuration

Environment variables:

| Variable             | Default   | Description                            |
| -------------------- | --------- | -------------------------------------- |
| `MODELS_DIR`         | `/models` | Directory containing model files       |
| `HOST`               | `0.0.0.0` | Server host                            |
| `PORT`               | `8003`    | Server port                            |
| `MAX_AUDIO_DURATION` | `300`     | Maximum audio length (seconds)         |
| `PRELOAD_MODELS`     | ``        | Comma-separated model names to preload |

## Testing

```bash
# Install test dependencies
pip install requests

# Run tests
python test_inference.py --url http://localhost:8003 --voice my_model

# Run all tests
python test_inference.py --all --voice my_model

# Test with custom audio
python test_inference.py --voice my_model --audio input.wav --pitch 3
```

## Performance

| Metric            | Target            | Notes            |
| ----------------- | ----------------- | ---------------- |
| Docker image size | < 3GB             | CPU version ~2GB |
| Model load time   | < 10s             | First load only  |
| Inference (CPU)   | < 2s per second   | 10s audio → ~20s |
| Inference (GPU)   | < 0.5s per second | 10s audio → ~5s  |

## Troubleshooting

### Out of Memory

- Reduce `MAX_AUDIO_DURATION`
- Process shorter audio clips
- Unload unused models with `/unload`

### Slow Inference

- Use `pm` instead of `harvest` for faster F0 extraction
- Use GPU if available
- Reduce audio sample rate

### Model Not Found

- Check `MODELS_DIR` environment variable
- Ensure `.pth` file exists
- Check file permissions

## Architecture

```
rvc-infer/
├── rvc_inference_server.py    # FastAPI application
├── models/
│   └── rvc_model.py           # Model loading & inference
├── utils/
│   └── audio_utils.py         # Audio processing
├── Dockerfile                  # Docker configuration
├── requirements.txt           # Dependencies
└── test_inference.py          # Test script
```

## License

MIT License
