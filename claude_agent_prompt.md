# Prompt untuk Claude Sonnet Opus 4.5 - RVC Inference Builder

## Objective
Build a **minimal, production-ready RVC (Retrieval-based Voice Conversion) inference service** that ONLY handles voice conversion inference. No training, no UI, no bloat.

## Core Requirements

### 1. Functionality
- Load RVC models from `.pth` (PyTorch checkpoint) and `.index` (FAISS speaker embeddings) files
- Perform voice conversion inference on audio input
- Return converted audio output
- Support multiple models loaded simultaneously
- RESTful API with FastAPI

### 2. Input/Output Specifications
**Input:**
- Audio file (WAV/MP3/OGG)
- Model name (string)
- Optional: pitch adjustment (semitones, default: 0)
- Optional: F0 extraction method (harvest/crepe/pm, default: harvest)

**Output:**
- Converted audio as WAV format (16kHz or 48kHz)
- Processing time metadata

### 3. Technical Constraints
**CRITICAL - Keep it MINIMAL:**
- Docker image size: **< 3GB** (target: ~2GB)
- Dependencies: **< 10 packages** (only what's absolutely necessary)
- Python 3.10
- Support both CPU and GPU (auto-detect)
- No training code
- No Web UI
- No unnecessary model downloads

**Core Dependencies (ONLY these):**
- `torch` (CPU version, or CUDA if GPU needed)
- `torchaudio`
- `librosa` (audio processing)
- `soundfile` (I/O)
- `faiss-cpu` or `faiss-gpu` (vector similarity search)
- `numpy`
- `fastapi` (API server)
- `uvicorn` (ASGI server)
- `python-multipart` (file upload)

### 4. Model Format Specifications
**`.pth` file contains:**
```python
{
    'model': <model_state_dict>,
    'config': {
        'sample_rate': 16000 or 48000,
        'hop_length': 160,
        'model_type': 'v1' or 'v2'
    },
    # Other metadata
}
```

**`.index` file:**
- FAISS index file (binary)
- Contains speaker embeddings for voice similarity matching

### 5. API Endpoints Required
```
GET  /health          -> Health check with loaded models list
GET  /voices          -> List available voice models
POST /convert         -> Convert audio with specified model
     - Form data: audio file, voice_name, f0_up_key (optional), f0_method (optional)
     - Returns: WAV audio file
```

### 6. Error Handling Requirements
- Graceful handling of missing models
- Audio format validation
- Memory management (avoid OOM on large files)
- Timeout handling for long conversions
- Clear error messages

### 7. Performance Requirements
- Model loading: < 10 seconds per model
- Inference latency: < 2 seconds per second of audio (CPU), < 0.5s (GPU)
- Support concurrent requests (async)
- Memory efficient (don't load all audio into RAM at once for large files)

### 8. Docker Requirements
**Dockerfile must:**
- Use Python 3.10-slim base image
- Multi-stage build to reduce size
- Mount `/models` volume for model files
- Expose port 8003
- Set environment variables to prevent memory issues:
  ```
  ENV OMP_NUM_THREADS=1
  ENV MKL_NUM_THREADS=1
  ENV PYTHONUNBUFFERED=1
  ```

### 9. Known Issues to Avoid
- **Double free error**: Use proper tensor type conversion (always `.float()` for FP32)
- **Mixed precision errors**: Force FP32, avoid FP16 unless explicitly supported
- **CUDA memory leaks**: Use `torch.cuda.empty_cache()` after inference
- **Slow cross-filesystem I/O**: Process files in `/tmp` if possible
- **Model download on first run**: Pre-download common models or document requirement

### 10. Testing Requirements
Provide a simple test script that:
- Uploads a test audio file
- Converts it with a loaded model
- Saves output
- Measures processing time

## Deliverables

1. **`rvc_inference_server.py`** - Main FastAPI server
2. **`Dockerfile`** - Optimized Docker build
3. **`requirements.txt`** - Pinned dependency versions
4. **`README.md`** - Setup and usage instructions
5. **`test_inference.py`** - Basic functionality test

## Implementation Guidelines

### Code Structure
```
rvc-infer/
├── Dockerfile
├── requirements.txt
├── rvc_inference_server.py    # Main server
├── models/                      # Model loading & inference
│   └── rvc_model.py
├── utils/                       # Audio processing utilities
│   └── audio_utils.py
├── test_inference.py
└── README.md
```

### Key Implementation Points

**Model Loading:**
```python
# Must handle both v1 and v2 model formats
# Must validate .pth and .index compatibility
# Cache models in memory (don't reload on every request)
```

**Audio Processing:**
```python
# Use librosa for loading (supports multiple formats)
# Resample to model's expected sample rate
# Chunk large files to prevent OOM
```

**Inference Pipeline:**
```python
# 1. Load audio
# 2. Extract F0 (pitch)
# 3. Apply pitch shift if requested
# 4. Run model inference
# 5. Post-process output
# 6. Return audio
```

**Error Recovery:**
```python
# Catch PyTorch errors gracefully
# Return meaningful HTTP status codes
# Log errors for debugging
```

## Success Criteria
- ✅ Docker image < 3GB
- ✅ Successfully loads .pth + .index models
- ✅ Converts audio in < 5 seconds (for 10s audio on CPU)
- ✅ No memory leaks after 100 conversions
- ✅ Handles concurrent requests
- ✅ Clear error messages
- ✅ Zero training/UI code

## Additional Context
- This will be part of a microservices architecture (STT → LLM → TTS → RVC)
- Will run in Docker on WSL2 with limited resources (laptop with 512GB SSD)
- User has models already trained, just needs inference
- Current full RVC Web UI is 38GB, causing storage and performance issues

## Notes for Agent
- Prioritize **simplicity** and **minimalism** over features
- If uncertain about model format, design for flexibility (detect format at runtime)
- Include type hints and docstrings
- Use async/await for I/O operations
- Don't reinvent the wheel - use proven libraries
- Test on CPU first, GPU is optional

---

**Start by creating the minimal viable implementation, then we can iterate based on testing.**
