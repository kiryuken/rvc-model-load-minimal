# =============================================================================
# RVC Inference Server - Production Dockerfile
# GPU-enabled Docker image for voice conversion inference
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder - Install dependencies
# -----------------------------------------------------------------------------
FROM nvidia/cuda:11.8-cudnn8-devel-ubuntu22.04 as builder

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install Python and build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    build-essential \
    git \
    cmake \
    libsndfile1-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# Copy requirements first for layer caching
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir \
    torch==2.1.0+cu118 \
    torchaudio==2.1.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Install fairseq for HuBERT
# Note: fairseq can be tricky to install, this uses the latest release
RUN pip install --no-cache-dir fairseq

# Install FAISS with GPU support
RUN pip install --no-cache-dir faiss-gpu

# Install remaining dependencies (excluding torch, fairseq, faiss)
RUN pip install --no-cache-dir \
    librosa>=0.10.0 \
    soundfile>=0.12.0 \
    numpy>=1.24.0 \
    scipy>=1.10.0 \
    fastapi>=0.100.0 \
    uvicorn[standard]>=0.22.0 \
    python-multipart>=0.0.6

# -----------------------------------------------------------------------------
# Stage 2: Runtime - Production image
# -----------------------------------------------------------------------------
FROM nvidia/cuda:11.8-cudnn8-runtime-ubuntu22.04 as runtime

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Set environment variables for optimal performance
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODELS_DIR=/models \
    HUBERT_PATH=/models/runtime/hubert_base.pt \
    HOST=0.0.0.0 \
    PORT=8003

# CUDA environment
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create symlinks for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY utils/ ./utils/
COPY models/ ./models/
COPY rvc_inference_server.py .

# Create models directories
RUN mkdir -p /models/voice /models/runtime

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app /models
USER appuser

# Expose port
EXPOSE 8003

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8003/health')" || exit 1

# Run the server
CMD ["python", "rvc_inference_server.py"]

# -----------------------------------------------------------------------------
# Usage:
# Build:   docker build -t rvc-inference .
# Run:     docker run --gpus all -p 8003:8003 \
#          -v /path/to/models/voice:/models/voice \
#          -v /path/to/models/runtime:/models/runtime \
#          rvc-inference
#
# Required model files:
#   /models/runtime/hubert_base.pt - HuBERT feature extractor
#   /models/voice/*.pth - RVC model weights
#   /models/voice/*.index - FAISS indices (optional but recommended)
# -----------------------------------------------------------------------------
