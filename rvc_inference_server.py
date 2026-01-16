"""
RVC Inference Server - Minimal FastAPI service for voice conversion.
Provides RESTful API for RVC model inference without training or UI.
"""

import io
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from models.rvc_model import model_manager
from utils.audio_utils import (
    audio_to_bytes,
    get_audio_duration,
    load_audio,
    normalize_audio,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration from environment
MODELS_DIR = os.getenv("MODELS_DIR", "/models")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8003"))
MAX_AUDIO_DURATION = float(os.getenv("MAX_AUDIO_DURATION", "300"))  # 5 minutes
DEFAULT_SAMPLE_RATE = 16000


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    # Startup
    logger.info(f"Starting RVC Inference Server...")
    logger.info(f"Models directory: {MODELS_DIR}")
    
    # Set models directory
    model_manager.set_models_dir(MODELS_DIR)
    
    # Scan for available models
    available = model_manager.scan_models()
    logger.info(f"Found {len(available)} available models: {list(available.keys())}")
    
    # Optionally preload models
    preload = os.getenv("PRELOAD_MODELS", "").split(",")
    for model_name in preload:
        model_name = model_name.strip()
        if model_name and model_name in available:
            try:
                logger.info(f"Preloading model: {model_name}")
                model_manager.load(model_name)
            except Exception as e:
                logger.error(f"Failed to preload model {model_name}: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down RVC Inference Server...")
    model_manager.unload_all()


# Create FastAPI app
app = FastAPI(
    title="RVC Inference Server",
    description="Minimal voice conversion inference service using RVC models",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    Returns server status and list of loaded models.
    """
    return {
        "status": "healthy",
        "loaded_models": model_manager.list_models(),
        "available_models": list(model_manager.list_available().keys()),
        "models_dir": MODELS_DIR
    }


@app.get("/voices")
async def list_voices():
    """
    List available voice models.
    Returns both loaded and unloaded models with their file paths.
    """
    available = model_manager.list_available()
    
    voices = []
    for name, info in available.items():
        voices.append({
            "name": name,
            "loaded": info.get("loaded", False),
            "has_index": info.get("index") is not None
        })
    
    return {
        "voices": voices,
        "total": len(voices)
    }


@app.post("/convert")
async def convert_audio(
    audio: UploadFile = File(..., description="Audio file to convert (WAV/MP3/OGG)"),
    voice_name: str = Form(..., description="Name of the voice model to use"),
    f0_up_key: int = Form(0, description="Pitch adjustment in semitones (-12 to +12)"),
    f0_method: str = Form("harvest", description="F0 extraction method: harvest or pm")
):
    """
    Convert audio using specified RVC voice model.
    
    Args:
        audio: Input audio file (WAV, MP3, or OGG format)
        voice_name: Name of the voice model to use
        f0_up_key: Pitch shift in semitones (default: 0)
        f0_method: Pitch extraction method - 'harvest' or 'pm' (default: harvest)
    
    Returns:
        Converted audio as WAV file
    """
    start_time = time.time()
    
    # Validate f0_method
    if f0_method not in ["harvest", "pm"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid f0_method: {f0_method}. Must be 'harvest' or 'pm'"
        )
    
    # Validate pitch range
    if not -24 <= f0_up_key <= 24:
        raise HTTPException(
            status_code=400,
            detail=f"f0_up_key must be between -24 and +24 semitones"
        )
    
    # Load model
    try:
        model = model_manager.get(voice_name)
        if model is None:
            # Try to load it
            model = model_manager.load(voice_name)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Voice model '{voice_name}' not found"
        )
    except Exception as e:
        logger.error(f"Failed to load model {voice_name}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model: {str(e)}"
        )
    
    # Read uploaded audio
    try:
        audio_bytes = await audio.read()
        audio_buffer = io.BytesIO(audio_bytes)
        
        # Load and process audio
        audio_data, sample_rate = load_audio(
            audio_buffer,
            target_sr=model.sample_rate
        )
    except Exception as e:
        logger.error(f"Failed to load audio: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to load audio file: {str(e)}"
        )
    
    # Check audio duration
    duration = get_audio_duration(audio_data, sample_rate)
    if duration > MAX_AUDIO_DURATION:
        raise HTTPException(
            status_code=400,
            detail=f"Audio too long: {duration:.1f}s (max: {MAX_AUDIO_DURATION}s)"
        )
    
    logger.info(f"Processing audio: {duration:.2f}s, model: {voice_name}, pitch: {f0_up_key}")
    
    # Perform inference
    try:
        output_audio = model.infer(
            audio_data,
            f0_up_key=f0_up_key,
            f0_method=f0_method
        )
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Voice conversion failed: {str(e)}"
        )
    
    # Normalize output
    output_audio = normalize_audio(output_audio)
    
    # Convert to WAV bytes
    output_bytes = audio_to_bytes(output_audio, model.sample_rate)
    
    processing_time = time.time() - start_time
    logger.info(f"Conversion complete in {processing_time:.2f}s")
    
    # Return audio with metadata headers
    return Response(
        content=output_bytes,
        media_type="audio/wav",
        headers={
            "X-Processing-Time": f"{processing_time:.3f}",
            "X-Input-Duration": f"{duration:.3f}",
            "X-Output-Sample-Rate": str(model.sample_rate),
            "X-Model-Name": voice_name,
            "Content-Disposition": f'attachment; filename="converted_{voice_name}.wav"'
        }
    )


@app.post("/load")
async def load_model(
    voice_name: str = Form(..., description="Name of the voice model to load")
):
    """
    Preload a voice model into memory.
    
    Args:
        voice_name: Name of the voice model to load
    
    Returns:
        Load status
    """
    try:
        model = model_manager.load(voice_name)
        return {
            "status": "loaded",
            "voice_name": voice_name,
            "version": model.version,
            "sample_rate": model.sample_rate
        }
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Voice model '{voice_name}' not found"
        )
    except Exception as e:
        logger.error(f"Failed to load model {voice_name}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model: {str(e)}"
        )


@app.post("/unload")
async def unload_model(
    voice_name: str = Form(..., description="Name of the voice model to unload")
):
    """
    Unload a voice model from memory.
    
    Args:
        voice_name: Name of the voice model to unload
    
    Returns:
        Unload status
    """
    if model_manager.unload(voice_name):
        return {"status": "unloaded", "voice_name": voice_name}
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Voice model '{voice_name}' not loaded"
        )


if __name__ == "__main__":
    uvicorn.run(
        "rvc_inference_server:app",
        host=HOST,
        port=PORT,
        reload=False,
        log_level="info"
    )
