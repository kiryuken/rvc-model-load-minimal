"""
Audio utilities for RVC inference service.
Handles loading, saving, normalization, and pitch extraction.
"""

import io
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Union

import librosa
import numpy as np
import soundfile as sf


def load_audio(
    file_path: Union[str, Path, io.BytesIO],
    target_sr: int = 16000,
    mono: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Load audio from file and resample to target sample rate.
    
    Args:
        file_path: Path to audio file or BytesIO object
        target_sr: Target sample rate (default: 16000)
        mono: Convert to mono if True (default: True)
    
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    if isinstance(file_path, io.BytesIO):
        # Handle BytesIO - write to temp file first for librosa
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(file_path.read())
            tmp_path = tmp.name
        try:
            audio, sr = librosa.load(tmp_path, sr=target_sr, mono=mono)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    else:
        audio, sr = librosa.load(str(file_path), sr=target_sr, mono=mono)
    
    return audio.astype(np.float32), sr


def save_audio(
    audio: np.ndarray,
    file_path: Union[str, Path],
    sample_rate: int = 16000
) -> None:
    """
    Save audio array to WAV file.
    
    Args:
        audio: Audio data as numpy array
        file_path: Output file path
        sample_rate: Sample rate of the audio
    """
    # Ensure audio is in correct format
    audio = np.asarray(audio, dtype=np.float32)
    
    # Clip to prevent distortion
    audio = np.clip(audio, -1.0, 1.0)
    
    sf.write(str(file_path), audio, sample_rate, subtype='PCM_16')


def audio_to_bytes(audio: np.ndarray, sample_rate: int = 16000) -> bytes:
    """
    Convert audio array to WAV bytes.
    
    Args:
        audio: Audio data as numpy array
        sample_rate: Sample rate of the audio
    
    Returns:
        WAV file as bytes
    """
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format='WAV', subtype='PCM_16')
    buffer.seek(0)
    return buffer.read()


def normalize_audio(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """
    Normalize audio to target dB level.
    
    Args:
        audio: Input audio array
        target_db: Target dB level (default: -20.0)
    
    Returns:
        Normalized audio array
    """
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 0:
        target_rms = 10 ** (target_db / 20)
        audio = audio * (target_rms / rms)
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def chunk_audio(
    audio: np.ndarray,
    chunk_size: int,
    overlap: int = 0
) -> list:
    """
    Split audio into chunks for memory-efficient processing.
    
    Args:
        audio: Input audio array
        chunk_size: Size of each chunk in samples
        overlap: Overlap between chunks in samples
    
    Returns:
        List of audio chunks
    """
    chunks = []
    start = 0
    step = chunk_size - overlap
    
    while start < len(audio):
        end = min(start + chunk_size, len(audio))
        chunks.append(audio[start:end])
        start += step
    
    return chunks


def merge_chunks(
    chunks: list,
    overlap: int = 0
) -> np.ndarray:
    """
    Merge audio chunks back together with crossfade.
    
    Args:
        chunks: List of audio chunks
        overlap: Overlap between chunks in samples
    
    Returns:
        Merged audio array
    """
    if not chunks:
        return np.array([], dtype=np.float32)
    
    if len(chunks) == 1:
        return chunks[0]
    
    # Calculate total length
    total_length = sum(len(c) for c in chunks) - overlap * (len(chunks) - 1)
    result = np.zeros(total_length, dtype=np.float32)
    
    pos = 0
    for i, chunk in enumerate(chunks):
        if i == 0:
            result[:len(chunk)] = chunk
            pos = len(chunk) - overlap
        else:
            # Crossfade in overlap region
            fade_in = np.linspace(0, 1, overlap)
            fade_out = np.linspace(1, 0, overlap)
            
            if overlap > 0:
                result[pos:pos + overlap] = (
                    result[pos:pos + overlap] * fade_out +
                    chunk[:overlap] * fade_in
                )
            
            # Copy rest of chunk
            end_pos = pos + len(chunk)
            result[pos + overlap:end_pos] = chunk[overlap:]
            pos = end_pos - overlap
    
    return result


def extract_f0_pm(
    audio: np.ndarray,
    sample_rate: int,
    hop_length: int = 160,
    f0_min: float = 50.0,
    f0_max: float = 1100.0
) -> np.ndarray:
    """
    Extract F0 (pitch) using pyin method.
    
    Args:
        audio: Input audio array
        sample_rate: Sample rate
        hop_length: Hop length for analysis
        f0_min: Minimum F0 frequency
        f0_max: Maximum F0 frequency
    
    Returns:
        F0 contour array
    """
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=f0_min,
        fmax=f0_max,
        sr=sample_rate,
        hop_length=hop_length
    )
    
    # Replace NaN with 0
    f0 = np.nan_to_num(f0, nan=0.0)
    return f0.astype(np.float32)


def extract_f0_harvest(
    audio: np.ndarray,
    sample_rate: int,
    hop_length: int = 160,
    f0_min: float = 50.0,
    f0_max: float = 1100.0
) -> np.ndarray:
    """
    Extract F0 using harvest-like method (simplified).
    Uses librosa's piptrack as fallback.
    
    Args:
        audio: Input audio array
        sample_rate: Sample rate
        hop_length: Hop length for analysis
        f0_min: Minimum F0 frequency
        f0_max: Maximum F0 frequency
    
    Returns:
        F0 contour array
    """
    # Use piptrack as harvest alternative
    pitches, magnitudes = librosa.piptrack(
        y=audio,
        sr=sample_rate,
        hop_length=hop_length,
        fmin=f0_min,
        fmax=f0_max
    )
    
    # Select pitch with highest magnitude for each frame
    f0 = []
    for i in range(pitches.shape[1]):
        index = magnitudes[:, i].argmax()
        pitch = pitches[index, i]
        f0.append(pitch if pitch > 0 else 0.0)
    
    return np.array(f0, dtype=np.float32)


def extract_f0(
    audio: np.ndarray,
    sample_rate: int,
    method: str = "harvest",
    hop_length: int = 160,
    f0_min: float = 50.0,
    f0_max: float = 1100.0
) -> np.ndarray:
    """
    Extract F0 (pitch) using specified method.
    
    Args:
        audio: Input audio array
        sample_rate: Sample rate
        method: Extraction method ('harvest' or 'pm')
        hop_length: Hop length for analysis
        f0_min: Minimum F0 frequency
        f0_max: Maximum F0 frequency
    
    Returns:
        F0 contour array
    """
    if method == "pm":
        return extract_f0_pm(audio, sample_rate, hop_length, f0_min, f0_max)
    else:
        # Default to harvest-like method
        return extract_f0_harvest(audio, sample_rate, hop_length, f0_min, f0_max)


def shift_f0(f0: np.ndarray, semitones: int) -> np.ndarray:
    """
    Shift F0 by given number of semitones.
    
    Args:
        f0: F0 contour array
        semitones: Number of semitones to shift (positive = higher)
    
    Returns:
        Shifted F0 contour
    """
    if semitones == 0:
        return f0
    
    # Calculate frequency ratio for semitone shift
    ratio = 2 ** (semitones / 12)
    
    # Only shift non-zero (voiced) frames
    shifted = f0.copy()
    voiced_mask = f0 > 0
    shifted[voiced_mask] = f0[voiced_mask] * ratio
    
    return shifted.astype(np.float32)


def get_audio_duration(audio: np.ndarray, sample_rate: int) -> float:
    """
    Get duration of audio in seconds.
    
    Args:
        audio: Audio array
        sample_rate: Sample rate
    
    Returns:
        Duration in seconds
    """
    return len(audio) / sample_rate
