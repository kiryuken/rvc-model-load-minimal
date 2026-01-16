"""
RVC Model handler for voice conversion inference.
Supports loading .pth and .index files, with auto-detection of model format.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import faiss
import numpy as np
import torch
import torch.nn.functional as F

from utils.audio_utils import extract_f0, shift_f0

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Auto-detect and return the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class HubertFeatureExtractor:
    """
    Simplified feature extractor for RVC.
    In production, this would load a HuBERT model for content extraction.
    """
    
    def __init__(self, device: torch.device):
        self.device = device
        self.model = None
        self._loaded = False
    
    def load(self) -> None:
        """Load the feature extraction model."""
        # Note: In a full implementation, this would load HuBERT
        # For now, we provide a placeholder that returns mel-like features
        self._loaded = True
        logger.info("Feature extractor initialized (simplified mode)")
    
    def extract(self, audio: np.ndarray, sample_rate: int = 16000) -> torch.Tensor:
        """
        Extract features from audio.
        
        Args:
            audio: Input audio array
            sample_rate: Sample rate
        
        Returns:
            Feature tensor
        """
        if not self._loaded:
            self.load()
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).float().to(self.device)
        
        # Simplified feature extraction using mel spectrogram
        # In production, this would be HuBERT features
        n_fft = 1024
        hop_length = 160
        n_mels = 256
        
        # Compute mel spectrogram as feature proxy
        window = torch.hann_window(n_fft).to(self.device)
        stft = torch.stft(
            audio_tensor,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            return_complex=True
        )
        magnitude = torch.abs(stft)
        
        # Simple mel filterbank
        features = magnitude[:n_mels, :]
        
        return features.unsqueeze(0)  # Add batch dimension


class RVCModel:
    """
    RVC voice conversion model handler.
    Handles model loading and inference for voice conversion.
    """
    
    def __init__(
        self,
        pth_path: str,
        index_path: Optional[str] = None,
        device: Optional[torch.device] = None
    ):
        self.pth_path = Path(pth_path)
        self.index_path = Path(index_path) if index_path else None
        self.device = device or get_device()
        
        self.model = None
        self.index = None
        self.config = {}
        self.version = "v2"  # Default version
        self.sample_rate = 16000
        self.hop_length = 160
        
        self._loaded = False
        self.feature_extractor = HubertFeatureExtractor(self.device)
    
    def load(self) -> None:
        """Load the model from .pth and optionally .index files."""
        if self._loaded:
            return
        
        logger.info(f"Loading RVC model from {self.pth_path}")
        
        # Load PyTorch checkpoint
        if not self.pth_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.pth_path}")
        
        checkpoint = torch.load(
            self.pth_path,
            map_location=self.device,
            weights_only=False
        )
        
        # Extract model configuration
        self._parse_checkpoint(checkpoint)
        
        # Load FAISS index if provided
        if self.index_path and self.index_path.exists():
            logger.info(f"Loading FAISS index from {self.index_path}")
            self.index = faiss.read_index(str(self.index_path))
        
        # Initialize feature extractor
        self.feature_extractor.load()
        
        self._loaded = True
        logger.info(f"Model loaded successfully (version: {self.version}, sr: {self.sample_rate})")
    
    def _parse_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Parse checkpoint and extract model weights and config."""
        # Handle different checkpoint formats
        if "config" in checkpoint:
            config = checkpoint["config"]
            self.sample_rate = config.get("sample_rate", 16000)
            self.hop_length = config.get("hop_length", 160)
            self.version = config.get("model_type", "v2")
        
        # Try to detect version from weight keys
        if "weight" in checkpoint:
            weight_keys = list(checkpoint["weight"].keys()) if isinstance(checkpoint["weight"], dict) else []
            if any("emb_g" in k for k in weight_keys):
                self.version = "v2"
            else:
                self.version = "v1"
        
        # Store the model weights
        if "weight" in checkpoint:
            self.model = checkpoint["weight"]
        elif "model" in checkpoint:
            self.model = checkpoint["model"]
        else:
            # Assume the checkpoint itself is the state dict
            self.model = checkpoint
        
        self.config = checkpoint.get("config", {})
    
    def infer(
        self,
        audio: np.ndarray,
        f0_up_key: int = 0,
        f0_method: str = "harvest",
        index_rate: float = 0.5
    ) -> np.ndarray:
        """
        Perform voice conversion inference.
        
        Args:
            audio: Input audio array (mono, float32)
            f0_up_key: Pitch shift in semitones
            f0_method: F0 extraction method ('harvest' or 'pm')
            index_rate: How much to use the index (0-1)
        
        Returns:
            Converted audio array
        """
        if not self._loaded:
            self.load()
        
        logger.info(f"Starting inference (pitch: {f0_up_key}, method: {f0_method})")
        
        # Ensure audio is float32
        audio = audio.astype(np.float32)
        
        # Extract F0
        f0 = extract_f0(audio, self.sample_rate, method=f0_method, hop_length=self.hop_length)
        
        # Apply pitch shift
        if f0_up_key != 0:
            f0 = shift_f0(f0, f0_up_key)
        
        # Extract content features
        features = self.feature_extractor.extract(audio, self.sample_rate)
        
        # Apply index-based speaker embedding if available
        if self.index is not None and index_rate > 0:
            features = self._apply_index(features, index_rate)
        
        # Run voice conversion
        with torch.no_grad():
            output = self._convert(features, f0)
        
        # Clear CUDA cache if using GPU
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        
        return output
    
    def _apply_index(self, features: torch.Tensor, index_rate: float) -> torch.Tensor:
        """Apply FAISS index for speaker similarity matching."""
        if self.index is None:
            return features
        
        # Flatten features for FAISS search
        feat_np = features.squeeze(0).cpu().numpy().T  # (time, features)
        
        # Search for nearest neighbors
        k = 8  # Number of neighbors
        try:
            distances, indices = self.index.search(feat_np.astype(np.float32), k)
            
            # Reconstruct features from index
            reconstructed = np.zeros_like(feat_np)
            for i in range(feat_np.shape[0]):
                weights = 1.0 / (distances[i] + 1e-6)
                weights /= weights.sum()
                for j, idx in enumerate(indices[i]):
                    if idx >= 0:
                        vec = self.index.reconstruct(int(idx))
                        reconstructed[i] += weights[j] * vec
            
            # Blend original and reconstructed
            blended = (1 - index_rate) * feat_np + index_rate * reconstructed
            features = torch.from_numpy(blended.T).unsqueeze(0).to(features.device)
        except Exception as e:
            logger.warning(f"Index application failed: {e}")
        
        return features
    
    def _convert(self, features: torch.Tensor, f0: np.ndarray) -> np.ndarray:
        """
        Run the voice conversion network.
        
        This is a simplified implementation. In production, this would
        use the actual RVC generator network from the loaded weights.
        """
        # Convert F0 to tensor
        f0_tensor = torch.from_numpy(f0).float().to(self.device)
        
        # Simplified conversion: use features and F0 to generate output
        # In production, this calls the actual generator model
        
        # For now, we'll use a simple vocoder-like approach
        # This generates audio from the features
        
        output_length = len(f0) * self.hop_length
        output = torch.zeros(output_length, device=self.device)
        
        # Generate output using F0-guided synthesis
        t = torch.arange(output_length, device=self.device).float()
        
        for i, freq in enumerate(f0_tensor):
            if freq > 0:
                start_idx = i * self.hop_length
                end_idx = min((i + 1) * self.hop_length, output_length)
                
                # Generate sinusoidal component at F0
                phase = 2 * np.pi * freq * t[start_idx:end_idx] / self.sample_rate
                output[start_idx:end_idx] += 0.3 * torch.sin(phase)
        
        # Apply feature-based modulation (simplified)
        if features.shape[-1] > 0:
            feat_envelope = F.interpolate(
                features.mean(dim=1, keepdim=True),
                size=output_length,
                mode='linear',
                align_corners=False
            ).squeeze()
            
            # Normalize envelope
            feat_envelope = feat_envelope / (feat_envelope.abs().max() + 1e-6)
            output = output * (0.5 + 0.5 * feat_envelope)
        
        # Convert to numpy
        output_np = output.cpu().numpy().astype(np.float32)
        
        # Normalize output
        max_val = np.abs(output_np).max()
        if max_val > 0:
            output_np = output_np / max_val * 0.95
        
        return output_np


class ModelManager:
    """
    Singleton manager for handling multiple RVC models.
    Caches loaded models to avoid reloading.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._models: Dict[str, RVCModel] = {}
            cls._instance._models_dir = Path(os.getenv("MODELS_DIR", "/models"))
        return cls._instance
    
    def set_models_dir(self, path: str) -> None:
        """Set the directory containing model files."""
        self._models_dir = Path(path)
    
    def scan_models(self) -> Dict[str, Dict[str, Optional[str]]]:
        """
        Scan models directory for available models.
        
        Returns:
            Dictionary of model names to their .pth and .index paths
        """
        available = {}
        
        if not self._models_dir.exists():
            logger.warning(f"Models directory not found: {self._models_dir}")
            return available
        
        # Look for .pth files
        for pth_file in self._models_dir.glob("**/*.pth"):
            name = pth_file.stem
            
            # Look for matching .index file
            index_file = pth_file.with_suffix(".index")
            if not index_file.exists():
                # Try finding index in same directory
                index_files = list(pth_file.parent.glob("*.index"))
                index_file = index_files[0] if index_files else None
            
            available[name] = {
                "pth": str(pth_file),
                "index": str(index_file) if index_file and index_file.exists() else None
            }
        
        return available
    
    def load(
        self,
        name: str,
        pth_path: Optional[str] = None,
        index_path: Optional[str] = None
    ) -> RVCModel:
        """
        Load a model by name or paths.
        
        Args:
            name: Model identifier
            pth_path: Optional explicit path to .pth file
            index_path: Optional explicit path to .index file
        
        Returns:
            Loaded RVCModel instance
        """
        if name in self._models:
            return self._models[name]
        
        # If paths not provided, scan for them
        if pth_path is None:
            available = self.scan_models()
            if name not in available:
                raise FileNotFoundError(f"Model '{name}' not found in {self._models_dir}")
            pth_path = available[name]["pth"]
            index_path = index_path or available[name]["index"]
        
        # Create and load model
        model = RVCModel(pth_path, index_path)
        model.load()
        
        self._models[name] = model
        return model
    
    def get(self, name: str) -> Optional[RVCModel]:
        """Get a loaded model by name."""
        return self._models.get(name)
    
    def list_models(self) -> list:
        """List all loaded model names."""
        return list(self._models.keys())
    
    def list_available(self) -> Dict[str, Dict[str, Optional[str]]]:
        """List all available models (loaded and unloaded)."""
        available = self.scan_models()
        for name in self._models:
            if name not in available:
                model = self._models[name]
                available[name] = {
                    "pth": str(model.pth_path),
                    "index": str(model.index_path) if model.index_path else None,
                    "loaded": True
                }
            else:
                available[name]["loaded"] = name in self._models
        return available
    
    def unload(self, name: str) -> bool:
        """Unload a model to free memory."""
        if name in self._models:
            del self._models[name]
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"Model '{name}' unloaded")
            return True
        return False
    
    def unload_all(self) -> None:
        """Unload all models."""
        self._models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("All models unloaded")


# Global model manager instance
model_manager = ModelManager()
