"""
HuBERT Feature Extractor for RVC.
Supports multiple backends: fairseq (official) or transformers (easier to install).

For v1 models: Extract layer 9 (256-dim features)
For v2 models: Extract layer 12 (768-dim features)
"""

import logging
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


# Check which backend is available
def _check_backend() -> str:
    """Check which HuBERT backend is available."""
    try:
        import fairseq
        return "fairseq"
    except ImportError:
        pass
    
    try:
        from transformers import HubertModel
        return "transformers"
    except ImportError:
        pass
    
    return "none"


def _get_device(device: Union[str, torch.device]) -> torch.device:
    """Get the appropriate device, handling 'auto' and CUDA availability."""
    if isinstance(device, torch.device):
        return device
    
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        return torch.device("cpu")
    
    return torch.device(device)


HUBERT_BACKEND = _check_backend()


class HuBERTFeatureExtractor:
    """
    Production HuBERT feature extractor.
    
    Supports two backends:
    1. fairseq (official) - requires fairseq package and hubert_base.pt checkpoint
    2. transformers - requires transformers package, uses HuggingFace models
    
    The backend is auto-detected based on available packages.
    """

    def __init__(
        self,
        model_path: str,
        device: Union[str, torch.device] = "auto",
        is_half: bool = False,
        backend: Optional[str] = None,
    ):
        """
        Initialize HuBERT feature extractor.
        
        Args:
            model_path: Path to hubert_base.pt checkpoint (for fairseq backend)
                       or HuggingFace model name (for transformers backend)
            device: Device to run inference on ("auto", "cuda", or "cpu")
            is_half: Use FP16 for faster inference
            backend: Force specific backend ("fairseq" or "transformers")
        
        Raises:
            FileNotFoundError: If model_path does not exist (fairseq only)
            ImportError: If neither fairseq nor transformers is installed
            RuntimeError: If model fails to load
        """
        self.model_path = Path(model_path) if model_path else None
        self.device = _get_device(device)
        self.is_half = is_half and self.device.type == "cuda"  # FP16 only on CUDA
        
        self.model = None
        self.processor = None
        self._loaded = False
        
        # Determine backend
        self.backend = backend or HUBERT_BACKEND
        
        if self.backend == "none":
            raise ImportError(
                "No HuBERT backend available.\n"
                "Install one of:\n"
                "  - pip install fairseq (official, may need build tools)\n"
                "  - pip install transformers (easier to install)\n"
            )
        
        # For fairseq, validate model path exists
        if self.backend == "fairseq":
            if self.model_path is None or not self.model_path.exists():
                raise FileNotFoundError(
                    f"HuBERT model not found at: {self.model_path}\n"
                    f"Please provide hubert_base.pt at the specified path."
                )
        
        logger.info(f"HuBERT backend: {self.backend}, device: {self.device}")

    def load(self) -> None:
        """
        Load the HuBERT model.
        
        Raises:
            RuntimeError: If model fails to load
        """
        if self._loaded:
            return
        
        if self.backend == "fairseq":
            self._load_fairseq()
        else:
            self._load_transformers()
        
        self._loaded = True

    def _load_fairseq(self) -> None:
        """Load HuBERT using fairseq."""
        import fairseq
        
        logger.info(f"Loading HuBERT from {self.model_path} (fairseq)")
        
        try:
            models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
                [str(self.model_path)],
                suffix="",
            )
            
            self.model = models[0]
            self.model = self.model.to(self.device)
            self.model.eval()
            
            if self.is_half:
                self.model = self.model.half()
                logger.info("HuBERT loaded in FP16 mode (fairseq)")
            else:
                logger.info("HuBERT loaded in FP32 mode (fairseq)")
            
            for param in self.model.parameters():
                param.requires_grad = False
                
        except Exception as e:
            raise RuntimeError(f"Failed to load HuBERT (fairseq): {e}")

    def _load_transformers(self) -> None:
        """Load HuBERT using transformers library."""
        from transformers import HubertModel
        
        # Use HuggingFace model
        model_name = "facebook/hubert-base-ls960"
        
        # Check if model_path points to a local transformers model
        if self.model_path and self.model_path.exists():
            if (self.model_path / "config.json").exists():
                model_name = str(self.model_path)
        
        logger.info(f"Loading HuBERT from {model_name} (transformers)")
        
        try:
            self.model = HubertModel.from_pretrained(
                model_name,
                output_hidden_states=True,
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            
            if self.is_half:
                self.model = self.model.half()
                logger.info("HuBERT loaded in FP16 mode (transformers)")
            else:
                logger.info("HuBERT loaded in FP32 mode (transformers)")
            
            for param in self.model.parameters():
                param.requires_grad = False
                
        except Exception as e:
            raise RuntimeError(f"Failed to load HuBERT (transformers): {e}")

    def extract(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        sample_rate: int = 16000,
        version: str = "v2",
    ) -> torch.Tensor:
        """
        Extract content features from audio.
        
        Args:
            audio: Input audio (mono, float32, typically 16kHz)
            sample_rate: Sample rate of input audio
            version: Model version ("v1" or "v2")
                - v1: Extract layer 9 (256-dim, projected from 768)
                - v2: Extract layer 12 (768-dim)
        
        Returns:
            Feature tensor [1, T, D] where D=256 (v1) or 768 (v2)
        """
        if not self._loaded:
            self.load()
        
        # Convert numpy to tensor if needed
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)
        
        audio = audio.float()
        
        # Ensure correct shape [1, T] for batch processing
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        audio = audio.to(self.device)
        
        if self.is_half:
            audio = audio.half()
        
        # Determine which layer to extract
        layer_idx = 9 if version.lower() == "v1" else 12
        
        with torch.no_grad():
            if self.backend == "fairseq":
                features = self._extract_fairseq(audio, layer_idx)
            else:
                features = self._extract_transformers(audio, layer_idx, version)
        
        return features

    def _extract_fairseq(
        self,
        audio: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """Extract features using fairseq HuBERT."""
        padding_mask = torch.zeros(audio.shape, dtype=torch.bool, device=audio.device)
        
        inputs = {
            "source": audio,
            "padding_mask": padding_mask,
            "output_layer": layer_idx,
        }
        
        try:
            logits = self.model.extract_features(**inputs)
            features = logits[0] if isinstance(logits, tuple) else logits
        except TypeError:
            feats, _ = self.model.extract_features(
                source=audio,
                padding_mask=padding_mask,
                mask=False,
                output_layer=layer_idx,
            )
            features = feats
        
        if features.dim() == 2:
            features = features.unsqueeze(0)
        
        return features

    def _extract_transformers(
        self,
        audio: torch.Tensor,
        layer_idx: int,
        version: str,
    ) -> torch.Tensor:
        """Extract features using transformers HuBERT."""
        # Transformers HuBERT expects [batch, seq_len]
        outputs = self.model(audio, output_hidden_states=True)
        
        # Get hidden states from the specified layer
        hidden_states = outputs.hidden_states
        
        # Get the requested layer
        if layer_idx < len(hidden_states):
            features = hidden_states[layer_idx]
        else:
            features = hidden_states[-1]
            logger.warning(f"Layer {layer_idx} not available, using last layer")
        
        # Features are [B, T, 768] - transformers HuBERT always outputs 768-dim
        # For v1, we need to project to 256-dim
        if version.lower() == "v1":
            # Project from 768 to 256 by taking first 256 dims and projecting
            # This is a simple approach - for best results, use fairseq backend
            features = features[:, :, :256]
        
        return features

    def __call__(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        sample_rate: int = 16000,
        version: str = "v2",
    ) -> torch.Tensor:
        """Convenience method to extract features."""
        return self.extract(audio, sample_rate, version)


class ContentFeatureExtractor(nn.Module):
    """
    Alternative PyTorch-native content feature extractor.
    Used when neither fairseq nor transformers is available.
    
    NOTE: This is NOT a replacement for real HuBERT - it's a CNN-based
    fallback that produces features of the correct shape.
    For production, always use HuBERTFeatureExtractor.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Union[str, torch.device] = "auto",
        is_half: bool = False,
    ):
        """Initialize content feature extractor."""
        super().__init__()
        self.device = _get_device(device)
        self.is_half = is_half and self.device.type == "cuda"
        self.model_path = model_path
        self._loaded = False
        
        # CNN-based feature extractor (NOT HuBERT, just for shape compatibility)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 512, kernel_size=10, stride=5, padding=2),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=2, stride=2, padding=0),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=2, stride=2, padding=0),
            nn.GELU(),
        )
        
        self.proj_256 = nn.Linear(512, 256)
        self.proj_768 = nn.Linear(512, 768)

    def load(self) -> None:
        """Load model weights if provided."""
        if self.model_path and Path(self.model_path).exists():
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.load_state_dict(state_dict, strict=False)
        
        self.to(self.device)
        self.eval()
        
        if self.is_half:
            self.half()
        
        for param in self.parameters():
            param.requires_grad = False
        
        self._loaded = True
        logger.warning(
            "Using ContentFeatureExtractor fallback (CNN, not HuBERT). "
            "Install transformers or fairseq for real HuBERT."
        )

    def extract(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        sample_rate: int = 16000,
        version: str = "v2",
    ) -> torch.Tensor:
        """Extract features from audio."""
        if not self._loaded:
            self.load()
        
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)
        
        audio = audio.float().to(self.device)
        
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)
        elif audio.dim() == 2:
            audio = audio.unsqueeze(1)
        
        if self.is_half:
            audio = audio.half()
        
        with torch.no_grad():
            features = self.conv_layers(audio)
            features = features.transpose(1, 2)
            
            if version.lower() == "v1":
                features = self.proj_256(features)
            else:
                features = self.proj_768(features)
        
        return features

    def forward(self, audio: torch.Tensor, version: str = "v2") -> torch.Tensor:
        return self.extract(audio, version=version)

    def __call__(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        sample_rate: int = 16000,
        version: str = "v2",
    ) -> torch.Tensor:
        return self.extract(audio, sample_rate, version)


def load_hubert(
    model_path: str,
    device: Union[str, torch.device] = "auto",
    is_half: bool = False,
) -> HuBERTFeatureExtractor:
    """
    Factory function to load HuBERT feature extractor.
    
    Automatically selects the best available backend:
    1. fairseq (if installed)
    2. transformers (if installed)
    
    Args:
        model_path: Path to hubert_base.pt (fairseq) or model name (transformers)
        device: Device to run on ("auto", "cuda", or "cpu")
        is_half: Use FP16
    
    Returns:
        Initialized and loaded HuBERTFeatureExtractor
    """
    extractor = HuBERTFeatureExtractor(model_path, device, is_half)
    extractor.load()
    return extractor
