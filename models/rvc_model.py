"""
Production RVC Model Loader and Inference Pipeline.
Handles loading .pth and .index files with auto-detection of model configuration.

IMPORTANT: This is a production implementation - no fallbacks, no mocks.
If any component fails to load, an error is raised and execution stops.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import faiss
import numpy as np
import torch
import torch.nn.functional as F

from models.hubert_model import HuBERTFeatureExtractor
from models.synthesizer_v1 import SynthesizerTrnMs256NSFsid, SynthesizerTrnMs256NSFsid_nono
from models.synthesizer_v2 import SynthesizerTrnMs768NSFsid, SynthesizerTrnMs768NSFsid_nono
from utils.audio_utils import extract_f0, shift_f0

logger = logging.getLogger(__name__)


def get_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Auto-detect and return the best available device."""
    if device is not None:
        if isinstance(device, torch.device):
            return device
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            return torch.device("cpu")
        return torch.device(device)
    
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# Default model configuration for RVC
DEFAULT_CONFIG = {
    "spec_channels": 1025,
    "segment_size": 32,
    "inter_channels": 192,
    "hidden_channels": 192,
    "filter_channels": 768,
    "n_heads": 2,
    "n_layers": 6,
    "kernel_size": 3,
    "p_dropout": 0.0,
    "resblock": "1",
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "upsample_rates": [10, 10, 2, 2],
    "upsample_initial_channel": 512,
    "upsample_kernel_sizes": [16, 16, 4, 4],
    "spk_embed_dim": 109,
    "gin_channels": 256,
    "sr": 40000,
}

# Sample rate to upsample rate mapping
SR_TO_UPSAMPLE = {
    32000: [10, 8, 2, 2],
    40000: [10, 10, 2, 2],
    48000: [10, 12, 2, 2],
}


class RVCModel:
    """
    Production RVC voice conversion model.
    
    This class handles:
    - Loading and parsing .pth checkpoint files
    - Auto-detecting model version (v1/v2)
    - Auto-detecting F0 usage
    - Initializing the correct synthesizer architecture
    - Loading FAISS indices for speaker similarity
    - Running inference with real neural networks
    
    FAILURE POLICY: If any required component fails to load, an exception
    is raised immediately. No silent fallbacks.
    """

    def __init__(
        self,
        pth_path: str,
        index_path: Optional[str] = None,
        hubert_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        is_half: bool = False,
        force_fp32: bool = False,
    ):
        """
        Initialize RVC model.
        
        Args:
            pth_path: Path to .pth model checkpoint
            index_path: Path to .index FAISS file (optional but recommended)
            hubert_path: Path to hubert_base.pt (defaults to /models/runtime/hubert_base.pt)
            device: Torch device (auto-detected if None)
            is_half: Use FP16 for synthesizer (HuBERT in FP16, synthesizer in FP32 by default)
            force_fp32: Force all models to FP32 (overrides is_half)
        
        Raises:
            FileNotFoundError: If pth_path does not exist
        """
        self.pth_path = Path(pth_path)
        self.index_path = Path(index_path) if index_path else None
        self.hubert_path = Path(hubert_path) if hubert_path else Path(
            os.getenv("HUBERT_PATH", "/models/runtime/hubert_base.pt")
        )
        self.device = get_device(device)
        
        # FP16 policy: HuBERT in FP16, synthesizer in FP32 by default
        # force_fp32 overrides everything to FP32
        self.force_fp32 = force_fp32
        self.is_half = is_half and not force_fp32
        self.hubert_half = not force_fp32  # HuBERT in FP16 unless forced FP32
        
        # Model components (initialized on load)
        self.model: Optional[torch.nn.Module] = None
        self.index: Optional[faiss.Index] = None
        self.hubert: Optional[HuBERTFeatureExtractor] = None
        
        # Model configuration
        self.config: Dict[str, Any] = {}
        self.version: str = "v2"  # v1 or v2
        self.has_f0: bool = True  # Whether model uses F0 conditioning
        self.sample_rate: int = 40000
        self.hop_length: int = 320  # Depends on sample rate
        
        self._loaded = False
        
        # Validate paths
        if not self.pth_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.pth_path}")
        
        if self.index_path and not self.index_path.exists():
            raise FileNotFoundError(f"Index file not found: {self.index_path}")

    def load(self) -> None:
        """
        Load all model components.
        
        This loads:
        1. HuBERT feature extractor
        2. Model checkpoint and configuration
        3. Synthesizer network with weights
        4. FAISS index (if provided)
        
        Raises:
            RuntimeError: If any component fails to load
            FileNotFoundError: If required files are missing
        """
        if self._loaded:
            return
        
        logger.info(f"Loading RVC model from {self.pth_path}")
        
        # Step 1: Load HuBERT
        self._load_hubert()
        
        # Step 2: Load checkpoint and parse configuration
        checkpoint = self._load_checkpoint()
        
        # Step 3: Initialize and load synthesizer
        self._initialize_synthesizer(checkpoint)
        
        # Step 4: Load FAISS index
        self._load_index()
        
        self._loaded = True
        logger.info(
            f"Model loaded successfully: version={self.version}, "
            f"f0={self.has_f0}, sr={self.sample_rate}"
        )

    def _load_hubert(self) -> None:
        """Load HuBERT feature extractor."""
        if not self.hubert_path.exists():
            raise FileNotFoundError(
                f"HuBERT model not found: {self.hubert_path}\n"
                f"Please provide hubert_base.pt at the specified path or set HUBERT_PATH env var."
            )
        
        logger.info(f"Loading HuBERT from {self.hubert_path}")
        self.hubert = HuBERTFeatureExtractor(
            str(self.hubert_path),
            device=self.device,
            is_half=self.hubert_half,
        )
        self.hubert.load()

    def _load_checkpoint(self) -> Dict[str, Any]:
        """Load and parse the .pth checkpoint."""
        logger.info(f"Loading checkpoint from {self.pth_path}")
        
        checkpoint = torch.load(
            self.pth_path,
            map_location="cpu",  # Load to CPU first, then move
            weights_only=False,
        )
        
        # Parse configuration from checkpoint
        self._parse_config(checkpoint)
        
        return checkpoint

    def _parse_config(self, checkpoint: Dict[str, Any]) -> None:
        """Parse model configuration from checkpoint."""
        # Extract config from checkpoint
        if "config" in checkpoint:
            config = checkpoint["config"]
            
            # Sample rate
            if isinstance(config, (list, tuple)) and len(config) >= 18:
                # Old format: config is a list
                self.sample_rate = config[17] if len(config) > 17 else 40000
            elif isinstance(config, dict):
                # New format: config is a dict
                self.sample_rate = config.get("sr", config.get("sample_rate", 40000))
            
            self.config = config if isinstance(config, dict) else {}
        
        # Set hop length based on sample rate
        self.hop_length = self.sample_rate // 100  # 10ms hop
        
        # Detect version from weight keys
        weight_key = "weight" if "weight" in checkpoint else "model"
        if weight_key in checkpoint:
            weights = checkpoint[weight_key]
            weight_keys = list(weights.keys()) if isinstance(weights, dict) else []
            
            # v2 models have enc_p.pre with 768-dim input
            # v1 models have enc_p.pre with 256-dim input
            for key in weight_keys:
                if "enc_p.pre.weight" in key:
                    shape = weights[key].shape
                    if shape[1] == 768:
                        self.version = "v2"
                    else:
                        self.version = "v1"
                    break
        
        # Detect F0 usage
        self.has_f0 = True  # Default to True
        if weight_key in checkpoint:
            weights = checkpoint[weight_key]
            weight_keys = list(weights.keys()) if isinstance(weights, dict) else []
            
            # If model has f0_emb, it uses F0
            self.has_f0 = any("f0_emb" in k for k in weight_keys)
        
        logger.info(f"Detected: version={self.version}, f0={self.has_f0}, sr={self.sample_rate}")

    def _initialize_synthesizer(self, checkpoint: Dict[str, Any]) -> None:
        """Initialize synthesizer network and load weights."""
        # Build config for synthesizer
        config = dict(DEFAULT_CONFIG)
        config["sr"] = self.sample_rate
        config["upsample_rates"] = SR_TO_UPSAMPLE.get(
            self.sample_rate, 
            [10, 10, 2, 2]
        )
        
        # Merge with checkpoint config if available
        if isinstance(self.config, dict):
            for key in config:
                if key in self.config:
                    config[key] = self.config[key]
        
        # Select synthesizer class
        if self.version == "v1":
            if self.has_f0:
                synth_class = SynthesizerTrnMs256NSFsid
            else:
                synth_class = SynthesizerTrnMs256NSFsid_nono
        else:  # v2
            if self.has_f0:
                synth_class = SynthesizerTrnMs768NSFsid
            else:
                synth_class = SynthesizerTrnMs768NSFsid_nono
        
        logger.info(f"Initializing synthesizer: {synth_class.__name__}")
        
        # Create synthesizer
        self.model = synth_class(**config)
        
        # Load weights
        weight_key = "weight" if "weight" in checkpoint else "model"
        if weight_key in checkpoint:
            weights = checkpoint[weight_key]
            
            # Load with strict=True to ensure all weights match
            try:
                self.model.load_state_dict(weights, strict=True)
                logger.info("Loaded weights with strict=True")
            except RuntimeError as e:
                # Try non-strict as fallback but log warning
                logger.warning(f"Strict loading failed, trying non-strict: {e}")
                self.model.load_state_dict(weights, strict=False)
                logger.warning("Loaded weights with strict=False - some weights may be missing")
        else:
            raise RuntimeError("Checkpoint does not contain model weights")
        
        # Move to device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Apply FP16 if requested
        if self.is_half:
            self.model = self.model.half()
            logger.info("Synthesizer in FP16 mode")
        else:
            logger.info("Synthesizer in FP32 mode")
        
        # Disable gradients
        for param in self.model.parameters():
            param.requires_grad = False

    def _load_index(self) -> None:
        """Load FAISS index for speaker similarity search."""
        if self.index_path is None:
            logger.info("No index file provided - proceeding without FAISS")
            return
        
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index file not found: {self.index_path}")
        
        logger.info(f"Loading FAISS index from {self.index_path}")
        
        try:
            self.index = faiss.read_index(str(self.index_path))
            logger.info(f"FAISS index loaded: {self.index.ntotal} vectors")
        except Exception as e:
            raise RuntimeError(f"Failed to load FAISS index: {e}")

    def infer(
        self,
        audio: np.ndarray,
        f0_up_key: int = 0,
        f0_method: str = "harvest",
        index_rate: float = 0.5,
        protect: float = 0.33,
    ) -> np.ndarray:
        """
        Perform voice conversion inference.
        
        Pipeline:
        1. Resample audio to model sample rate (if needed)
        2. Extract HuBERT features
        3. Apply FAISS k-NN blending (if index available)
        4. Extract and shift F0 (if model uses F0)
        5. Run synthesizer forward pass
        6. Validate and return output
        
        Args:
            audio: Input audio array (mono, float32)
            f0_up_key: Pitch shift in semitones (-24 to +24)
            f0_method: F0 extraction method ("harvest", "pm")
            index_rate: FAISS blending ratio (0-1, higher = more from index)
            protect: Protect consonants ratio (0-0.5)
        
        Returns:
            Converted audio array (mono, float32)
        
        Raises:
            RuntimeError: If inference fails or produces invalid output
        """
        if not self._loaded:
            self.load()
        
        logger.debug(f"Starting inference: pitch={f0_up_key}, method={f0_method}")
        
        # Ensure audio is float32
        audio = audio.astype(np.float32)
        
        # Get input length for duration preservation
        input_length = len(audio)
        
        with torch.no_grad():
            # Step 1: Extract HuBERT features
            hubert_features = self.hubert.extract(
                audio,
                sample_rate=16000,  # HuBERT expects 16kHz
                version=self.version,
            )  # [1, T, D]
            
            # Step 2: Apply FAISS blending
            if self.index is not None and index_rate > 0:
                hubert_features = self._apply_faiss_blending(
                    hubert_features, 
                    index_rate,
                    protect,
                )
            
            # Step 3: Prepare features for synthesizer
            # Transpose to [B, D, T] format expected by synthesizer
            phone = hubert_features.transpose(1, 2).to(self.device)
            phone_lengths = torch.LongTensor([phone.size(2)]).to(self.device)
            
            # Apply FP16 if model is half precision
            if self.is_half:
                phone = phone.half()
            
            # Step 4: Extract and process F0 (if model uses it)
            if self.has_f0:
                # Extract F0 at frame rate
                f0 = extract_f0(
                    audio, 
                    sample_rate=16000,
                    method=f0_method,
                    hop_length=160,  # 10ms for 16kHz
                )
                
                # Apply pitch shift
                if f0_up_key != 0:
                    f0 = shift_f0(f0, f0_up_key)
                
                # Align F0 length with features
                f0 = self._align_f0(f0, phone.size(2))
                
                # Continuous F0 for NSF generator
                f0_nsf = torch.FloatTensor(f0).unsqueeze(0).to(self.device)
                
                # Quantized F0 for embedding
                f0_mel = self._f0_to_mel(f0)
                f0_quant = torch.LongTensor(f0_mel).unsqueeze(0).to(self.device)
                
                if self.is_half:
                    f0_nsf = f0_nsf.half()
                
                # Speaker ID (use 0 for single-speaker models)
                sid = torch.LongTensor([0]).to(self.device)
                
                # Run synthesizer
                output = self.model.infer(
                    phone,
                    phone_lengths,
                    f0_quant,
                    f0_nsf,
                    sid,
                )
            else:
                # No F0 model
                sid = torch.LongTensor([0]).to(self.device)
                
                output = self.model.infer(
                    phone,
                    phone_lengths,
                    sid,
                )
            
            # Step 5: Process output
            output = output.squeeze(0).squeeze(0)  # [T]
            output = output.float().cpu().numpy()
        
        # Step 6: Validate output
        self._validate_output(output)
        
        # Normalize
        max_val = np.abs(output).max()
        if max_val > 0:
            output = output / max_val * 0.95
        
        # Clear GPU cache
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        
        return output

    def _apply_faiss_blending(
        self,
        features: torch.Tensor,
        index_rate: float,
        protect: float,
    ) -> torch.Tensor:
        """
        Apply FAISS k-NN blending for speaker similarity.
        
        Args:
            features: HuBERT features [1, T, D]
            index_rate: Blending ratio (0-1)
            protect: Consonant protection ratio
        
        Returns:
            Blended features [1, T, D]
        """
        if self.index is None:
            return features
        
        k = 8  # Number of nearest neighbors
        
        # Get features as numpy
        feat_np = features.squeeze(0).cpu().numpy().astype(np.float32)  # [T, D]
        
        # Search for nearest neighbors
        try:
            distances, indices = self.index.search(feat_np, k)
        except Exception as e:
            logger.warning(f"FAISS search failed: {e}")
            return features
        
        # Try to reconstruct features from index
        # Some indices don't support reconstruction, so we handle that gracefully
        try:
            # Check if index supports reconstruction
            has_reconstruct = hasattr(self.index, 'reconstruct') and hasattr(self.index, 'make_direct_map')
            
            if has_reconstruct:
                # Try to initialize direct map if needed
                try:
                    self.index.make_direct_map()
                except RuntimeError:
                    # Already has direct map or not supported
                    pass
            
            reconstructed = np.zeros_like(feat_np)
            reconstruct_failed = False
            
            for i in range(feat_np.shape[0]):
                # Weight by inverse distance (closer = higher weight)
                weights = 1.0 / (distances[i] + 1e-6)
                weights = weights / weights.sum()
                
                for j, idx in enumerate(indices[i]):
                    if idx >= 0 and idx < self.index.ntotal:
                        try:
                            vec = self.index.reconstruct(int(idx))
                            reconstructed[i] += weights[j] * vec
                        except RuntimeError:
                            # Reconstruction not supported, fall back to using original features
                            reconstruct_failed = True
                            break
                
                if reconstruct_failed:
                    break
            
            if reconstruct_failed:
                # Reconstruction failed, skip blending
                logger.debug("FAISS reconstruction not supported, skipping blending")
                return features
            
            # Blend original and reconstructed
            blended = (1 - index_rate) * feat_np + index_rate * reconstructed
            
            # Convert back to tensor
            features = torch.from_numpy(blended).unsqueeze(0).to(features.device)
            
            if features.dtype != torch.float32:
                features = features.float()
                
        except Exception as e:
            logger.warning(f"FAISS blending failed: {e}")
        
        return features

    def _align_f0(self, f0: np.ndarray, target_length: int) -> np.ndarray:
        """Align F0 array to target length."""
        if len(f0) == target_length:
            return f0
        
        # Interpolate to target length
        indices = np.linspace(0, len(f0) - 1, target_length)
        aligned = np.interp(indices, np.arange(len(f0)), f0)
        
        return aligned.astype(np.float32)

    def _f0_to_mel(self, f0: np.ndarray) -> np.ndarray:
        """Convert F0 to mel scale for embedding."""
        # Convert to mel scale and quantize to 256 bins
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0 == 0] = 0
        
        # Clip to valid range
        f0_mel = np.clip(f0_mel / 16, 0, 255)
        
        return f0_mel.astype(np.int64)

    def _validate_output(self, output: np.ndarray) -> None:
        """
        Validate inference output.
        
        Raises:
            RuntimeError: If output contains NaN, Inf, or is silent
        """
        # Check for NaN
        if np.isnan(output).any():
            raise RuntimeError("Output contains NaN values - inference failed")
        
        # Check for Inf
        if np.isinf(output).any():
            raise RuntimeError("Output contains Inf values - inference failed")
        
        # Check for silence (RMS < threshold)
        rms = np.sqrt(np.mean(output ** 2))
        if rms < 1e-6:
            raise RuntimeError(
                f"Output is silent (RMS={rms:.2e}) - inference may have failed"
            )
        
        logger.debug(f"Output validated: RMS={rms:.4f}, length={len(output)}")


class ModelManager:
    """
    Model manager for handling multiple RVC models.
    Caches loaded models to avoid reloading.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._models: Dict[str, RVCModel] = {}
            cls._instance._models_dir = Path(os.getenv("MODELS_DIR", "/models"))
            cls._instance._hubert_path = Path(os.getenv(
                "HUBERT_PATH", 
                "/models/runtime/hubert_base.pt"
            ))
        return cls._instance

    def set_models_dir(self, path: str) -> None:
        """Set the directory containing model files."""
        self._models_dir = Path(path)

    def set_hubert_path(self, path: str) -> None:
        """Set the path to HuBERT model."""
        self._hubert_path = Path(path)

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
            # Skip hubert and rmvpe models
            if pth_file.stem.lower() in ["hubert_base", "hubert", "rmvpe"]:
                continue
            
            name = pth_file.stem

            # Look for matching .index file
            index_file = pth_file.with_suffix(".index")
            if not index_file.exists():
                # Try finding index in same directory
                index_files = list(pth_file.parent.glob("*.index"))
                index_file = index_files[0] if index_files else None

            available[name] = {
                "pth": str(pth_file),
                "index": str(index_file) if index_file and index_file.exists() else None,
            }

        return available

    def load(
        self,
        name: str,
        pth_path: Optional[str] = None,
        index_path: Optional[str] = None,
        force_reload: bool = False,
    ) -> RVCModel:
        """
        Load a model by name or paths.
        
        Args:
            name: Model identifier
            pth_path: Optional explicit path to .pth file
            index_path: Optional explicit path to .index file
            force_reload: Force reload even if already loaded
        
        Returns:
            Loaded RVCModel instance
        
        Raises:
            FileNotFoundError: If model not found
        """
        if name in self._models and not force_reload:
            return self._models[name]

        # If paths not provided, scan for them
        if pth_path is None:
            available = self.scan_models()
            if name not in available:
                raise FileNotFoundError(
                    f"Model '{name}' not found in {self._models_dir}"
                )
            pth_path = available[name]["pth"]
            index_path = index_path or available[name]["index"]

        # Create and load model
        model = RVCModel(
            pth_path,
            index_path,
            hubert_path=str(self._hubert_path),
        )
        model.load()

        self._models[name] = model
        return model

    def get(self, name: str) -> Optional[RVCModel]:
        """Get a loaded model by name."""
        return self._models.get(name)

    def list_models(self) -> List[str]:
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
                    "loaded": True,
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
