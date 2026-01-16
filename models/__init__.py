"""
RVC Models Package.
Contains all neural network components for RVC voice conversion.
"""

from models.rvc_model import RVCModel, ModelManager, model_manager
from models.hubert_model import HuBERTFeatureExtractor, load_hubert
from models.synthesizer_v1 import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
)
from models.synthesizer_v2 import (
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from models.generator import Generator, GeneratorNSF
from models.flow import ResidualCouplingBlock
from models.text_encoder import TextEncoder256, TextEncoder768

__all__ = [
    # Main model classes
    "RVCModel",
    "ModelManager",
    "model_manager",
    # Feature extraction
    "HuBERTFeatureExtractor",
    "load_hubert",
    # Synthesizers v1
    "SynthesizerTrnMs256NSFsid",
    "SynthesizerTrnMs256NSFsid_nono",
    # Synthesizers v2
    "SynthesizerTrnMs768NSFsid",
    "SynthesizerTrnMs768NSFsid_nono",
    # Generators
    "Generator",
    "GeneratorNSF",
    # Other components
    "ResidualCouplingBlock",
    "TextEncoder256",
    "TextEncoder768",
]
