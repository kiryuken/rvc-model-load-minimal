"""
Generator module wrapper for RVC.
Re-exports generators from nsf.py for backwards compatibility.
"""

from models.nsf import (
    Generator,
    GeneratorNSF,
    ResBlock1,
    ResBlock2,
    SourceModuleHnNSF,
)

__all__ = [
    "Generator",
    "GeneratorNSF",
    "ResBlock1",
    "ResBlock2",
    "SourceModuleHnNSF",
]
