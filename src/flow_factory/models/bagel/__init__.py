# src/flow_factory/models/bagel/__init__.py
"""
Bagel Model Adapter

Integrates ByteDance's Bagel multimodal model into Flow-Factory.
Supports Text-to-Image and Image(s)-to-Image generation tasks.
"""

from .bagel import BagelAdapter, BagelSample, BagelI2ISample
from .pipeline import BagelPseudoPipeline

__all__ = [
    "BagelAdapter",
    "BagelSample",
    "BagelI2ISample",
    "BagelPseudoPipeline",
]