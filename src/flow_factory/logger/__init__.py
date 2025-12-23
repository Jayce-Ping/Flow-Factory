# src/flow_factory/logger/__init__.py
"""
Logging Module

Provides logging backends for experiment tracking with a registry-based
loading system for easy extensibility.

Supported backends:
- WandB (Weights & Biases)
- SwanLab
- Custom backends via registry
"""

from .abc import Logger, LogImage, LogVideo
from .registry import (
    register_logger,
    get_logger_class,
    list_registered_loggers,
)
from .loader import load_logger

# Pre-registered loggers (lazy-loaded)
# from .wandb import WandbLogger
# from .swanlab import SwanlabLogger

__all__ = [
    # Core classes
    "Logger",
    "LogImage",
    "LogVideo",
    
    # Registry functions
    "register_logger",
    "get_logger_class",
    "list_registered_loggers",
    
    # Factory function
    "load_logger",
]