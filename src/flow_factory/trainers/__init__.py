# src/flow_factory/trainers/__init__.py
"""
Trainers module for various RL algorithms.
"""
from .trainer import BaseTrainer
from .registry import register_trainer, get_trainer_class, list_registered_trainers
from .loader import load_trainer

# Auto-import registered trainers
from .grpo_trainer import GRPOTrainer

__all__ = [
    'BaseTrainer',
    'GRPOTrainer',
    'register_trainer',
    'get_trainer_class',
    'list_registered_trainers',
    'load_trainer',
]