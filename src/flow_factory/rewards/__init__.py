# src/flow_factory/rewards/__init__.py
"""
Reward models module for evaluating generated content.
"""
from .reward_model import BaseRewardModel, RewardModelOutput
from .registry import register_reward_model, get_reward_model_class, list_registered_reward_models
from .loader import load_reward_model

# Auto-import registered reward models
from .pick_score import PickScoreRewardModel

__all__ = [
    'BaseRewardModel',
    'RewardModelOutput',
    'PickScoreRewardModel',
    'register_reward_model',
    'get_reward_model_class',
    'list_registered_reward_models',
    'load_reward_model',
]