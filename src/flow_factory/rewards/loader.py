# src/flow_factory/rewards/loader.py
"""
Reward Model Loader
Factory function using registry pattern for extensibility.
"""
from typing import Optional
from accelerate import Accelerator

from .abc import BaseRewardModel
from .registry import get_reward_model_class, list_registered_reward_models
from ..hparams import RewardArguments


def load_reward_model(
    reward_args: RewardArguments,
    accelerator: Accelerator,
) -> BaseRewardModel:
    """
    Load and initialize the appropriate reward model based on configuration.
    
    Uses registry pattern for automatic reward model discovery and loading.
    Supports both built-in models and custom backends via python paths.
    
    Args:
        reward_args: reward model configuration arguments
        accelerator: Accelerator instance for distributed setup
    
    Returns:
        Reward model instance
    
    Raises:
        ImportError: If the reward model is not registered or cannot be imported
    
    Examples:
        # Using built-in reward model
        reward_args.reward_model = "PickScore"
        reward_model = load_reward_model(reward_args, accelerator)
        
        # Using custom reward model
        reward_args.reward_model = "my_package.rewards.ImageReward"
        reward_model = load_reward_model(reward_args, accelerator)
    """
    reward_model_identifier = reward_args.reward_model
    
    try:
        # Get reward model class from registry or direct import
        reward_model_class = get_reward_model_class(reward_model_identifier)
        
        # Instantiate reward model
        reward_model = reward_model_class(reward_args=reward_args, accelerator=accelerator)
        
        return reward_model
        
    except ImportError as e:
        registered_models = list(list_registered_reward_models().keys())
        raise ImportError(
            f"Failed to load reward model '{reward_model_identifier}'. "
            f"Available models: {registered_models}"
        ) from e