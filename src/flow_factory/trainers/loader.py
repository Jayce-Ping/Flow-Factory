# src/flow_factory/trainers/loader.py
"""
Trainer loader factory for extensibility.
Supports multiple RL algorithms and can be easily extended.
"""
from typing import Literal
from logging import getLogger

from .trainer import BaseTrainer
from .grpo_trainer import GRPOTrainer
from ..hparams.data_args import DataArguments
from ..hparams.training_args import TrainingArguments
from ..hparams.reward_args import RewardArguments
from ..models.adapter import BaseAdapter

logger = getLogger(__name__)


def load_trainer(
    trainer_type: Literal["grpo"] = "grpo",
    data_args: DataArguments = None,
    training_args: TrainingArguments = None,
    reward_args: RewardArguments = None,
    adapter: BaseAdapter = None,
) -> BaseTrainer:
    """
    Factory function to instantiate the correct trainer based on algorithm type.
    
    Args:
        trainer_type: Algorithm type (grpo, ppo, dpo, reinforce, etc.)
        data_args: Data configuration
        training_args: Training configuration
        reward_args: Reward model configuration
        adapter: Model adapter instance
    
    Returns:
        An instance of a subclass of BaseTrainer
    """
    trainer_type = trainer_type.lower()
    
    logger.info(f"Loading trainer: {trainer_type}...")
    
    trainer_mapping = {
        "grpo": GRPOTrainer,
        # Future extensions:
    }
    
    if trainer_type not in trainer_mapping:
        raise ValueError(
            f"Unknown trainer type: {trainer_type}. "
            f"Supported: {list(trainer_mapping.keys())}"
        )
    
    trainer_cls = trainer_mapping[trainer_type]
    
    return trainer_cls(
        data_args=data_args,
        training_args=training_args,
        reward_args=reward_args,
        adapter=adapter,
    )