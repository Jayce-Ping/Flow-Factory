# src/flow_factory/models/trainer.py
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataclasses import dataclass
from PIL import Image
from diffusers.utils.outputs import BaseOutput
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration

from ..hparams.training_args import TrainingArguments
from ..hparams.data_args import DataArguments
from ..hparams.reward_args import RewardArguments
from ..models.adapter import BaseAdapter
from ..data_utils.loader import get_dataloader
from ..rewards.reward_model import BaseRewardModel


class BaseTrainer(ABC):
    """
    Abstract Base Class for Flow-Factory trainers.
    """
    def __init__(
            self,
            data_args : DataArguments,
            training_args : TrainingArguments,
            reward_args : RewardArguments,
            adapter : BaseAdapter,
        ):
        self.data_args = data_args
        self.training_args = training_args
        self.reward_args = reward_args
        self.adapter = adapter

        self._init_accelerator()
        # Offload text-encoder to save memory
        self.adapter.off_load_text_encoder()
    
    @abstractmethod
    def _init_reward_model(self) -> BaseRewardModel:
        """Initialize reward model."""
        self.reward_model : BaseRewardModel
        pass

    def _init_dataloader(self) -> Tuple[DataLoader, Union[None, DataLoader]]:
        dataloader, test_dataloader = get_dataloader(
            data_args=self.data_args,
            training_args=self.training_args,
            text_encode_func=self.adapter.encode_prompts,
            image_encode_func=self.adapter.encode_images,
        )
        return dataloader, test_dataloader
    
    def _init_optimizer(self) -> torch.optim.Optimizer:
        """Initialize optimizer."""
        transformer_trainable_parameters = self.adapter.get_trainable_parameters()
        self.optimizer = torch.optim.AdamW(
            transformer_trainable_parameters,
            lr=self.training_args.learning_rate,
            betas=(self.training_args.adam_beta1, self.training_args.adam_beta2),
            weight_decay=self.training_args.adam_weight_decay,
            eps=self.training_args.adam_epsilon,
        )
        return self.optimizer

    def _init_accelerator(self):
        accelerator_config = ProjectConfiguration(
            project_dir=os.path.join(self.training_args.save_dir, self.training_args.run_name),
            automatic_checkpoint_naming=True,
        )

        self.accelerator = Accelerator(
            mixed_precision=self.training_args.mixed_precision,
            project_config=accelerator_config,
            gradient_accumulation_steps=self.training_args.gradient_accumulation_steps * self.training_args.num_timesteps,
        )
        set_seed(self.training_args.seed)

        # Init dataloader and optimizer
        self.dataloader, self.test_dataloader = self._init_dataloader()
        self.optimizer = self._init_optimizer()
        # Prepare everything with accelerator
        # Here, `self.dataloader` is not prepared since it has been handled with DistributedKRepeatSampler
        if self.test_dataloader is not None:
            self.adapter, self.optimizer, self.test_dataloader = self.accelerator.prepare(
                self.adapter,
                self.optimizer,
                self.test_dataloader,
            )
        else:
            self.adapter, self.optimizer = self.accelerator.prepare(
                self.adapter,
                self.optimizer,
            )

        # Initialize reward model
        self._init_reward_model()

    @abstractmethod
    def compute_loss(self):
        """Main training loop."""
        pass

    @abstractmethod
    def evaluate(self):
        """Evaluation loop."""
        pass

    def save_checkpoint(self, path: str):
        """Save trainer state to a specific path."""
        self.adapter.save_checkpoint(path)

    def load_checkpoint(self, path: str):
        """Load trainer state from a specific path."""
        self.adapter.load_checkpoint(path)