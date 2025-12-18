# src/flow_factory/models/reward_model.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List, Union
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from PIL import Image
from ..hparams.training_args import TrainingArguments
from ..hparams.reward_args import RewardArguments

from diffusers.utils.outputs import BaseOutput
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

@dataclass
class RewardModelOutput(BaseOutput):
    """
    Output class for Reward models.
    """
    rewards : Union[torch.FloatTensor, np.ndarray, List[float]]
    extra_info : Optional[Dict[str, Any]] = None

class BaseRewardModel(ABC, nn.Module):
    """
    Abstract Base Class for reward models.
    """
    def __init__(
            self,
            reward_args : RewardArguments,
        ):
        self.reward_args = reward_args
        self.device = reward_args.torch_device
        self.dtype = reward_args.torch_dtype

    @abstractmethod
    def __call__(self, **inputs) -> RewardModelOutput:
        """Compute reward given inputs."""
        pass