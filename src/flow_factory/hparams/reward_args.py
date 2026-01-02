# src/flow_factory/hparams/reward_args.py
import os
import math
import yaml
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional, Type, Union
import logging
import torch

from .abc import ArgABC


dtype_map = {
    'fp16': torch.float16,
    'bf16': torch.bfloat16,    
    'fp32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
    'float32': torch.float32,
}

@dataclass
class RewardArguments(ArgABC):
    r"""Arguments pertaining to reward configuration."""

    reward_model : Optional[str] = field(
        default=None,
        metadata={"help": "The path or name of the reward model to use. You can specify 'PickScore' to use the registered PickScore model. Or /path/to/your/model:class_name to use your own reward model."},
    )

    dtype: Union[Literal['float16', 'bfloat16', 'float32'], torch.dtype] = field(
        default='bfloat16',
        metadata={"help": "The data type for the reward model."},
        repr=False,
    )

    device: Union[Literal['cpu', 'cuda'], torch.device] = field(
        default='cuda',
        metadata={"help": "The device to load the reward model on."},
        repr=False,
    )

    batch_size: int = field(
        default=16,
        metadata={"help": "Batch size for reward model inference."},
    )

    def __post_init__(self):

        if isinstance(self.dtype, str):
            self.dtype = dtype_map[self.dtype]

        if isinstance(self.device, str):
            self.device = torch.device(self.device)        

    def to_dict(self) -> dict[str, Any]:
        # Use super().to_dict() here as well
        d = super().to_dict()
        d['dtype'] = str(self.dtype).split('.')[-1]
        d['device'] = str(self.device)
        return d

    def __str__(self) -> str:
        """Pretty print configuration as YAML."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False, indent=2)
    
    def __repr__(self) -> str:
        """Same as __str__ for consistency."""
        return self.__str__()

    def __eq__(self, other):
        """
        Compare RewardArguments instances considering all fields including extra_kwargs.
        Handles torch.dtype and torch.device comparison properly.
        """
        if not isinstance(other, RewardArguments):
            return False
        
        # Compare core fields
        core_equal = (
            self.reward_model == other.reward_model and
            self.dtype == other.dtype and
            self.device == other.device and
            self.batch_size == other.batch_size
        )
        
        if not core_equal:
            return False
        
        # Compare extra_kwargs
        return self.extra_kwargs == other.extra_kwargs