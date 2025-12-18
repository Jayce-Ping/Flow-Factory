from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional
import yaml
from .data_args import DataArguments
from .model_args import ModelArguments
from .training_args import TrainingArguments
from .reward_args import RewardArguments

@dataclass
class Arguments:
    r"""Main arguments class encapsulating all configurations."""
    data_args: DataArguments = field(
        default_factory=DataArguments,
        metadata={"help": "Arguments for data configuration."},
    )
    model_args: ModelArguments = field(
        default_factory=ModelArguments,
        metadata={"help": "Arguments for model configuration."},
    )
    training_args: TrainingArguments = field(
        default_factory=TrainingArguments,
        metadata={"help": "Arguments for training configuration."},
    )
    reward_args: RewardArguments = field(
        default_factory=RewardArguments,
        metadata={"help": "Arguments for reward model configuration."},
    )
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
    

def from_dict(cls, args_dict: dict[str, Any]) -> 'Arguments':
    """Create Arguments instance from dictionary."""
    data_args = DataArguments(**args_dict.get('data', {}))
    model_args = ModelArguments(**args_dict.get('model', {}))
    training_args = TrainingArguments(**args_dict.get('train', {}))
    reward_args = RewardArguments(**args_dict.get('reward', {}))
    return cls(
        data_args=data_args,
        model_args=model_args,
        training_args=training_args,
        reward_args=reward_args,
    )

def load_from_yaml(cls, yaml_file: str) -> 'Arguments':
    """Load Arguments from a YAML file."""
    with open(yaml_file, 'r') as f:
        args_dict = yaml.safe_load(f)
    return from_dict(cls, args_dict)