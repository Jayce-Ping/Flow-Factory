
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional
from logging import getLogger
import torch.distributed as dist
logger = getLogger(__name__)

def get_world_size() -> int:
    if dist.is_initialized():
        return dist.get_world_size()
    return 1

@dataclass
class ModelArguments:
    r"""Arguments pertaining to model configuration."""

    model_name_or_path: str = field(
        default="black-forest-labs/FLUX.1-dev",
        metadata={"help": "Path to pre-trained model or model identifier from huggingface.co/models"},
    )

    model_type: Literal["sd3", "flux1", "flux1-kontext", 'flux2', 'qwenimage', 'qwenimage-edit'] = field(
        default="flux1",
        metadata={"help": "Type of model to use."},
    )

    lora_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to LoRA weights."},
    )

    save_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save the model checkpoints."},
    )

    def __post_init__(self):
        pass

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)