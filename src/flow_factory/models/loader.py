# src/flow_factory/model/loader.py
import logging
from typing import Tuple
from .adapter import BaseAdapter
from .flux import FluxAdapter
from .sd3 import StableDiffusionAdapter
from ..hparams.training_args import TrainingArguments
from ..hparams.model_args import ModelArguments

logger = logging.getLogger(__name__)

def load_model(model_args : ModelArguments, training_args : TrainingArguments) -> BaseAdapter:
    """
    Factory function to instantiate the correct model adapter based on configuration.
    
    Args:
        model_args: DataClass containing 'model_type', 'model_name_or_path', etc.
        training_args: DataClass containing bf16/fp16 settings.
    
    Returns:
        An instance of a subclass of BaseAdapter.
    """
    model_type = model_args.model_type.lower()
    
    logger.info(f"Loading model architecture: {model_type}...")
    
    if model_type == "flux":
        return FluxAdapter(model_args)
        
    elif model_type in ["sd3", "sdxl", "stable_diffusion"]:
        return StableDiffusionAdapter(model_args)
        
    else:
        raise ValueError(f"Unknown model type: {model_type}. Supported: ['flux', 'sd3', 'sd']")