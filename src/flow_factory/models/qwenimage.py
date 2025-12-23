# src/flow_factory/models/qwen_image.py
from __future__ import annotations

import os
from typing import Union, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import torch
from diffusers.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline
from PIL import Image
import logging

from .adapter import BaseAdapter, BaseSample
from .registry import register_model_adapter
from ..hparams import *
from ..scheduler.flow_matching import FlowMatchEulerDiscreteSDEScheduler, FlowMatchEulerDiscreteSDESchedulerOutput, set_scheduler_timesteps
from ..utils.base import filter_kwargs

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class QwenImageSample(BaseSample):
    """Output class for Qwen-Image models."""
    prompt_embeds_mask : Optional[torch.FloatTensor] = None
    negative_prompt_embeds_mask : Optional[torch.FloatTensor] = None

@register_model_adapter('qwenimage')
class QwenImageAdapter(BaseAdapter):
    """Adapter for Qwen-Image text-to-image models."""
    
    def __init__(self, config: Arguments):
        super().__init__(config)
    
    def load_pipeline(self) -> QwenImagePipeline:
        return QwenImagePipeline.from_pretrained(
            self.model_args.model_name_or_path,
            low_cpu_mem_usage=False
        )
    
    @property
    def default_target_modules(self) -> List[str]:
        """Default LoRA target modules for Qwen-Image transformer."""
        return [
            "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
            "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "attn.to_add_out",
            "img_mlp.net.0.proj", "img_mlp.net.2.proj",
            "txt_mlp.net.0.proj", "txt_mlp.net.2.proj"
        ]