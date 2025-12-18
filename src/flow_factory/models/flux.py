# src/flow_factory/model/flux_adapter.py
from typing import Union, List, Dict, Any, Optional
from dataclasses import dataclass
import torch
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from torch.cuda import is_available
from .adapter import BaseAdapter, BaseSample

@dataclass
class FluxSample(BaseSample):
    """
    Output class for Flux Adapter models.
    """
    # Inherits all fields from BaseSample

class FluxAdapter(BaseAdapter):
    """
    Concrete implementation for Flow Matching models (e.g., FLUX.1).
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 1. Load Core Pipeline
        self.pipeline = FluxPipeline.from_pretrained(
            self.model_args.model_name_or_path,
            torch_dtype=torch.bfloat16 if self.training_args.mixed_precision == "bf16" else torch.float16,
        )
        
        self.pipeline.vae.requires_grad_(False)
        self.pipeline.text_encoder.requires_grad_(False)
        self.pipeline.text_encoder_2.requires_grad_(False)
        
        self.pipeline.scheduler = self.scheduler

    def off_load_text_encoder(self):
        """Off-load text encoder to CPU to save GPU memory."""
        self.pipeline.text_encoder.to("cpu")
        self.pipeline.text_encoder_2.to("cpu")

    def off_load_vae(self):
        """Off-load VAE to CPU to save GPU memory."""
        self.pipeline.vae.to("cpu")

    def off_load_transformer(self):
        """Off-load transformer to CPU to save GPU memory."""
        self.pipeline.transformer.to("cpu")

    def on_load_text_encoder(self, device: Union[torch.device, str] = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        """Load text encoder to specified device."""
        self.pipeline.text_encoder.to(device)
        self.pipeline.text_encoder_2.to(device)

    def on_load_vae(self, device: Union[torch.device, str] = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        """Load VAE to specified device."""
        self.pipeline.vae.to(device)

    def on_load_transformer(self, device: Union[torch.device, str] = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        """Load transformer to specified device."""
        self.pipeline.transformer.to(device)


    def encode_prompts(self, prompts: Union[str, List[str]], **kwargs) -> Dict[str, Any]:
        """Encode text prompts using the pipeline's text encoder."""
        prompt_embeds, pooled_prompt_embeds, text_ids = self.pipeline.encode_prompt(
            prompt=prompts,
            device=self.device,
        )
        prompt_ids = self.pipeline.tokenizer_2(
            prompts,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)
        return {
            'prompt_ids': prompt_ids,
            'prompt_embeds': prompt_embeds,
            'pooled_prompt_embeds': pooled_prompt_embeds,
        }
    
    def encode_images(self, images: Union[torch.Tensor, List[torch.Tensor]], **kwargs) -> torch.Tensor:
        # No need to encode images for FLUX models
        pass
    