# src/flow_factory/models/bagel/pipeline.py
"""
Bagel Pseudo-Pipeline

Lightweight wrapper that mimics the diffusers DiffusionPipeline interface,
allowing BaseAdapter's component management (get_component, set_component,
freeze, LoRA, offload) to work unchanged.

Bagel differs from diffusers pipelines in that:
  - Text encoding is internal to the Bagel model (no separate text_encoder)
  - The VAE is a custom autoencoder (not diffusers' AutoencoderKL)
  - Image understanding uses a ViT (SiglipVisionModel) inside the Bagel model
  - Context is built via KV-cache, not via separate encoder embeddings

Component Mapping:
  pipeline.transformer  →  Bagel model (LLM + generation heads)
  pipeline.vae          →  Custom autoencoder (encode/decode images)
"""
from __future__ import annotations

import os
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn


class BagelPseudoPipeline:
    """
    Pseudo-pipeline holding Bagel components under diffusers-compatible names.

    This is NOT a real DiffusionPipeline; it's a thin namespace that the
    BaseAdapter can query via ``getattr(self.pipeline, name)``.
    """

    def __init__(
        self,
        transformer: nn.Module,
        vae: nn.Module,
        scheduler: Optional[Any] = None,
        config: Optional[Any] = None,
    ):
        self.transformer = transformer
        self.vae = vae
        self.scheduler = scheduler

        # Store the original BagelConfig for reference
        self._bagel_config = config or getattr(transformer, "config", None)

    # ---- DiffusionPipeline-like interface stubs ----
    def maybe_free_model_hooks(self):
        """No-op: Bagel doesn't use diffusers model hooks."""
        pass

    @property
    def device(self) -> torch.device:
        """Infer device from transformer parameters."""
        try:
            return next(self.transformer.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        """Infer dtype from transformer parameters."""
        try:
            return next(self.transformer.parameters()).dtype
        except StopIteration:
            return torch.bfloat16

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        vae_path: Optional[str] = None,
        low_cpu_mem_usage: bool = False,
        **kwargs,
    ) -> "BagelPseudoPipeline":
        """
        Construct Bagel components from a pretrained checkpoint directory.

        Expected directory layout (BAGEL-7B-MoT style):
            model_path/
            ├── llm_config.json
            ├── vit_config.json
            ├── ae.safetensors        # VAE weights
            ├── ema.safetensors       # Bagel model weights
            ├── tokenizer files ...
            └── ...

        Args:
            model_path: Path to the Bagel model checkpoint directory.
            vae_path: Optional separate path for VAE weights.
                      Defaults to ``model_path/ae.safetensors``.
            low_cpu_mem_usage: If True, use ``init_empty_weights`` to defer
                               weight materialization (for multi-GPU dispatch).
        """
        from modeling.bagel import (
            BagelConfig, Bagel,
            Qwen2Config, Qwen2ForCausalLM,
            SiglipVisionConfig, SiglipVisionModel,
        )
        from modeling.autoencoder import load_ae
        from safetensors.torch import load_file

        # ---- LLM Config ----
        llm_config = Qwen2Config.from_json_file(
            os.path.join(model_path, "llm_config.json")
        )
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = kwargs.get("layer_module", "Qwen2MoTDecoderLayer")

        # ---- ViT Config ----
        vit_config = SiglipVisionConfig.from_json_file(
            os.path.join(model_path, "vit_config.json")
        )
        vit_config.rope = kwargs.get("vit_rope", False)
        vit_config.num_hidden_layers = (
            vit_config.num_hidden_layers - 1
        )  # Default for inference

        # ---- VAE ----
        ae_path = vae_path or os.path.join(model_path, "ae.safetensors")
        vae_model, vae_config = load_ae(local_path=ae_path)

        # ---- Bagel Config ----
        config = BagelConfig(
            visual_gen=True,
            visual_und=True,
            llm_config=llm_config,
            vit_config=vit_config,
            vae_config=vae_config,
            vit_max_num_patch_per_side=kwargs.get("vit_max_num_patch_per_side", 70),
            connector_act=kwargs.get("connector_act", "gelu_pytorch_tanh"),
            latent_patch_size=kwargs.get("latent_patch_size", 2),
            max_latent_size=kwargs.get("max_latent_size", 64),
        )

        # ---- Build Models ----
        if low_cpu_mem_usage:
            from accelerate import init_empty_weights

            with init_empty_weights():
                language_model = Qwen2ForCausalLM(llm_config)
                vit_model = SiglipVisionModel(vit_config)
                model = Bagel(language_model, vit_model, config)
                model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(
                    vit_config, meta=True
                )
        else:
            language_model = Qwen2ForCausalLM(llm_config)
            vit_model = SiglipVisionModel(vit_config)
            model = Bagel(language_model, vit_model, config)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

            # Load weights
            ema_path = os.path.join(model_path, "ema.safetensors")
            if os.path.exists(ema_path):
                state_dict = load_file(ema_path)
                model.load_state_dict(state_dict, strict=False)

        return cls(
            transformer=model,
            vae=vae_model,
            config=config,
        )