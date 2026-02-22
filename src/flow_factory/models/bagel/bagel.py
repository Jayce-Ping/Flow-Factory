"""
Bagel Model Adapter for Flow-Factory (Refactored)

Key Design Changes vs. Original:
    1. ``_forward_flow`` lives here (not in pipeline) so ``self.transformer``
       resolves to the accelerator-wrapped module for distributed training.
    2. Context KV-caches are rebuilt at training time via ``_rebuild_context``
       using the current (possibly updated) transformer parameters.
    3. Sampling stores lightweight "context specs" (prompts, image tensors)
       instead of KV-cache tensors, enabling correct re-materialization.

Architecture::

    BagelAdapter
      └── self.pipeline: BagelPseudoPipeline
            ├── .transformer        Qwen2ForCausalLM  (unwrapped)
            ├── .vit / .vae / ...
            └── prepare_*  (stateless utilities only)

    self.transformer → accelerator-wrapped transformer (for gradients)
"""
from __future__ import annotations

import os
import random
from contextlib import contextmanager
from copy import deepcopy
from typing import Union, List, Dict, Any, Optional, Tuple, Literal, ClassVar
from dataclasses import dataclass, field
from collections import defaultdict
import math
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image

from accelerate import Accelerator

from ...samples import T2ISample, I2ISample
from ..abc import BaseAdapter
from ...hparams import Arguments
from ...scheduler import (
    FlowMatchEulerDiscreteSDEScheduler,
    SDESchedulerOutput,
)
from ...utils.base import filter_kwargs
from ...utils.trajectory_collector import (
    TrajectoryCollector,
    CallbackCollector,
    TrajectoryIndicesType,
    create_trajectory_collector,
    create_callback_collector,
)
from ...utils.image import (
    ImageSingle,
    ImageBatch,
    MultiImageBatch,
    is_image,
    is_image_batch,
    is_multi_image_batch,
    standardize_image_batch,
    pil_image_to_tensor,
)
from ...utils.logger_utils import setup_logger

from .pipeline import BagelPseudoPipeline
from .modeling.bagel import NaiveCache, Qwen2ForCausalLM

logger = setup_logger(__name__)

CONDITION_IMAGE_SIZE = (1024, 1024)


# ============================================================================
# Context Spec: Lightweight description of how to rebuild KV-cache
# ============================================================================

@dataclass
class ContextSpec:
    """
    Stores the *recipe* for rebuilding a KV-cache context, not the cache itself.

    This allows training to re-materialize context with updated transformer
    parameters instead of reusing stale cached KV states.
    
    Fields:
        steps: Ordered list of (type, data) tuples describing context construction.
            type is one of 'text', 'vit_image', 'vae_image'.
            data is the raw input (string for text, tensor for images).
        kv_lens: Final KV lengths after all steps (needed for latent preparation).
        ropes: Final rope positions after all steps.
    """
    steps: List[Tuple[str, Any]] = field(default_factory=list)
    kv_lens: List[int] = field(default_factory=lambda: [0])
    ropes: List[int] = field(default_factory=lambda: [0])


# ============================================================================
# Sample Dataclasses
# ============================================================================

@dataclass
class BagelSample(T2ISample):
    """Sample for Bagel T2I with context specs for KV-cache re-materialization."""

    _shared_fields: ClassVar[frozenset[str]] = frozenset({"image_shape"})

    # Context specs (lightweight, for rebuilding KV-cache during training)
    gen_context_spec: Optional[ContextSpec] = None
    cfg_text_context_spec: Optional[ContextSpec] = None
    cfg_img_context_spec: Optional[ContextSpec] = None

    # Generation input tensors (stateless, no transformer dependency)
    generation_input: Optional[Dict[str, torch.Tensor]] = None
    cfg_text_generation_input: Optional[Dict[str, torch.Tensor]] = None
    cfg_img_generation_input: Optional[Dict[str, torch.Tensor]] = None
    image_shape: Optional[Tuple[int, int]] = None


@dataclass
class BagelI2ISample(I2ISample):
    """Sample for Bagel I2I with context specs for KV-cache re-materialization."""

    _shared_fields: ClassVar[frozenset[str]] = frozenset({"image_shape"})

    gen_context_spec: Optional[ContextSpec] = None
    cfg_text_context_spec: Optional[ContextSpec] = None
    cfg_img_context_spec: Optional[ContextSpec] = None

    generation_input: Optional[Dict[str, torch.Tensor]] = None
    cfg_text_generation_input: Optional[Dict[str, torch.Tensor]] = None
    cfg_img_generation_input: Optional[Dict[str, torch.Tensor]] = None
    image_shape: Optional[Tuple[int, int]] = None


def calculate_dimensions(target_area, ratio):
    """Calculate (width, height) from target pixel area and aspect ratio (h/w)."""
    height = math.sqrt(target_area * ratio)
    width = height / ratio
    width = round(width / 32) * 32
    height = round(height / 32) * 32
    return width, height


# ============================================================================
# BagelAdapter
# ============================================================================

class BagelAdapter(BaseAdapter):
    """
    Flow-Factory adapter for Bagel multimodal models.

    Key design: ``_forward_flow`` is defined here so that
    ``self.transformer`` resolves to the accelerator-wrapped module,
    ensuring proper gradient flow in distributed training (FSDP/DDP).
    """

    def __init__(self, config: Arguments, accelerator: Accelerator):
        self._model_path = config.model_args.model_name_or_path
        self._init_tokenizer_and_transforms()
        super().__init__(config, accelerator)
        self.pipeline: BagelPseudoPipeline
        self.scheduler: FlowMatchEulerDiscreteSDEScheduler

    # ─────────────────── Tokenizer & Transforms ───────────────────

    def _init_tokenizer_and_transforms(self):
        """Initialize tokenizer, special tokens, and image transforms."""
        from .modeling.qwen2 import Qwen2Tokenizer
        from .data.data_utils import add_special_tokens
        from .data.transforms import ImageTransform

        self._tokenizer = Qwen2Tokenizer.from_pretrained(self._model_path)
        self._tokenizer, self.new_token_ids, _ = add_special_tokens(self._tokenizer)

        self.vae_transform = ImageTransform(1024, 512, 16)
        self.vit_transform = ImageTransform(980, 224, 14)

    # ======================== Pipeline & Scheduler ========================

    def load_pipeline(self) -> BagelPseudoPipeline:
        return BagelPseudoPipeline.from_pretrained(
            self._model_path, low_cpu_mem_usage=False,
            **self.model_args.extra_kwargs,
        )

    def load_scheduler(self) -> FlowMatchEulerDiscreteSDEScheduler:
        scheduler_kwargs = {"num_train_timesteps": 1000}
        if hasattr(self.config, "scheduler_args") and self.config.scheduler_args:
            scheduler_kwargs.update(self.config.scheduler_args.to_dict())
        return FlowMatchEulerDiscreteSDEScheduler(**scheduler_kwargs)

    # ======================== Module Management ========================

    @property
    def default_target_modules(self) -> List[str]:
        return [
            "self_attn.q_proj_moe_gen", "self_attn.k_proj_moe_gen",
            "self_attn.v_proj_moe_gen", "self_attn.o_proj_moe_gen",
            "mlp_moe_gen.gate_proj", "mlp_moe_gen.up_proj", "mlp_moe_gen.down_proj",
        ]

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @property
    def text_encoder_names(self) -> List[str]:
        return []

    @property
    def text_encoders(self) -> List[nn.Module]:
        return []

    @property
    def text_encoder(self) -> Optional[nn.Module]:
        return None

    @property
    def preprocessing_modules(self) -> List[str]:
        return ["vae", "vit", "connector", "vit_pos_embed"]

    @property
    def inference_modules(self) -> List[str]:
        return [
            "transformer", "vit", "vae",
            "vae2llm", "llm2vae",
            "time_embedder", "latent_pos_embed",
            "connector", "vit_pos_embed",
        ]

    @property
    def bagel_model(self) -> nn.Module:
        return self.get_component("transformer")

    @property
    def bagel_config(self):
        return self.pipeline.config

    def _set_attention_backend(self) -> None:
        backend = self.model_args.attn_backend
        if backend is None:
            return
        try:
            from .modeling.bagel.attention_dispatch import set_attn_backend
            set_attn_backend(backend)
        except ImportError:
            pass

    # ======================== Encoding ========================

    def encode_prompt(self, prompt: Union[str, List[str]]) -> Dict[str, Any]:
        if isinstance(prompt, str):
            prompt = [prompt]
        return {"prompt": prompt}

    def standardize_images(
        self, images: Union[ImageSingle, ImageBatch],
        output_type: Literal['pil', 'pt', 'np'] = 'pil',
    ) -> ImageBatch:
        if is_image(images):
            images = [images]
        return standardize_image_batch(images, output_type=output_type)

    def encode_image(
        self, images: Union[ImageSingle, ImageBatch, MultiImageBatch],
        condition_image_size: Union[int, Tuple[int, int]] = CONDITION_IMAGE_SIZE,
        device: Optional[torch.device] = None,
    ) -> Optional[Dict[str, Any]]:
        if images is None:
            return None
        device = device or self.device

        if is_image(images):
            images = [[images]]
        elif is_image_batch(images):
            images = [images]

        condition_image_size = (
            (condition_image_size, condition_image_size)
            if isinstance(condition_image_size, int) else condition_image_size
        )
        max_area = condition_image_size[0] * condition_image_size[1]

        images = [self.standardize_images(batch, output_type='pil') for batch in images]

        condition_images = []
        for batch in images:
            batch_tensors = []
            for img in batch:
                w, h = img.size
                if w * h > max_area:
                    new_w, new_h = calculate_dimensions(max_area, h / w)
                    img = img.resize((new_w, new_h), resample=Image.BICUBIC)
                t = pil_image_to_tensor(img).to(device)
                batch_tensors.append(t)
            condition_images.append(batch_tensors)

        return {"condition_images": condition_images}

    def encode_video(self, videos: Any):
        pass

    # ======================== Decoding ========================

    def decode_latents(
        self,
        latents: torch.Tensor,
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> Union[Image.Image, List[Image.Image]]:
        """Decode packed latent tokens back into PIL images."""
        vae = self.pipeline.vae
        ds = self.pipeline.latent_downsample
        p = self.pipeline.latent_patch_size
        ch = self.pipeline.latent_channel

        single = latents.dim() == 2
        if single:
            latents = latents.unsqueeze(0)

        images = []
        for lat in latents:
            H, W = image_shape
            h, w = H // ds, W // ds
            lat = lat.reshape(1, h, w, p, p, ch)
            lat = torch.einsum("nhwpqc->nchpwq", lat)
            lat = lat.reshape(1, ch, h * p, w * p)
            decoded = vae.decode(lat.to(vae.dtype if hasattr(vae, 'dtype') else torch.bfloat16))
            decoded = (decoded * 0.5 + 0.5).clamp(0, 1)[0].float()
            images.append(decoded)

        return images[0] if single else images

    # ════════════════════════════════════════════════════════════════
    # Forward Cache Update (on Adapter — uses self.transformer)
    # ════════════════════════════════════════════════════════════════

    def forward_cache_update_text(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.Tensor,
        packed_text_position_ids: torch.Tensor,
        text_token_lens: torch.Tensor,
        packed_text_indexes: torch.Tensor,
        packed_key_value_indexes: torch.Tensor,
        key_values_lens: torch.Tensor,
    ) -> NaiveCache:
        """Update KV-cache with text tokens via ``self.transformer`` (accelerator-wrapped)."""
        packed_text_embedding = self.transformer(text_ids=packed_text_ids, mode='embed')
        extra = {"mode": "und"} if self.pipeline.use_moe else {}

        output = self.transformer(
            packed_query_sequence=packed_text_embedding,
            query_lens=text_token_lens,
            packed_query_position_ids=packed_text_position_ids,
            packed_query_indexes=packed_text_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            is_causal=True,
            **extra,
        )
        return output.past_key_values

    def forward_cache_update_vit(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.Tensor,
        packed_text_indexes: torch.Tensor,
        packed_vit_tokens: torch.Tensor,
        packed_vit_token_indexes: torch.Tensor,
        packed_vit_position_ids: torch.Tensor,
        vit_token_seqlens: torch.Tensor,
        packed_position_ids: torch.Tensor,
        packed_seqlens: torch.Tensor,
        packed_indexes: torch.Tensor,
        packed_key_value_indexes: torch.Tensor,
        key_values_lens: torch.Tensor,
    ) -> NaiveCache:
        """Update KV-cache with ViT image tokens via ``self.transformer`` (accelerator-wrapped)."""
        packed_text_embedding = self.transformer(text_ids=packed_text_ids, mode='embed')
        hidden_size = self.pipeline.hidden_size
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        cu_seqlens = torch.nn.functional.pad(
            torch.cumsum(vit_token_seqlens, dim=0), (1, 0)
        ).to(torch.int32)
        max_seqlen = torch.max(vit_token_seqlens).item()
        packed_vit_token_embed = self.pipeline.vit(
            packed_pixel_values=packed_vit_tokens,
            packed_flattened_position_ids=packed_vit_position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        packed_vit_token_embed = self.pipeline.connector(packed_vit_token_embed)
        pos_emb = self.pipeline.vit_pos_embed(packed_vit_position_ids)
        packed_vit_token_embed = packed_vit_token_embed + pos_emb
        if packed_vit_token_embed.dtype != packed_sequence.dtype:
            packed_vit_token_embed = packed_vit_token_embed.to(packed_sequence.dtype)
        packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed

        extra = {"mode": "und"} if self.pipeline.use_moe else {}

        output = self.transformer(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            is_causal=False,
            **extra,
        )
        return output.past_key_values

    def forward_cache_update_vae(
        self,
        past_key_values: NaiveCache,
        padded_images: torch.Tensor,
        patchified_vae_latent_shapes: list,
        packed_vae_position_ids: torch.Tensor,
        packed_timesteps: torch.Tensor,
        packed_vae_token_indexes: torch.Tensor,
        packed_text_ids: torch.Tensor,
        packed_text_indexes: torch.Tensor,
        packed_position_ids: torch.Tensor,
        packed_seqlens: torch.Tensor,
        packed_indexes: torch.Tensor,
        key_values_lens: torch.Tensor,
        packed_key_value_indexes: torch.Tensor,
    ) -> NaiveCache:
        """Update KV-cache with VAE-encoded image tokens via ``self.transformer`` (accelerator-wrapped)."""
        packed_text_embedding = self.transformer(text_ids=packed_text_ids, mode='embed')
        hidden_size = self.pipeline.hidden_size
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        padded_latent = self.pipeline.vae.encode(padded_images)

        packed_latent = []
        for latent, (h, w) in zip(padded_latent, patchified_vae_latent_shapes):
            packed_latent.append(self.pipeline.patchify_latent(latent, h, w))
        packed_latent = torch.cat(packed_latent, dim=0)

        packed_pos_embed = self.pipeline.latent_pos_embed(packed_vae_position_ids)
        packed_timestep_embeds = self.pipeline.time_embedder(packed_timesteps)
        packed_latent = self.pipeline.vae2llm(packed_latent) + packed_timestep_embeds + packed_pos_embed
        if packed_latent.dtype != packed_sequence.dtype:
            packed_latent = packed_latent.to(packed_sequence.dtype)
        packed_sequence[packed_vae_token_indexes] = packed_latent

        extra = {}
        if self.pipeline.use_moe:
            extra = {
                "mode": "gen",
                "packed_vae_token_indexes": packed_vae_token_indexes,
                "packed_text_indexes": packed_text_indexes,
            }

        output = self.transformer(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=True,
            is_causal=False,
            **extra,
        )
        return output.past_key_values

    # ════════════════════════════════════════════════════════════════
    # Context Building — returns both KV-cache AND context specs
    # ════════════════════════════════════════════════════════════════

    def _build_gen_context(
        self,
        prompt: str,
        condition_images: Optional[List[torch.Tensor]] = None,
        think: bool = False,
    ) -> Tuple[Dict, Dict, Dict, ContextSpec, ContextSpec, ContextSpec]:
        """
        Build KV-cache contexts AND context specs for later re-materialization.

        Returns:
            (gen_ctx, cfg_text_ctx, cfg_img_ctx,
             gen_spec, cfg_text_spec, cfg_img_spec)
        """
        from .modeling.bagel.qwen2_navit import NaiveCache

        num_layers = self.pipeline.config.llm_config.num_hidden_layers

        def _init_ctx():
            return {
                "kv_lens": [0], "ropes": [0],
                "past_key_values": NaiveCache(num_layers),
            }

        gen_ctx = _init_ctx()
        cfg_text_ctx = _init_ctx()
        cfg_img_ctx = _init_ctx()

        gen_spec = ContextSpec()
        cfg_text_spec = ContextSpec()
        cfg_img_spec = ContextSpec()

        if think:
            system_prompt = (
                "You should first think about the planning process in the mind "
                "and then generate the image.\nThe planning process is enclosed "
                "within <think> </think> tags."
            )
            gen_ctx = self._update_context_text(system_prompt, gen_ctx)
            cfg_img_ctx = self._update_context_text(system_prompt, cfg_img_ctx)
            gen_spec.steps.append(("text", system_prompt))
            cfg_img_spec.steps.append(("text", system_prompt))

        if condition_images is not None:
            for img_tensor in condition_images:
                gen_ctx = self._update_context_image(img_tensor, gen_ctx)
                gen_spec.steps.append(("vae_image", img_tensor))
                gen_spec.steps.append(("vit_image", img_tensor))

        cfg_text_ctx = deepcopy(gen_ctx)
        cfg_text_spec = deepcopy(gen_spec)

        gen_ctx = self._update_context_text(prompt, gen_ctx)
        cfg_img_ctx = self._update_context_text(prompt, cfg_img_ctx)
        gen_spec.steps.append(("text", prompt))
        cfg_img_spec.steps.append(("text", prompt))

        gen_spec.kv_lens = gen_ctx["kv_lens"]
        gen_spec.ropes = gen_ctx["ropes"]
        cfg_text_spec.kv_lens = cfg_text_ctx["kv_lens"]
        cfg_text_spec.ropes = cfg_text_ctx["ropes"]
        cfg_img_spec.kv_lens = cfg_img_ctx["kv_lens"]
        cfg_img_spec.ropes = cfg_img_ctx["ropes"]

        return (gen_ctx, cfg_text_ctx, cfg_img_ctx,
                gen_spec, cfg_text_spec, cfg_img_spec)

    @torch.no_grad()
    def _update_context_text(self, text: str, gen_context: Dict) -> Dict:
        """Add text tokens to KV-cache context (inference-time helper)."""
        device = self.device
        gen_input, kv_lens, ropes = self.pipeline.prepare_prompts(
            curr_kvlens=gen_context["kv_lens"], curr_rope=gen_context["ropes"],
            prompts=[text], tokenizer=self._tokenizer,
            new_token_ids=self.new_token_ids,
        )
        gen_input = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in gen_input.items()}
        past_key_values = self.forward_cache_update_text(
            gen_context["past_key_values"], **gen_input
        )
        return {"kv_lens": kv_lens, "ropes": ropes, "past_key_values": past_key_values}

    @torch.no_grad()
    def _update_context_image(
        self, image, gen_context: Dict,
        vae: bool = True, vit: bool = True,
    ) -> Dict:
        """Add image tokens (ViT + VAE) to KV-cache context (inference-time helper)."""
        device = self.device
        kv_lens = gen_context["kv_lens"]
        ropes = gen_context["ropes"]
        past_key_values = gen_context["past_key_values"]
        image = self.standardize_images(image, output_type='pil')[0]

        if vae:
            gen_input, kv_lens, ropes = self.pipeline.prepare_vae_images(
                curr_kvlens=kv_lens, curr_rope=ropes,
                images=[image], transforms=self.vae_transform,
                new_token_ids=self.new_token_ids,
            )
            gen_input = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in gen_input.items()}
            past_key_values = self.forward_cache_update_vae(past_key_values, **gen_input)

        if vit:
            gen_input, kv_lens, ropes = self.pipeline.prepare_vit_images(
                curr_kvlens=kv_lens, curr_rope=ropes,
                images=[image], transforms=self.vit_transform,
                new_token_ids=self.new_token_ids,
            )
            gen_input = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in gen_input.items()}
            past_key_values = self.forward_cache_update_vit(past_key_values, **gen_input)

        return {"kv_lens": kv_lens, "ropes": ropes, "past_key_values": past_key_values}

    # ════════════════════════════════════════════════════════════════
    # Context Re-materialization (for training with updated params)
    # ════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def _rebuild_context(self, spec: ContextSpec) -> NaiveCache:
        """
        Rebuild KV-cache from a ContextSpec using ``self.transformer``.

        Ensures context reflects current transformer parameters during
        training, not stale cached states from sampling.
        """
        from .modeling.bagel.qwen2_navit import NaiveCache

        num_layers = self.pipeline.config.llm_config.num_hidden_layers
        device = self.device

        past_key_values = NaiveCache(num_layers)
        kv_lens = [0]
        ropes = [0]

        for step_type, step_data in spec.steps:
            if step_type == "text":
                gen_input, kv_lens, ropes = self.pipeline.prepare_prompts(
                    curr_kvlens=kv_lens, curr_rope=ropes,
                    prompts=[step_data], tokenizer=self._tokenizer,
                    new_token_ids=self.new_token_ids,
                )
                gen_input = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in gen_input.items()}
                past_key_values = self.forward_cache_update_text(past_key_values, **gen_input)

            elif step_type == "vit_image":
                image = self.standardize_images(step_data, output_type='pil')[0]
                gen_input, kv_lens, ropes = self.pipeline.prepare_vit_images(
                    curr_kvlens=kv_lens, curr_rope=ropes,
                    images=[image], transforms=self.vit_transform,
                    new_token_ids=self.new_token_ids,
                )
                gen_input = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in gen_input.items()}
                past_key_values = self.forward_cache_update_vit(past_key_values, **gen_input)

            elif step_type == "vae_image":
                image = self.standardize_images(step_data, output_type='pil')[0]
                gen_input, kv_lens, ropes = self.pipeline.prepare_vae_images(
                    curr_kvlens=kv_lens, curr_rope=ropes,
                    images=[image], transforms=self.vae_transform,
                    new_token_ids=self.new_token_ids,
                )
                gen_input = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in gen_input.items()}
                past_key_values = self.forward_cache_update_vae(past_key_values, **gen_input)

        return past_key_values

    # ════════════════════════════════════════════════════════════════
    # _forward_flow — lives here for distributed training compat
    # ════════════════════════════════════════════════════════════════

    def _forward_flow(
        self,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        # Generation context inputs
        packed_vae_token_indexes: torch.Tensor,
        packed_vae_position_ids: torch.Tensor,
        packed_text_ids: torch.Tensor,
        packed_text_indexes: torch.Tensor,
        packed_indexes: torch.Tensor,
        packed_position_ids: torch.Tensor,
        packed_seqlens: torch.Tensor,
        key_values_lens: torch.Tensor,
        past_key_values: NaiveCache,
        packed_key_value_indexes: torch.Tensor,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        # cfg_text
        cfg_text_scale: float = 1.0,
        cfg_text_packed_position_ids: Optional[torch.Tensor] = None,
        cfg_text_packed_query_indexes: Optional[torch.Tensor] = None,
        cfg_text_key_values_lens: Optional[torch.Tensor] = None,
        cfg_text_past_key_values: Optional[NaiveCache] = None,
        cfg_text_packed_key_value_indexes: Optional[torch.Tensor] = None,
        # cfg_img
        cfg_img_scale: float = 1.0,
        cfg_img_packed_position_ids: Optional[torch.Tensor] = None,
        cfg_img_packed_query_indexes: Optional[torch.Tensor] = None,
        cfg_img_key_values_lens: Optional[torch.Tensor] = None,
        cfg_img_past_key_values: Optional[NaiveCache] = None,
        cfg_img_packed_key_value_indexes: Optional[torch.Tensor] = None,
    ):
        """
        Flow velocity prediction with CFG.
        """

        packed_text_embedding = self.transformer(text_ids=packed_text_ids, mode='embed')
        hidden_size = self.pipeline.config.llm_config.hidden_size
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        assert timestep.unique().shape[0] == 1
        packed_pos_embed = self.pipeline.latent_pos_embed(packed_vae_position_ids)
        packed_timestep_embeds = self.pipeline.time_embedder(timestep)
        x_t = self.pipeline.vae2llm(x_t) + packed_timestep_embeds + packed_pos_embed
        if x_t.dtype != packed_sequence.dtype:
            x_t = x_t.to(packed_sequence.dtype)
        packed_sequence[packed_vae_token_indexes] = x_t

        extra_inputs = {}
        if self.pipeline.use_moe:
            extra_inputs = {
                "mode": "gen",
                "packed_vae_token_indexes": packed_vae_token_indexes,
                "packed_text_indexes": packed_text_indexes,
            }

        output = self.transformer(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=False,
            is_causal=False,
            **extra_inputs,
        )
        v_t = self.pipeline.llm2vae(output.packed_query_sequence)
        v_t = v_t[packed_vae_token_indexes]

        # ── CFG Text ──
        if cfg_text_scale > 1.0:
            cfg_text_output = self.transformer(
                packed_query_sequence=packed_sequence,
                query_lens=packed_seqlens,
                packed_query_position_ids=cfg_text_packed_position_ids,
                packed_query_indexes=cfg_text_packed_query_indexes,
                past_key_values=cfg_text_past_key_values,
                key_values_lens=cfg_text_key_values_lens,
                packed_key_value_indexes=cfg_text_packed_key_value_indexes,
                update_past_key_values=False,
                is_causal=False,
                **extra_inputs,
            )
            cfg_text_v_t = self.pipeline.llm2vae(cfg_text_output.packed_query_sequence)
            cfg_text_v_t = cfg_text_v_t[packed_vae_token_indexes]

        # ── CFG Image ──
        if cfg_img_scale > 1.0:
            cfg_img_output = self.transformer(
                packed_query_sequence=packed_sequence,
                query_lens=packed_seqlens,
                packed_query_position_ids=cfg_img_packed_position_ids,
                packed_query_indexes=cfg_img_packed_query_indexes,
                past_key_values=cfg_img_past_key_values,
                key_values_lens=cfg_img_key_values_lens,
                packed_key_value_indexes=cfg_img_packed_key_value_indexes,
                update_past_key_values=False,
                is_causal=False,
                **extra_inputs,
            )
            cfg_img_v_t = self.pipeline.llm2vae(cfg_img_output.packed_query_sequence)
            cfg_img_v_t = cfg_img_v_t[packed_vae_token_indexes]

        # ── CFG Combination ──
        if cfg_text_scale > 1.0:
            if cfg_renorm_type == "text_channel":
                v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
                norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                norm_v_t_text_ = torch.norm(v_t_text_, dim=-1, keepdim=True)
                scale = (norm_v_t / (norm_v_t_text_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                v_t_text = v_t_text_ * scale
                if cfg_img_scale > 1.0:
                    v_t = cfg_img_v_t + cfg_img_scale * (v_t_text - cfg_img_v_t)
                else:
                    v_t = v_t_text
            else:
                v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
                if cfg_img_scale > 1.0:
                    v_t_ = cfg_img_v_t + cfg_img_scale * (v_t_text_ - cfg_img_v_t)
                else:
                    v_t_ = v_t_text_

                if cfg_renorm_type == "global":
                    norm_v_t = torch.norm(v_t)
                    norm_v_t_ = torch.norm(v_t_)
                elif cfg_renorm_type == "channel":
                    norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                    norm_v_t_ = torch.norm(v_t_, dim=-1, keepdim=True)
                else:
                    raise NotImplementedError(f"{cfg_renorm_type} is not supported")
                scale = (norm_v_t / (norm_v_t_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                v_t = v_t_ * scale

        return v_t

    # ════════════════════════════════════════════════════════════════
    # Forward (Training & Inference)
    # ════════════════════════════════════════════════════════════════

    def forward(
        self,
        t: torch.Tensor,
        latents: torch.Tensor,
        # Generation Input
        generation_input: Dict[str, torch.Tensor],
        # Context specs (for training re-materialization)
        gen_context_spec: Optional[Union[ContextSpec, List[ContextSpec]]] = None,
        cfg_text_context_spec: Optional[Union[ContextSpec, List[ContextSpec]]] = None,
        cfg_img_context_spec: Optional[Union[ContextSpec, List[ContextSpec]]] = None,
        # OR pre-built KV-caches (for inference)
        past_key_values: Optional[Union[NaiveCache, List[NaiveCache]]] = None,
        cfg_text_past_kv: Optional[Union[NaiveCache, List[NaiveCache]]] = None,
        cfg_img_past_kv: Optional[Union[NaiveCache, List[NaiveCache]]] = None,
        # CFG generation inputs
        cfg_text_generation_input: Optional[Dict[str, torch.Tensor]] = None,
        cfg_img_generation_input: Optional[Dict[str, torch.Tensor]] = None,
        # CFG parameters
        cfg_text_scale: float = 4.0,
        cfg_img_scale: float = 1.5,
        cfg_interval: Tuple[float, float] = (0.4, 1.0),
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        t_next: Optional[torch.Tensor] = None,
        next_latents: Optional[torch.Tensor] = None,
        noise_level: Optional[float] = None,
        compute_log_prob: bool = True,
        return_kwargs: List[str] = [
            "noise_pred", "next_latents", "next_latents_mean",
            "std_dev_t", "dt", "log_prob",
        ],
        # Whether to rebuild context (True during training)
        rebuild_context: Optional[bool] = None,
    ) -> SDESchedulerOutput:
        """
        Single denoising step: flow prediction → scheduler step.

        During training (when context specs are provided), KV-caches are
        rebuilt from specs using current transformer parameters.
        During inference, pre-built KV-caches are used directly.
        """
        device = latents.device

        # Decide whether to rebuild context
        if rebuild_context is None:
            # Auto-detect: rebuild if we have specs
            rebuild_context = gen_context_spec is not None and past_key_values is None 

        # 1. Build or reuse KV-caches
        if rebuild_context:
            def to_single_spec(spec):
                if isinstance(spec, list):
                    assert len(spec) == 1, f"Only batch_size 1 is supported for Bagel context rebuilding, but got batch of size {len(spec)}"
                    return spec[0]
                return spec
            gen_context_spec = to_single_spec(gen_context_spec)
            if cfg_text_context_spec is not None:
                cfg_text_context_spec = to_single_spec(cfg_text_context_spec)
            if cfg_img_context_spec is not None:
                cfg_img_context_spec = to_single_spec(cfg_img_context_spec)
            # Training: rebuild context with current (updated) transformer params
            past_key_values = self._rebuild_context(gen_context_spec)
            if cfg_text_context_spec is not None:
                cfg_text_past_kv = self._rebuild_context(cfg_text_context_spec)
            if cfg_img_context_spec is not None:
                cfg_img_past_kv = self._rebuild_context(cfg_img_context_spec)
        else:
            # Inference: use provided KV-caches directly
            def convert_naive_cache(kv) -> NaiveCache:
                if isinstance(kv, NaiveCache):
                    return kv
                return NaiveCache.from_NaiveCache_list(kv)

            past_key_values = convert_naive_cache(past_key_values)
            if cfg_text_past_kv is not None:
                cfg_text_past_kv = convert_naive_cache(cfg_text_past_kv)
            if cfg_img_past_kv is not None:
                cfg_img_past_kv = convert_naive_cache(cfg_img_past_kv)

        # 2. Handle batch dimension
        all_first_dimension = [
            t.shape[0] for t in (
                list(generation_input.values())
                + list((cfg_text_generation_input or {}).values())
                + list((cfg_img_generation_input or {}).values())
            ) if isinstance(t, torch.Tensor)
        ]
        if all_first_dimension and all(d == all_first_dimension[0] for d in all_first_dimension):
            if all(d == 1 for d in all_first_dimension):
                generation_input = {
                    k: v.squeeze(0) if isinstance(v, torch.Tensor) else v
                    for k, v in generation_input.items()
                }
                if cfg_text_generation_input is not None:
                    cfg_text_generation_input = {
                        k: v.squeeze(0) if isinstance(v, torch.Tensor) else v
                        for k, v in cfg_text_generation_input.items()
                    }
                if cfg_img_generation_input is not None:
                    cfg_img_generation_input = {
                        k: v.squeeze(0) if isinstance(v, torch.Tensor) else v
                        for k, v in cfg_img_generation_input.items()
                    }

        # 3. Timestep conversion [0, 1000] → [0, 1]
        sigma = t.float() / 1000.0
        timestep_for_bagel = sigma.expand(latents.shape[0])

        # 4. CFG gating
        sigma_val = sigma.flatten()[0].item()
        cfg_text_s = cfg_text_scale if cfg_interval[0] < sigma_val <= cfg_interval[1] else 1.0
        cfg_img_s = cfg_img_scale if cfg_interval[0] < sigma_val <= cfg_interval[1] else 1.0

        def _cfg(d, key):
            if d is None:
                return None
            v = d.get(key)
            return v.to(device) if isinstance(v, torch.Tensor) else None

        # 5. Flow velocity prediction
        noise_pred = self._forward_flow(
            x_t=latents,
            timestep=timestep_for_bagel,
            packed_vae_token_indexes=generation_input["packed_vae_token_indexes"].to(device),
            packed_vae_position_ids=generation_input["packed_vae_position_ids"].to(device),
            packed_text_ids=generation_input["packed_text_ids"].to(device),
            packed_text_indexes=generation_input["packed_text_indexes"].to(device),
            packed_position_ids=generation_input["packed_position_ids"].to(device),
            packed_indexes=generation_input["packed_indexes"].to(device),
            packed_seqlens=generation_input["packed_seqlens"].to(device),
            key_values_lens=generation_input["key_values_lens"].to(device),
            past_key_values=past_key_values,
            packed_key_value_indexes=generation_input["packed_key_value_indexes"].to(device),
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            cfg_text_scale=cfg_text_s,
            cfg_text_packed_position_ids=_cfg(cfg_text_generation_input, "cfg_packed_position_ids"),
            cfg_text_packed_query_indexes=_cfg(cfg_text_generation_input, "cfg_packed_query_indexes"),
            cfg_text_key_values_lens=_cfg(cfg_text_generation_input, "cfg_key_values_lens"),
            cfg_text_past_key_values=cfg_text_past_kv,
            cfg_text_packed_key_value_indexes=_cfg(cfg_text_generation_input, "cfg_packed_key_value_indexes"),
            cfg_img_scale=cfg_img_s,
            cfg_img_packed_position_ids=_cfg(cfg_img_generation_input, "cfg_packed_position_ids"),
            cfg_img_packed_query_indexes=_cfg(cfg_img_generation_input, "cfg_packed_query_indexes"),
            cfg_img_key_values_lens=_cfg(cfg_img_generation_input, "cfg_key_values_lens"),
            cfg_img_past_key_values=cfg_img_past_kv,
            cfg_img_packed_key_value_indexes=_cfg(cfg_img_generation_input, "cfg_packed_key_value_indexes"),
        )

        # 6. Scheduler step
        output = self.scheduler.step(
            noise_pred=noise_pred,
            timestep=t,
            latents=latents,
            timestep_next=t_next,
            next_latents=next_latents,
            noise_level=noise_level,
            return_dict=True,
            compute_log_prob=compute_log_prob,
            return_kwargs=return_kwargs,
        )

        return output

    # ════════════════════════════════════════════════════════════════
    # Denoising Loop
    # ════════════════════════════════════════════════════════════════

    def _denoise_loop(
        self,
        generation_input: Dict[str, torch.Tensor],
        cfg_text_generation_input: Dict[str, torch.Tensor],
        cfg_img_generation_input: Dict[str, torch.Tensor],
        past_key_values: NaiveCache,
        cfg_text_past_kv: NaiveCache,
        cfg_img_past_kv: NaiveCache,
        num_inference_steps: int,
        cfg_text_scale: float,
        cfg_img_scale: float,
        cfg_interval: Tuple[float, float],
        cfg_renorm_min: float,
        cfg_renorm_type: str,
        compute_log_prob: bool,
        trajectory_indices: TrajectoryIndicesType,
        extra_call_back_kwargs: List[str],
        device: torch.device,
    ) -> Dict[str, Any]:
        """Core denoising loop (inference only, uses pre-built KV-caches)."""
        unshifted_sigmas = np.linspace(1.0, 0.0, num_inference_steps)[:-1]
        self.scheduler.set_timesteps(sigmas=unshifted_sigmas.tolist(), device=device) # Shift is applied inside
        timesteps = self.scheduler.timesteps

        x_t = generation_input["packed_init_noises"].to(device)
        generation_input = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in generation_input.items()
        }

        latent_collector = create_trajectory_collector(trajectory_indices, num_inference_steps)
        latent_collector.collect(x_t, step_idx=0)
        log_prob_collector = (
            create_trajectory_collector(trajectory_indices, num_inference_steps)
            if compute_log_prob else None
        )
        callback_collector = create_callback_collector(trajectory_indices, num_inference_steps)

        for i, t in enumerate(timesteps):
            t_next = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(0.0, device=device)
            current_noise_level = self.scheduler.get_noise_level_for_timestep(t)
            current_compute_log_prob = compute_log_prob and current_noise_level > 0
            return_kwargs = list(set(
                ['next_latents', 'log_prob', 'noise_pred'] + extra_call_back_kwargs
            ))

            output = self.forward(
                t=t.unsqueeze(0),
                latents=x_t,
                past_key_values=past_key_values,
                cfg_text_past_kv=cfg_text_past_kv,
                cfg_img_past_kv=cfg_img_past_kv,
                generation_input=generation_input,
                cfg_text_generation_input=cfg_text_generation_input,
                cfg_img_generation_input=cfg_img_generation_input,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                cfg_interval=cfg_interval,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                t_next=t_next.unsqueeze(0),
                noise_level=current_noise_level,
                compute_log_prob=current_compute_log_prob,
                return_kwargs=return_kwargs,
                rebuild_context=False,  # Inference: use pre-built caches
            )

            x_t = output.next_latents
            latent_collector.collect(x_t, step_idx=i + 1)
            if current_compute_log_prob and log_prob_collector is not None:
                log_prob_collector.collect(output.log_prob, step_idx=i)
            callback_collector.collect_step(
                step_idx=i, output=output,
                keys=extra_call_back_kwargs,
                capturable={"noise_level": current_noise_level},
            )

        extra_call_back_res = callback_collector.get_result()
        callback_index_map = callback_collector.get_index_map()
        all_latents = latent_collector.get_result()
        latent_index_map = latent_collector.get_index_map()
        all_log_probs = log_prob_collector.get_result() if log_prob_collector is not None else None
        log_prob_index_map = log_prob_collector.get_index_map() if log_prob_collector is not None else None

        return {
            "final_packed_latent": x_t,
            "all_latents": all_latents,
            "all_log_probs": all_log_probs,
            "timesteps": timesteps,
            "latent_index_map": latent_index_map,
            "log_prob_index_map": log_prob_index_map,
            "callback_results": extra_call_back_res,
            "callback_index_map": callback_index_map,
        }

    # ════════════════════════════════════════════════════════════════
    # Inference
    # ════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def inference(
        self,
        num_inference_steps: int = 50,
        height: int = 1024,
        width: int = 1024,
        prompt: Union[str, List[str]] = None,
        images: Optional[Union[ImageSingle, ImageBatch, MultiImageBatch]] = None,
        condition_images: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        condition_image_size: Union[int, Tuple[int, int]] = CONDITION_IMAGE_SIZE,
        cfg_text_scale: float = 4.0,
        cfg_img_scale: float = 1.5,
        cfg_interval: Tuple[float, float] = (0.4, 1.0),
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        compute_log_prob: bool = True,
        extra_call_back_kwargs: List[str] = [],
        trajectory_indices: TrajectoryIndicesType = "all",
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        think: bool = False,
    ) -> List[Union[BagelSample, BagelI2ISample]]:
        """Full generation: build context → denoise → decode → return samples."""
        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)

        device = self.device
        image_shape = (height, width)
        is_i2i = (condition_images is not None or images is not None)
        if is_i2i:
            if condition_images is None:
                encoded = self.encode_image(images, condition_image_size, device)
                condition_images = encoded["condition_images"] if encoded else None
            else:
                condition_images = [[t.to(device) for t in imgs] for imgs in condition_images]

        samples = []
        for b in range(batch_size):
            cur_prompt = prompt[b]
            cur_cond_images = condition_images[b] if condition_images is not None else None

            # 1. Build KV-cache contexts AND specs
            (gen_ctx, cfg_text_ctx, cfg_img_ctx,
             gen_spec, cfg_text_spec, cfg_img_spec) = self._build_gen_context(
                prompt=cur_prompt,
                condition_images=cur_cond_images,
                think=think,
            )

            # 2. Prepare latent generation inputs
            gen_input = self.pipeline.prepare_vae_latent(
                curr_kvlens=gen_ctx["kv_lens"], curr_rope=gen_ctx["ropes"],
                image_sizes=[image_shape], new_token_ids=self.new_token_ids,
                device=device, generator=generator,
            )
            cfg_text_gen_input = self.pipeline.prepare_vae_latent_cfg(
                curr_kvlens=cfg_text_ctx["kv_lens"], curr_rope=cfg_text_ctx["ropes"],
                image_sizes=[image_shape], device=device,
            )
            cfg_img_gen_input = self.pipeline.prepare_vae_latent_cfg(
                curr_kvlens=cfg_img_ctx["kv_lens"], curr_rope=cfg_img_ctx["ropes"],
                image_sizes=[image_shape], device=device,
            )

            # 3. Denoise (using pre-built KV-caches)
            result = self._denoise_loop(
                generation_input=gen_input,
                cfg_text_generation_input=cfg_text_gen_input,
                cfg_img_generation_input=cfg_img_gen_input,
                past_key_values=gen_ctx["past_key_values"],
                cfg_text_past_kv=cfg_text_ctx["past_key_values"],
                cfg_img_past_kv=cfg_img_ctx["past_key_values"],
                num_inference_steps=num_inference_steps,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                cfg_interval=cfg_interval,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                compute_log_prob=compute_log_prob,
                trajectory_indices=trajectory_indices,
                extra_call_back_kwargs=extra_call_back_kwargs,
                device=device,
            )

            # 4. Decode
            image = self.decode_latents(result["final_packed_latent"], image_shape=image_shape)

            # 5. Build sample — store specs instead of KV-caches
            SampleCls = BagelI2ISample if is_i2i else BagelSample
            sample = SampleCls(
                timesteps=result["timesteps"],
                all_latents=(
                    torch.stack([lat[0] for lat in result['all_latents']], dim=0)
                    if result['all_latents'] is not None else None
                ),
                log_probs=(
                    torch.stack([lp[0] for lp in result['all_log_probs']], dim=0)
                    if result['all_log_probs'] is not None else None
                ),
                latent_index_map=result["latent_index_map"],
                log_prob_index_map=result["log_prob_index_map"],
                image=image,
                height=height,
                width=width,
                prompt=cur_prompt,
                # Store context SPECS (not KV-caches!) for training re-materialization
                gen_context_spec=gen_spec,
                cfg_text_context_spec=cfg_text_spec,
                cfg_img_context_spec=cfg_img_spec,
                # Stateless generation inputs (no transformer dependency)
                generation_input=gen_input,
                cfg_text_generation_input=cfg_text_gen_input,
                cfg_img_generation_input=cfg_img_gen_input,
                image_shape=image_shape,
                **(
                    {"condition_images": cur_cond_images}
                    if is_i2i and hasattr(SampleCls, "condition_images")
                    else {}
                ),
                extra_kwargs={
                    **result.get("callback_results", {}),
                    "callback_index_map": result.get("callback_index_map"),
                },
            )
            samples.append(sample)

        return samples