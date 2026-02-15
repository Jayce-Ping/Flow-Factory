# src/flow_factory/models/bagel/pipeline.py
"""
Bagel Pseudo-Pipeline (Refactored — Flat Component Hierarchy)

Owns all sub-models as direct attributes (no Bagel wrapper class).
Provides ``prepare_*`` / ``forward_cache_update_*`` methods for
KV-cache context building and a ``forward_denoise_step`` for single-step
flow velocity prediction with CFG.

Architecture (flat, diffusers-like)::

    BagelPseudoPipeline
      ├── .transformer        Qwen2ForCausalLM   (LLM backbone)
      ├── .vit                SiglipVisionModel   (image understanding)
      ├── .vae                AutoEncoder          (encode / decode images)
      ├── .vae2llm            nn.Linear            (latent → LLM space)
      ├── .llm2vae            nn.Linear            (LLM → latent space)
      ├── .time_embedder      TimestepEmbedder     (timestep conditioning)
      ├── .latent_pos_embed   PositionEmbedding    (latent positions)
      ├── .connector          MLPconnector         (ViT → LLM projection)
      ├── .vit_pos_embed      PositionEmbedding    (ViT positions)
      └── .config             BagelConfig
"""
from __future__ import annotations

import os
import logging
from typing import Optional, Any, Dict, List, Tuple

import torch
import torch.nn as nn
from transformers.configuration_utils import PretrainedConfig
from .modeling import (
    Qwen2ForCausalLM,
    Qwen2Config,
    SiglipVisionModel,
    SiglipVisionConfig,
    AutoEncoder,
    AutoEncoderParams,
    TimestepEmbedder,
    MLPconnector,
    PositionEmbedding
)

logger = logging.getLogger(__name__)

# ── Checkpoint key remapping (Bagel wrapper → flat pipeline) ──────────────

_PREFIX_MAP = {
    "language_model.": "transformer.",
    "vit_model.":      "vit.",
    "vae2llm.":        "vae2llm.",
    "llm2vae.":        "llm2vae.",
    "time_embedder.":  "time_embedder.",
    "latent_pos_embed.": "latent_pos_embed.",
    "connector.":      "connector.",
    "vit_pos_embed.":  "vit_pos_embed.",
}


def _remap_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remap Bagel checkpoint keys to flat pipeline layout."""
    new_sd: Dict[str, torch.Tensor] = {}
    for key, val in state_dict.items():
        for old_prefix, new_prefix in _PREFIX_MAP.items():
            if key.startswith(old_prefix):
                new_sd[key.replace(old_prefix, new_prefix, 1)] = val
                break
        # Keys that don't match any prefix are skipped (e.g. lm_head)
    return new_sd


def _resolve_model_path(model_path: str, **kwargs) -> str:
    """Resolve *model_path* to a local directory (download from Hub if needed)."""
    if os.path.isdir(model_path):
        return model_path

    from huggingface_hub import snapshot_download

    _SNAPSHOT_KEYS = {
        "revision", "cache_dir", "token", "local_dir",
        "allow_patterns", "ignore_patterns",
        "force_download", "resume_download", "local_files_only",
    }
    dl_kwargs = {k: v for k, v in kwargs.items() if k in _SNAPSHOT_KEYS}

    local_dir = snapshot_download(repo_id=model_path, **dl_kwargs)
    return local_dir

# ===========================================================================
# BagelConfig
# ==========================================================================

class BagelConfig(PretrainedConfig):
    def __init__(
        self,
        llm_config : Qwen2Config,
        vit_config : SiglipVisionConfig,
        vae_config : AutoEncoderParams,
        visual_gen : bool = True,
        visual_und : bool = True,
        latent_patch_size : int = 2,
        max_latent_size : int = 32,
        vit_max_num_patch_per_side : int = 70,
        connector_act : str ="gelu_pytorch_tanh",
        interpolate_pos : bool = False,
        timestep_shift : float = 1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.visual_gen = visual_gen
        self.visual_und = visual_und
        self.llm_config = llm_config
        self.vit_config = vit_config
        self.vae_config = vae_config
        self.latent_patch_size = latent_patch_size
        self.max_latent_size = max_latent_size
        self.vit_max_num_patch_per_side = vit_max_num_patch_per_side
        self.connector_act = connector_act
        self.interpolate_pos = interpolate_pos
        self.timestep_shift = timestep_shift


# ============================================================================
# BagelPseudoPipeline
# ============================================================================

class BagelPseudoPipeline:
    """
    Pseudo-pipeline with flat component ownership.

    All nn.Modules are direct attributes.
    The ``BaseAdapter`` accesses components via ``getattr(self.pipeline, name)``.
    """

    def __init__(
        self,
        config: BagelConfig,                    # BagelConfig
        transformer: Qwen2ForCausalLM,        # Qwen2ForCausalLM
        vit: SiglipVisionModel,                 # SiglipVisionModel
        vae: AutoEncoder,                 # AutoEncoder
        vae2llm: nn.Linear,
        llm2vae: nn.Linear,
        time_embedder: TimestepEmbedder,       # TimestepEmbedder
        latent_pos_embed: PositionEmbedding,    # PositionEmbedding
        connector: MLPconnector,           # MLPconnector
        vit_pos_embed: PositionEmbedding,       # PositionEmbedding
        scheduler: Optional[Any] = None,
    ):
        # ── Direct ownership — single source of truth ──
        self.transformer = transformer
        self.vit = vit
        self.vae = vae
        self.vae2llm = vae2llm
        self.llm2vae = llm2vae
        self.time_embedder = time_embedder
        self.latent_pos_embed = latent_pos_embed
        self.connector = connector
        self.vit_pos_embed = vit_pos_embed
        self.config = config
        self.scheduler = scheduler

        # ── Derived constants from config ──
        self.hidden_size = config.llm_config.hidden_size
        self.use_moe = "Mo" in config.llm_config.layer_module
        self.latent_patch_size = config.latent_patch_size
        self.latent_channel = config.vae_config.z_channels
        self.latent_downsample = config.vae_config.downsample * config.latent_patch_size
        self.max_latent_size = config.max_latent_size
        self.patch_latent_dim = self.latent_patch_size ** 2 * self.latent_channel
        self.vit_patch_size = config.vit_config.patch_size
        self.vit_max_num_patch_per_side = config.vit_max_num_patch_per_side
        self.interpolate_pos = getattr(config, "interpolate_pos", False)

        # ── Backward compatibility alias ──
        self._bagel_config = config

        # Choose position-id function
        if self.interpolate_pos:
            from .data.data_utils import get_flattened_position_ids_interpolate
            self.get_flattened_position_ids = get_flattened_position_ids_interpolate
        else:
            from .data.data_utils import get_flattened_position_ids_extrapolate
            self.get_flattened_position_ids = get_flattened_position_ids_extrapolate

    # ════════════════════════════════════════════════════════════════
    # from_pretrained (Flat Loading)
    # ════════════════════════════════════════════════════════════════

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        vae_path: Optional[str] = None,
        low_cpu_mem_usage: bool = False,
        **kwargs,
    ) -> "BagelPseudoPipeline":
        """
        Load and construct all components individually (no Bagel wrapper).

        Weights are loaded from ``ema.safetensors`` and remapped from
        the original ``language_model.*`` / ``vit_model.*`` prefixes to
        the flat ``transformer.*`` / ``vit.*`` layout.
        """
        from .modeling.bagel import (
            Qwen2Config, Qwen2ForCausalLM,
            SiglipVisionConfig, SiglipVisionModel,
        )
        from .modeling.bagel.modeling_utils import MLPconnector, TimestepEmbedder, PositionEmbedding
        from .modeling.autoencoder import load_ae
        from safetensors.torch import load_file

        # ── Resolve to local directory ──
        model_path = _resolve_model_path(model_path, **kwargs)

        # ── LLM Config ──
        llm_config = Qwen2Config.from_json_file(
            os.path.join(model_path, "llm_config.json")
        )
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = kwargs.get("layer_module", "Qwen2MoTDecoderLayer")

        # ── ViT Config ──
        vit_config = SiglipVisionConfig.from_json_file(
            os.path.join(model_path, "vit_config.json")
        )
        vit_config.rope = kwargs.get("vit_rope", False)
        vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

        # ── VAE ──
        ae_path = vae_path or os.path.join(model_path, "ae.safetensors")
        vae_model, vae_config = load_ae(local_path=ae_path)

        # ── Bagel Config ──
        config = BagelConfig(
            llm_config=llm_config,
            vit_config=vit_config,
            vae_config=vae_config,
            visual_gen=True,
            visual_und=True,
            vit_max_num_patch_per_side=kwargs.get("vit_max_num_patch_per_side", 70),
            connector_act=kwargs.get("connector_act", "gelu_pytorch_tanh"),
            latent_patch_size=kwargs.get("latent_patch_size", 2),
            max_latent_size=kwargs.get("max_latent_size", 64),
        )

        # ── Derived dims ──
        hidden_size = llm_config.hidden_size
        vit_hidden_size = vit_config.hidden_size
        patch_latent_dim = config.latent_patch_size ** 2 * vae_config.z_channels

        # ── Build components independently ──
        if low_cpu_mem_usage:
            from accelerate import init_empty_weights
            ctx = init_empty_weights()
        else:
            from contextlib import nullcontext
            ctx = nullcontext()

        with ctx:
            transformer = Qwen2ForCausalLM(llm_config)
            vit = SiglipVisionModel(vit_config)
            vae2llm = nn.Linear(patch_latent_dim, hidden_size)
            llm2vae = nn.Linear(hidden_size, patch_latent_dim)
            time_embedder = TimestepEmbedder(hidden_size)
            latent_pos_embed = PositionEmbedding(config.max_latent_size, hidden_size)
            connector = MLPconnector(vit_hidden_size, hidden_size, config.connector_act)
            vit_pos_embed = PositionEmbedding(config.vit_max_num_patch_per_side, hidden_size)

        # Convert ViT conv2d → linear
        vit.vision_model.embeddings.convert_conv2d_to_linear(
            vit_config, meta=low_cpu_mem_usage
        )

        # ── Load & remap weights ──
        if not low_cpu_mem_usage:
            ema_path = os.path.join(model_path, "ema.safetensors")
            if os.path.exists(ema_path):
                raw_sd = load_file(ema_path)
                flat_sd = _remap_state_dict(raw_sd)

                # Load into each component by prefix
                component_map = {
                    "transformer.": transformer,
                    "vit.": vit,
                    "vae2llm.": vae2llm,
                    "llm2vae.": llm2vae,
                    "time_embedder.": time_embedder,
                    "latent_pos_embed.": latent_pos_embed,
                    "connector.": connector,
                    "vit_pos_embed.": vit_pos_embed,
                }
                for prefix, module in component_map.items():
                    sub_sd = {
                        k[len(prefix):]: v
                        for k, v in flat_sd.items()
                        if k.startswith(prefix)
                    }
                    if sub_sd:
                        module.load_state_dict(sub_sd, strict=False)

        return cls(
            config=config,
            transformer=transformer,
            vit=vit,
            vae=vae_model,
            vae2llm=vae2llm,
            llm2vae=llm2vae,
            time_embedder=time_embedder,
            latent_pos_embed=latent_pos_embed,
            connector=connector,
            vit_pos_embed=vit_pos_embed,
        )

    # ════════════════════════════════════════════════════════════════
    # Component Enumeration
    # ════════════════════════════════════════════════════════════════

    def named_components(self) -> Dict[str, nn.Module]:
        """Return all nn.Module components (no aliases, no duplicates)."""
        return {
            "transformer":      self.transformer,
            "vit":              self.vit,
            "vae":              self.vae,
            "vae2llm":          self.vae2llm,
            "llm2vae":          self.llm2vae,
            "time_embedder":    self.time_embedder,
            "latent_pos_embed": self.latent_pos_embed,
            "connector":        self.connector,
            "vit_pos_embed":    self.vit_pos_embed,
        }

    # ════════════════════════════════════════════════════════════════
    # DiffusionPipeline-like Interface
    # ════════════════════════════════════════════════════════════════

    def maybe_free_model_hooks(self):
        """No-op: Bagel doesn't use diffusers model hooks."""

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

    def to(self, *args, **kwargs):
        """Move all components to given device/dtype."""
        for comp in self.named_components().values():
            comp.to(*args, **kwargs)
        return self

    # ════════════════════════════════════════════════════════════════
    # Patchify / Unpatchify Utilities
    # ════════════════════════════════════════════════════════════════

    def patchify_latent(self, latent: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """``(C, H, W) → (h*w, patch_dim)`` using ``latent_patch_size``."""
        p = self.latent_patch_size
        ch = self.latent_channel
        latent = latent[:, :h * p, :w * p].reshape(ch, h, p, w, p)
        return torch.einsum("chpwq->hwpqc", latent).reshape(-1, self.patch_latent_dim)

    def unpatchify_latent(self, tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """``(h*w, patch_dim) → (C, h*p, w*p)``."""
        p = self.latent_patch_size
        ch = self.latent_channel
        return (
            tokens.reshape(h, w, p, p, ch)
            .permute(4, 0, 2, 1, 3)
            .reshape(ch, h * p, w * p)
        )

    # ════════════════════════════════════════════════════════════════
    # Prepare Methods (moved from Bagel class, inference-only)
    # ════════════════════════════════════════════════════════════════

    def prepare_prompts(self, curr_kvlens, curr_rope, prompts, tokenizer, new_token_ids):
        """Tokenize prompts and build packed sequence inputs for KV-cache update."""
        packed_text_ids = []
        packed_text_position_ids = []
        text_token_lens = []
        packed_text_indexes = []
        packed_key_value_indexes = []

        curr = 0
        newlens, new_rope_out = [], []
        for prompt, curr_kvlen, curr_position_id in zip(prompts, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            text_ids = tokenizer.encode(prompt)
            text_ids = [new_token_ids['bos_token_id']] + text_ids + [new_token_ids['eos_token_id']]
            text_token_lens.append(len(text_ids))
            packed_text_ids.extend(text_ids)
            packed_text_position_ids.extend(range(curr_position_id, curr_position_id + len(text_ids)))
            packed_text_indexes.extend(range(curr, curr + len(text_ids)))
            newlens.append(curr_kvlen + len(text_ids))
            new_rope_out.append(curr_position_id + len(text_ids))
            curr += len(text_ids)

        generation_input = {
            "text_token_lens": torch.tensor(text_token_lens, dtype=torch.int),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_position_ids": torch.tensor(packed_text_position_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }
        return generation_input, newlens, new_rope_out

    def prepare_vit_images(self, curr_kvlens, curr_rope, images, transforms, new_token_ids):
        """Prepare ViT image tokens for KV-cache update."""
        from .data.data_utils import patchify

        packed_vit_token_indexes = []
        vit_token_seqlens, packed_vit_tokens, packed_vit_position_ids = [], [], []
        packed_text_ids, packed_text_indexes = [], []
        packed_seqlens, packed_position_ids, packed_indexes = [], [], []
        packed_key_value_indexes = []

        _curr = curr = 0
        newlens, new_rope_out = [], []
        for image, curr_kvlen, curr_position_id in zip(images, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            image_tensor = transforms(image) if not isinstance(image, torch.Tensor) else image
            vit_position_ids = self.get_flattened_position_ids(
                image_tensor.size(1), image_tensor.size(2),
                self.vit_patch_size,
                max_num_patches_per_side=self.vit_max_num_patch_per_side,
            )
            vit_tokens = patchify(image_tensor, self.vit_patch_size)
            packed_vit_tokens.append(vit_tokens)
            num_img_tokens = vit_tokens.shape[0]
            packed_vit_position_ids.append(vit_position_ids)
            vit_token_seqlens.append(num_img_tokens)
            packed_vit_token_indexes.extend(range(_curr, _curr + num_img_tokens))
            packed_indexes.extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens
            _curr += num_img_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            packed_position_ids.extend([curr_position_id] * (num_img_tokens + 2))
            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(curr_kvlen + num_img_tokens + 2)
            new_rope_out.append(curr_position_id + 1)

        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "vit_token_seqlens": torch.tensor(vit_token_seqlens, dtype=torch.int),
            "packed_vit_tokens": torch.cat(packed_vit_tokens, dim=0),
            "packed_vit_position_ids": torch.cat(packed_vit_position_ids, dim=0),
            "packed_vit_token_indexes": torch.tensor(packed_vit_token_indexes, dtype=torch.long),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }
        return generation_input, newlens, new_rope_out

    def prepare_vae_images(self, curr_kvlens, curr_rope, images, transforms, new_token_ids, timestep=0):
        patchified_vae_latent_shapes, packed_vae_position_ids = list(), list()
        packed_vae_token_indexes = list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        _curr = curr = 0
        vae_image_tensors = list()
        newlens, new_rope = list(), list()
        for image, curr_kvlen, curr_position_id in zip(images, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            image_tensor = transforms(image) if not isinstance(image, torch.Tensor) else image
            vae_image_tensors.append(image_tensor)
            vae_posiiton_ids = self.get_flattened_position_ids(
                image_tensor.size(1), image_tensor.size(2),
                self.latent_downsample, 
                max_num_patches_per_side=self.max_latent_size
            )
            packed_vae_position_ids.append(vae_posiiton_ids)
            H, W = image_tensor.shape[1:]
            h = H // self.latent_downsample
            w = W // self.latent_downsample
            patchified_vae_latent_shapes.append((h, w))

            num_img_tokens = w * h
            packed_vae_token_indexes.extend(range(_curr, _curr + num_img_tokens))
            packed_indexes.extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens
            _curr += num_img_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            packed_position_ids.extend([curr_position_id] * (num_img_tokens + 2))
            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(curr_kvlen + num_img_tokens + 2)
            new_rope.append(curr_position_id + 1)

        image_sizes = [item.shape for item in vae_image_tensors]
        max_image_size = [max(item) for item in list(zip(*image_sizes))]
        padded_images = torch.zeros(size=(len(vae_image_tensors), *max_image_size))
        for i, image_tensor in enumerate(vae_image_tensors):
            padded_images[i, :, :image_tensor.shape[1], :image_tensor.shape[2]] = image_tensor

        generation_input = {
            "padded_images": padded_images,
            "patchified_vae_latent_shapes": patchified_vae_latent_shapes,
            "packed_vae_position_ids": torch.cat(packed_vae_position_ids, dim=0),
            "packed_timesteps": torch.tensor([timestep]),
            "packed_vae_token_indexes": torch.tensor(packed_vae_token_indexes, dtype=torch.long),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    def prepare_vae_latent(self, curr_kvlens, curr_rope, image_sizes, new_token_ids, device=None):
        device = device or torch.device("cpu")
        packed_text_ids, packed_text_indexes = list(), list()
        packed_vae_position_ids, packed_vae_token_indexes, packed_init_noises = list(), list(), list()
        packed_position_ids, packed_seqlens, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        query_curr = curr = 0
        for (H, W), curr_kvlen, curr_position_id in zip(image_sizes, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(query_curr)
            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            vae_posiiton_ids = self.get_flattened_position_ids(
                H, W,
                self.latent_downsample, 
                max_num_patches_per_side=self.max_latent_size
            )
            packed_vae_position_ids.append(vae_posiiton_ids)

            h, w = H // self.latent_downsample, W // self.latent_downsample
            num_image_tokens = h * w
            packed_init_noises.append(
                torch.randn(num_image_tokens, self.latent_channel * self.latent_patch_size ** 2)
            )
            packed_vae_token_indexes.extend(range(query_curr, query_curr + num_image_tokens))
            packed_indexes.extend(range(curr, curr + num_image_tokens))
            curr += num_image_tokens
            query_curr += num_image_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(query_curr)
            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            packed_position_ids.extend([curr_position_id] * (num_image_tokens + 2))
            packed_seqlens.append(num_image_tokens + 2)

        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long, device=device),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long, device=device),
            "packed_init_noises": torch.cat(packed_init_noises, dim=0).to(device=device),
            "packed_vae_position_ids": torch.cat(packed_vae_position_ids, dim=0).to(device=device),
            "packed_vae_token_indexes": torch.tensor(packed_vae_token_indexes, dtype=torch.long, device=device),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int, device=device),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long, device=device),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int, device=device),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long, device=device),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long, device=device),
        }

        return generation_input
    
    def prepare_vae_latent_cfg(self, curr_kvlens, curr_rope, image_sizes, device=None):
        device = device or self.device
        packed_position_ids = []
        packed_indexes = []
        packed_key_value_indexes = []

        query_curr = curr = 0
        for (H, W), curr_kvlen, curr_position_id in zip(image_sizes, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            h, w = H // self.latent_downsample, W // self.latent_downsample
            num_vae_tokens = h * w

            # start_of_image token
            packed_indexes.append(curr)
            curr += 1

            # VAE tokens
            packed_indexes.extend(range(curr, curr + num_vae_tokens))
            curr += num_vae_tokens

            # end_of_image token
            packed_indexes.append(curr)
            curr += 1

            # Position IDs for ALL tokens (vae + 2 text markers)
            packed_position_ids.extend([curr_position_id] * (num_vae_tokens + 2))

        return {
            "cfg_packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long, device=device),
            "cfg_packed_query_indexes": torch.tensor(packed_indexes, dtype=torch.long, device=device),
            "cfg_key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int, device=device),
            "cfg_packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long, device=device),
        }

    # ════════════════════════════════════════════════════════════════
    # Forward Cache Update Methods
    # ════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def forward_cache_update_text(
        self,
        past_key_values,
        packed_text_ids: torch.IntTensor,
        packed_text_position_ids: torch.LongTensor,
        text_token_lens: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
    ):
        """Update KV-cache with text tokens."""
        packed_text_embedding = self.transformer.model.embed_tokens(packed_text_ids)

        extra = {"mode": "und"} if self.use_moe else {}

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

    @torch.no_grad()
    def forward_cache_update_vit(
        self,
        past_key_values,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_vit_tokens: torch.Tensor,
        packed_vit_token_indexes: torch.LongTensor,
        packed_vit_position_ids: torch.LongTensor,
        vit_token_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
    ):
        """Update KV-cache with ViT image tokens."""
        packed_text_embedding = self.transformer.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0))
        cu_seqlens = cu_seqlens.to(torch.int32)
        max_seqlen = torch.max(vit_token_seqlens).item()
        packed_vit_token_embed = self.vit(
            packed_pixel_values=packed_vit_tokens,
            packed_flattened_position_ids=packed_vit_position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        packed_vit_token_embed = self.connector(packed_vit_token_embed)
        pos_emb = self.vit_pos_embed(packed_vit_position_ids)
        packed_vit_token_embed = packed_vit_token_embed + pos_emb
        if packed_vit_token_embed.dtype != packed_sequence.dtype:
            packed_vit_token_embed = packed_vit_token_embed.to(packed_sequence.dtype)
        packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed

        extra = {"mode": "und"} if self.use_moe else {}

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

    @torch.no_grad()
    def forward_cache_update_vae(
        self,
        past_key_values,
        padded_images: torch.Tensor,
        patchified_vae_latent_shapes: List,
        packed_vae_position_ids: torch.LongTensor,
        packed_timesteps: torch.Tensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_key_value_indexes: torch.Tensor,
    ):
        """Update KV-cache with VAE-encoded image tokens."""
        packed_text_embedding = self.transformer.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        padded_latent = self.vae.encode(padded_images)

        p = self.latent_patch_size
        packed_latent = []
        for latent, (h, w) in zip(padded_latent, patchified_vae_latent_shapes):
            packed_latent.append(self.patchify_latent(latent, h, w))
        packed_latent = torch.cat(packed_latent, dim=0)

        packed_pos_embed = self.latent_pos_embed(packed_vae_position_ids)
        packed_timestep_embeds = self.time_embedder(packed_timesteps)
        packed_latent = self.vae2llm(packed_latent) + packed_timestep_embeds + packed_pos_embed
        if packed_latent.dtype != packed_sequence.dtype:
            packed_latent = packed_latent.to(packed_sequence.dtype)
        packed_sequence[packed_vae_token_indexes] = packed_latent

        extra = {}
        if self.use_moe:
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