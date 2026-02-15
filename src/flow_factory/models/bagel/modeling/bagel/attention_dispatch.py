# Copyright 2025 Flow-Factory Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified attention dispatch for variable-length packed sequences.

This module provides a single ``varlen_attention`` entry-point that routes to
one of several backends (flash-attn-3, flash-attn-2, SDPA, eager) based on a
global or per-call ``backend`` parameter.  It handles the unique requirement of
Bagel's NaViT-style packed sequences where inputs are
``(total_tokens, num_heads, head_dim)`` with cumulative sequence-length tensors
(``cu_seqlens``).

Supported backends:

- **auto**: FA3 → FA2 → SDPA (first available)
- **flash3**: Flash Attention 3 (Hopper sm90 kernels, H100/H200 only)
- **flash**: Flash Attention 2 (Ampere+)
- **sdpa**: PyTorch ``F.scaled_dot_product_attention`` with per-sequence split
- **eager**: Manual matmul + softmax (debug / reference)

Usage::

    from .attention_dispatch import varlen_attention, set_attn_backend, AttnBackend

    # Global default (set once at init)
    set_attn_backend(AttnBackend.FLASH3)

    # Per-call override
    out = varlen_attention(q, k, v, cu_seqlens_q, cu_seqlens_k,
                           max_seqlen_q, max_seqlen_k, causal=True,
                           backend=AttnBackend.SDPA)
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

import torch
import torch.nn.functional as F
from transformers.utils.import_utils import is_flash_attn_2_available, is_flash_attn_3_available

logger = logging.getLogger(__name__)


# ── Lazy-cached function handles ─────────────────────────────────────────────
# We use transformers' availability guards for resolution but defer the actual
# import until first call, avoiding import errors at module load time.

_fa3_varlen_func = None
_fa2_varlen_func = None


def _get_fa3_varlen_func():
    """Lazy-import FA3's ``flash_attn_varlen_func`` on first use."""
    global _fa3_varlen_func
    if _fa3_varlen_func is None:
        from flash_attn_interface import flash_attn_varlen_func
        _fa3_varlen_func = flash_attn_varlen_func
    return _fa3_varlen_func


def _get_fa2_varlen_func():
    """Lazy-import FA2's ``flash_attn_varlen_func`` on first use."""
    global _fa2_varlen_func
    if _fa2_varlen_func is None:
        from flash_attn import flash_attn_varlen_func
        _fa2_varlen_func = flash_attn_varlen_func
    return _fa2_varlen_func


# ── Backend enum ─────────────────────────────────────────────────────────────

class AttnBackend(str, Enum):
    """Supported attention backends for varlen packed sequences."""
    AUTO   = "auto"     # fa3 → fa2 → sdpa (first available)
    FLASH3 = "flash3"   # Flash Attention 3 (Hopper sm90, H100/H200)
    FLASH  = "flash"    # Flash Attention 2 (Ampere+)
    SDPA   = "sdpa"     # torch SDPA with per-sequence splitting
    EAGER  = "eager"    # manual matmul + softmax (debug / reference)


_GLOBAL_BACKEND: AttnBackend = AttnBackend.AUTO


def set_attn_backend(backend: str | AttnBackend) -> None:
    """Set the global default attention backend.

    Args:
        backend: One of ``"auto"``, ``"flash3"``, ``"flash"``, ``"sdpa"``,
                 ``"eager"``.
    """
    global _GLOBAL_BACKEND
    _GLOBAL_BACKEND = AttnBackend(backend)
    logger.info("Global varlen attention backend set to: %s", _GLOBAL_BACKEND.value)


def get_attn_backend() -> AttnBackend:
    """Return the current global attention backend."""
    return _GLOBAL_BACKEND


# ── Resolution logic ─────────────────────────────────────────────────────────

_FALLBACK_CHAIN = [
    (AttnBackend.FLASH3, is_flash_attn_3_available),
    (AttnBackend.FLASH,  is_flash_attn_2_available),
    (AttnBackend.SDPA,   lambda: True),
]


def _resolve_backend(backend: Optional[AttnBackend]) -> AttnBackend:
    """Resolve ``None`` / ``AUTO`` to a concrete backend, with graceful fallback."""
    if backend is None:
        backend = _GLOBAL_BACKEND

    if backend == AttnBackend.AUTO:
        for candidate, available_fn in _FALLBACK_CHAIN:
            if available_fn():
                return candidate
        return AttnBackend.EAGER  # unreachable, but safe

    # Explicit backend — verify availability and cascade
    if backend == AttnBackend.FLASH3 and not is_flash_attn_3_available():
        _warn_once(
            "flash3 (Hopper) requested but not available; trying flash-attn-2."
        )
        backend = AttnBackend.FLASH

    if backend == AttnBackend.FLASH and not is_flash_attn_2_available():
        _warn_once("flash (FA2) requested but not available; falling back to SDPA.")
        backend = AttnBackend.SDPA

    return backend


# ── Backend implementations ──────────────────────────────────────────────────

def _flash3_varlen(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int, max_seqlen_k: int,
    causal: bool = False, dropout_p: float = 0.0,
) -> torch.Tensor:
    """Flash Attention 3 varlen backend (Hopper sm90 kernels).

    FA3 provides ~1.5-2x throughput over FA2 on H100/H200.
    Note: FA3 does **not** support ``dropout_p``; accepted for API uniformity.
    Expects ``(total_tokens, num_heads, head_dim)`` layout.
    """
    fn = _get_fa3_varlen_func()
    dtype = q.dtype if q.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
    out = fn(
        q.to(dtype), k.to(dtype), v.to(dtype),
        cu_seqlens_q.to(torch.int32), cu_seqlens_k.to(torch.int32),
        max_seqlen_q, max_seqlen_k,
        causal=causal,
    )
    # FA3 returns (out, softmax_lse, *rest)
    return out[0] if isinstance(out, tuple) else out


def _flash2_varlen(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int, max_seqlen_k: int,
    causal: bool = False, dropout_p: float = 0.0,
) -> torch.Tensor:
    """Flash Attention 2 varlen backend (Ampere / Ada / Hopper).

    Expects ``(total_tokens, num_heads, head_dim)`` layout.
    """
    fn = _get_fa2_varlen_func()
    return fn(
        q=q.to(torch.bfloat16), k=k.to(torch.bfloat16), v=v.to(torch.bfloat16),
        cu_seqlens_q=cu_seqlens_q.to(torch.int32),
        cu_seqlens_k=cu_seqlens_k.to(torch.int32),
        max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
        causal=causal, dropout_p=dropout_p,
    )


def _sdpa_varlen(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int, max_seqlen_k: int,
    causal: bool = False, dropout_p: float = 0.0,
) -> torch.Tensor:
    """PyTorch SDPA fallback for variable-length packed sequences.

    Splits packed ``(total_tokens, H, D)`` along *cu_seqlens*, runs
    ``F.scaled_dot_product_attention`` per sub-sequence, and re-packs.
    """
    num_seqs = cu_seqlens_q.numel() - 1
    assert num_seqs == cu_seqlens_k.numel() - 1

    chunks = []
    for i in range(num_seqs):
        sq, eq = cu_seqlens_q[i].item(), cu_seqlens_q[i + 1].item()
        sk, ek = cu_seqlens_k[i].item(), cu_seqlens_k[i + 1].item()
        # (seq, H, D) → (1, H, seq, D)
        qi = q[sq:eq].transpose(0, 1).unsqueeze(0)
        ki = k[sk:ek].transpose(0, 1).unsqueeze(0)
        vi = v[sk:ek].transpose(0, 1).unsqueeze(0)
        oi = F.scaled_dot_product_attention(qi, ki, vi, is_causal=causal, dropout_p=dropout_p)
        chunks.append(oi.squeeze(0).transpose(0, 1))
    return torch.cat(chunks, dim=0)


def _eager_varlen(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int, max_seqlen_k: int,
    causal: bool = False, dropout_p: float = 0.0,
) -> torch.Tensor:
    """Manual matmul + softmax reference implementation (slow, for debugging)."""
    num_seqs = cu_seqlens_q.numel() - 1
    scale = q.shape[-1] ** -0.5

    chunks = []
    for i in range(num_seqs):
        sq, eq = cu_seqlens_q[i].item(), cu_seqlens_q[i + 1].item()
        sk, ek = cu_seqlens_k[i].item(), cu_seqlens_k[i + 1].item()
        qi = q[sq:eq].transpose(0, 1).float()
        ki = k[sk:ek].transpose(0, 1).float()
        vi = v[sk:ek].transpose(0, 1).float()

        scores = torch.matmul(qi, ki.transpose(-1, -2)) * scale
        if causal:
            lq, lk = scores.shape[-2], scores.shape[-1]
            scores = scores + torch.triu(
                torch.full((lq, lk), float("-inf"), device=scores.device),
                diagonal=lk - lq + 1,
            )
        attn = torch.softmax(scores, dim=-1)
        if dropout_p > 0.0 and q.requires_grad:
            attn = F.dropout(attn, p=dropout_p)
        chunks.append(torch.matmul(attn, vi).transpose(0, 1).to(q.dtype))
    return torch.cat(chunks, dim=0)


_BACKEND_FN = {
    AttnBackend.FLASH3: _flash3_varlen,
    AttnBackend.FLASH:  _flash2_varlen,
    AttnBackend.SDPA:   _sdpa_varlen,
    AttnBackend.EAGER:  _eager_varlen,
}


# ── Public API ───────────────────────────────────────────────────────────────

def varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool = False,
    dropout_p: float = 0.0,
    backend: Optional[AttnBackend | str] = None,
) -> torch.Tensor:
    """Compute variable-length packed attention with selectable backend.

    All inputs follow the flash-attn varlen layout:

    - ``q, k, v``: ``(total_tokens, num_heads, head_dim)``
    - ``cu_seqlens_q, cu_seqlens_k``: ``(batch + 1,)`` cumulative lengths
    - ``max_seqlen_q, max_seqlen_k``: scalar max lengths

    Args:
        q, k, v: Query, key, value tensors.
        cu_seqlens_q: Cumulative query sequence lengths.
        cu_seqlens_k: Cumulative key sequence lengths.
        max_seqlen_q: Maximum query sequence length.
        max_seqlen_k: Maximum key sequence length.
        causal: Whether to apply causal masking.
        dropout_p: Dropout probability (training only).
        backend: Override backend (``"auto"``, ``"flash3"``, ``"flash"``,
                 ``"sdpa"``, ``"eager"``). ``None`` uses the global default.

    Returns:
        Attention output ``(total_tokens, num_heads, head_dim)``.
    """
    if isinstance(backend, str):
        backend = AttnBackend(backend)
    resolved = _resolve_backend(backend)
    return _BACKEND_FN[resolved](
        q, k, v, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        causal=causal, dropout_p=dropout_p,
    )


# ── Utility ──────────────────────────────────────────────────────────────────

_warned_messages: set[str] = set()

def _warn_once(msg: str) -> None:
    """Log a warning only the first time it is encountered."""
    if msg not in _warned_messages:
        _warned_messages.add(msg)
        logger.warning(msg)