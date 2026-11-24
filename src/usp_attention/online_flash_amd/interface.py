# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Online Flash Attention interface for AMD CDNA3 (MI300X / gfx942).

The AMD counterpart to ``usp_attention.online_flash`` (which targets NVIDIA SM90 via
``flash_attn.cute``). Same *shape* of API — a per-chunk ``online_flash_attn_func`` that threads
running softmax state across chunked-KV calls, plus an ``online_flash_attn_chunks`` convenience —
but it drives a standalone HIP kernel instead of CUTLASS, so it is self-contained (no
``flash_attn.cute`` dependency).

Differences from the SM90 interface (by design of the AMD kernel):

- The running state is a single opaque per-thread ``state`` tensor (the fp32 ``o_acc`` numerator
  plus per-lane ``row_max`` / ``row_sum``), NOT separate ``max_`` / ``sum_`` tensors.
- ``softmax_scale`` is fixed to ``1/sqrt(head_dim)`` (the kernel folds it into ``exp2``).
- Packed causal/multimodal masks use ``sparse_plan`` on online calls.
- Sliding windows and LSE output are not exposed yet.
- ``head_dim == 128``; bf16 in/out; unpacked GQA/MQA is supported when ``Hq >= Hkv`` and
  ``Hq % Hkv == 0``, using contiguous query-head groups. Packed-GQA optimization is unsupported; BSHD ``(batch, seqlen, num_head, head_dim)`` or BHSD.

Attribution: the underlying forward kernel is adapted from MoonMath's open-source amd-kernels
(https://github.com/moonmath-ai/amd-kernels, MIT); the RingAttention online path is the addition.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch

from ._ext import _C
from .sparse import SparsePlan

__all__ = [
    "forward",
    "offline_flash_attn_func",
    "online_flash_attn_func",
    "online_flash_attn_chunks",
]

_ROUND_MODES = ("rtne", "rtna", "rtz")


def _check_scale(softmax_scale: Optional[float], head_dim: int) -> None:
    if softmax_scale is None:
        return
    default = 1.0 / math.sqrt(head_dim)
    if not math.isclose(softmax_scale, default, rel_tol=1e-6, abs_tol=1e-9):
        raise NotImplementedError(
            "online_flash_4_amd only supports softmax_scale == 1/sqrt(head_dim) "
            f"(got {softmax_scale}, expected {default})."
        )


def forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    round_mode: str = "rtne",
    layout: str = "bshd",
) -> torch.Tensor:
    """Full (non-online) forward attention: ``O = softmax(QKᵀ / √d)·V``.

    bf16 in/out, ``head_dim == 128``. ``layout`` is ``"bshd"`` (batch, seqlen, num_head, head_dim)
    or ``"bhsd"``. ``round_mode`` selects bf16 accumulator packing (``"rtne"`` / ``"rtna"`` /
    ``"rtz"``).

    K and V may have fewer heads than Q for unpacked GQA/MQA when ``Hq >= Hkv`` and
    ``Hq % Hkv == 0``. Contiguous query groups map with
    ``kv_head = query_head / (Hq / Hkv)``.

    Also exported as :data:`offline_flash_attn_func` — the online path's ``num_chunks == 1``
    degenerate case reduces to this full forward, so the two names are aliases.
    """
    if round_mode not in _ROUND_MODES:
        raise ValueError(
            f"round_mode must be one of {_ROUND_MODES}; got {round_mode!r}"
        )
    _check_scale(softmax_scale, q.size(-1))
    return _C.forward(q, k, v, out, round_mode, layout)


# The "offline"/reference path is just the full forward kernel (num_chunks == 1).
offline_flash_attn_func = forward


def online_flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    state: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    is_online_first: bool = False,
    is_online_last: bool = False,
    round_mode: str = "rtne",
    layout: str = "bshd",
    sparse_plan: SparsePlan | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Online (RingAttention) forward attention — one call per KV chunk.

    Folds ``k`` / ``v`` (this chunk) into a running softmax state that persists in HBM between
    calls, and returns ``(out, state)``.

    On the first call, pass ``is_online_first=True`` and leave ``out`` / ``state`` as ``None`` —
    the state is allocated internally and returned. On subsequent calls, thread the returned
    tensors back in as ``out=...`` / ``state=...``. On the last call, pass ``is_online_last=True``
    to finalize (cross-lane sum + normalize); ``out`` then holds the final attention output.
    ``out`` is only meaningful once ``is_online_last`` — on earlier calls it is untouched.

    State is allocated and indexed per query head, not per KV head. ``sparse_plan``
    supplies a precomputed 384x64 BCSR plan and original-position metadata for
    this query/KV window. Partial tiles use cached exact bitmaps. Fully masked rows produce zero; empty intermediate
    steps preserve state. The plan is shared across batch entries and heads.

    With no sparse plan, computes dense attention over concatenated KV up to
    floating-point rounding. A single first-and-last call also supports sparse attention.
    """
    if round_mode not in _ROUND_MODES:
        raise ValueError(
            f"round_mode must be one of {_ROUND_MODES}; got {round_mode!r}"
        )
    _check_scale(softmax_scale, q.size(-1))
    return _C.forward_online_chunk(
        q,
        k,
        v,
        bool(is_online_first),
        bool(is_online_last),
        state,
        out,
        round_mode,
        layout,
        sparse_plan.tensors if sparse_plan is not None else [],
    )


def online_flash_attn_chunks(
    q: torch.Tensor,
    k_chunks: Sequence[torch.Tensor],
    v_chunks: Sequence[torch.Tensor],
    *,
    softmax_scale: Optional[float] = None,
    round_mode: str = "rtne",
    layout: str = "bshd",
) -> torch.Tensor:
    """Convenience: run online flash attention over a pre-split list of KV chunks.

    Walks the chunks in order, threading ``out`` / ``state`` between calls and marking the
    first/last chunks. Returns the final ``out``. Equivalent to :func:`forward` on the
    concatenated KV.
    """
    k_chunks = list(k_chunks)
    v_chunks = list(v_chunks)
    if len(k_chunks) != len(v_chunks) or len(k_chunks) < 1:
        raise ValueError(
            "k_chunks and v_chunks must be non-empty lists of equal length"
        )
    n = len(k_chunks)

    out: Optional[torch.Tensor] = None
    state: Optional[torch.Tensor] = None
    for i, (k_c, v_c) in enumerate(zip(k_chunks, v_chunks)):
        out, state = online_flash_attn_func(
            q,
            k_c,
            v_c,
            out=out,
            state=state,
            softmax_scale=softmax_scale,
            is_online_first=(i == 0),
            is_online_last=(i == n - 1),
            round_mode=round_mode,
            layout=layout,
        )
    return out
