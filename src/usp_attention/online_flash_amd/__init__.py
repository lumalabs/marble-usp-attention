# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Online Flash Attention 4 (AMD CDNA3 / MI300X / gfx942).

The AMD counterpart to ``usp_attention.online_flash`` (NVIDIA SM90). Exposes a full
forward attention kernel and a RingAttention *online* path — one call per KV chunk, threading the
running softmax state through HBM (``is_online_first`` / ``is_online_last``) — mirroring the SM90
interface's shape while driving a standalone HIP kernel.

The forward kernel is adapted from MoonMath's open-source amd-kernels
(https://github.com/moonmath-ai/amd-kernels, MIT; see https://moonmath.ai); the online path is
the addition here.

Build the ``_C`` extension once with a ROCm PyTorch + hipcc:

    cd src/usp_attention/online_flash_amd && python setup.py build_ext --inplace
"""

from usp_attention.online_flash_amd.interface import (
    forward,
    offline_flash_attn_func,
    online_flash_attn_chunks,
    online_flash_attn_func,
)
from usp_attention.online_flash_amd.sparse import SparsePlan, build_sparse_plan

__all__ = [
    "SparsePlan",
    "build_sparse_plan",
    "forward",
    "offline_flash_attn_func",
    "online_flash_attn_chunks",
    "online_flash_attn_func",
]
