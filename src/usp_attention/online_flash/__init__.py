# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Online Flash Attention 4 (SM90).

Exposes the online-accumulation variant of FA4's SM90 forward kernel and a
small Python interface that mirrors `flash_attn.cute.interface.flash_attn_func`
but threads `out` / `max_` / `sum_` across chunked KV calls.
"""

from usp_attention.online_flash.interface import (
    offline_flash_attn_func,
    online_flash_attn_chunks,
    online_flash_attn_func,
)
from usp_attention.online_flash.online_flash_fwd_sm90 import (
    OnlineFlashAttentionForwardSm90,
    Softmax2TheSequel,
)

__all__ = [
    "OnlineFlashAttentionForwardSm90",
    "Softmax2TheSequel",
    "offline_flash_attn_func",
    "online_flash_attn_chunks",
    "online_flash_attn_func",
]
