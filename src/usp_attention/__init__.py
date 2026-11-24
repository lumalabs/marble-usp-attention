# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Ulysses x ring context parallelism on FlashAttention-4 with online ring combine."""

from usp_attention.distributed import USPGroups, init_distributed, seq_all_to_all
from usp_attention.types import RingCombineMode
from usp_attention.usp import (
    FA4_BLOCK_SIZE,
    Transport,
    USPConfig,
    USPPlan,
    build_sliding_window_plan,
    build_usp_plan,
    pad_packed_layout,
    required_seq_multiple,
    usp_attention,
)

__all__ = [
    "FA4_BLOCK_SIZE",
    "RingCombineMode",
    "Transport",
    "USPConfig",
    "USPGroups",
    "USPPlan",
    "build_sliding_window_plan",
    "build_usp_plan",
    "init_distributed",
    "pad_packed_layout",
    "required_seq_multiple",
    "seq_all_to_all",
    "usp_attention",
]
