# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Shared type aliases."""

from __future__ import annotations

from typing import Literal

RingCombineMode = Literal["auto", "flash", "python", "online", "fp32"]
"""How per-step ring attention partials are merged.

``"online"`` threads fp32 running ``out`` / ``max`` / ``sum`` state through the
online FA4 kernel across ring steps and finalizes once on the last step.
``"flash"`` runs a standard FA4 forward per step and merges each partial with
the fused ``flash_attn_combine`` kernel. ``"python"`` does the same merge with a
pure-PyTorch log-sum-exp rescale. ``"fp32"`` merges like ``"python"``
but keeps every step's partial output in fp32: each step runs FA4's split-KV
forward and takes its fp32 partials before they are rounded (SM100-class
GPUs, where the online kernel is not available; same accuracy as
``"online"``). ``"auto"`` picks ``"flash"`` when the fused
combine kernel is importable and ``"python"`` otherwise; it never selects
``"online"``.
"""

ONLINE_MAX_HEAD_DIM = 256
"""Largest head_dim the SM90 online kernel runs at (FA4's SM90 limit).

Up to 128 the fp32 output state is staged through shared memory with TMA;
above it the state moves directly between global memory and registers, since
an fp32 tile would not fit next to Q / K / V in Hopper's 227 KiB. "Above 128"
means the larger of the Q/K and V head dims: the K tiles grow with head_dim
too, so (192, 128) would need 279,552 B with the staged tile.
"""
