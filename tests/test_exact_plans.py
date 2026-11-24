# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""With ``exact_dense_ring`` no plan merges bf16 partials across ring steps."""

from __future__ import annotations

import pytest
import torch

import usp_attention.usp as usp_module
from usp_attention.distributed import USPGroups
from usp_attention.layouts import build_layout
from usp_attention.usp import USPConfig, build_usp_plan


def _plan(name: str, ring: int, ulysses: int, exact: bool):
    docs, splits, modes = build_layout(name, 65536, seed=0)[:3]
    groups = USPGroups(
        ring_size=ring,
        ulysses_size=ulysses,
        ring_rank=0,
        ulysses_rank=0,
        cp_rank=0,
        ring_group=None,
        ulysses_group=None,
        cp_group=None,
    )
    # dense_ring / interval_masks on: the Blackwell defaults, without a GPU.
    config = USPConfig(
        dense_ring=True,
        interval_masks=True,
        combine_mode="fp32",
        exact_dense_ring=exact,
    )
    return build_usp_plan(
        document_lens=docs,
        split_lens=splits,
        attn_modes=modes,
        groups=groups,
        config=config,
        device=torch.device("cpu"),
    )


@pytest.mark.parametrize("name", ["mixed", "vlm", "video", "docs"])
@pytest.mark.parametrize(("ring", "ulysses"), [(2, 4), (4, 2), (1, 8)])
def test_exact_plans_avoid_bf16_merges(
    monkeypatch: pytest.MonkeyPatch, name: str, ring: int, ulysses: int
) -> None:
    def bf16_dense_ring(*args, **kwargs):
        raise AssertionError("exact plan took the bf16-merging dense ring")

    monkeypatch.setattr(usp_module, "build_dense_ring_plan", bf16_dense_ring)
    try:
        plan = _plan(name, ring, ulysses, exact=True)
    except RuntimeError as err:
        # Short image spans fall through to the block-sparse ring, whose
        # masks need flash_attn.cute; reaching it is the point.
        if "flash_attn.cute" not in str(err):
            raise
        return
    assert plan.dense_ring is None
    assert plan.interval_plan is None


def test_default_short_spans_keep_the_dense_ring() -> None:
    """Without the flag, image crops at ring 2 still take the faster dense ring."""
    assert _plan("vlm", 2, 4, exact=False).dense_ring is not None
