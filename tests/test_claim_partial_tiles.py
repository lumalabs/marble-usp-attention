# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""``benchmarks/claim_partial_tiles.py`` tile counts against the dense mask."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from usp_attention.layouts import build_layout
from usp_attention.reference import dense_mask_rows
from usp_attention.ring.group_order import vision_group_order
from usp_attention.ring.packed_layout import build_packed_layout

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from claim_partial_tiles import tile_counts  # noqa: E402


@pytest.mark.parametrize("name", ["vlm_group", "video_group", "mixed"])
@pytest.mark.parametrize("reorder", [False, True])
def test_tile_counts_match_dense_mask(name: str, reorder: bool) -> None:
    n = 8192
    docs, splits, modes = build_layout(name, n, seed=1)
    perm = vision_group_order(docs, splits, modes) if reorder else None
    if reorder and perm is None:
        pytest.skip("no vision groups")
    mask = dense_mask_rows(build_packed_layout(docs, splits, modes), 0, n)
    if perm is not None:
        mask = mask[perm][:, perm]
    tiles = mask.reshape(n // 128, 128, n // 128, 128).sum((1, 3))
    got = tile_counts(docs, splits, modes, None if perm is None else perm.numpy())
    assert got.full == int((tiles == 128 * 128).sum())
    assert got.partial == int(((tiles > 0) & (tiles < 128 * 128)).sum())
    assert got.pairs == int(tiles.sum())
