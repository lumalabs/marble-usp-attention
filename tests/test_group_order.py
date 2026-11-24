# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Tile-friendly token order for vision-group layouts."""

from __future__ import annotations

import pytest
import torch

from usp_attention.layouts import build_layout
from usp_attention.reference import dense_mask_rows
from usp_attention.ring.group_order import vision_group_order
from usp_attention.ring.packed_layout import build_packed_layout


def test_no_groups_no_permutation() -> None:
    assert vision_group_order(*build_layout("vlm", 8192, seed=1)) is None


@pytest.mark.parametrize("name", ["vlm_group", "video_group"])
def test_reordered_keys_see_contiguous_queries(name: str) -> None:
    docs, splits, modes = build_layout(name, 8192, seed=2)
    perm = vision_group_order(docs, splits, modes)
    assert perm is not None
    assert torch.equal(torch.sort(perm).values, torch.arange(8192))
    layout = build_packed_layout(docs, splits, modes)
    vis = dense_mask_rows(layout, 0, 8192)[perm][:, perm]
    # Every key column: the visible query rows form one run.
    starts = (vis[1:] & ~vis[:-1]).sum(dim=0) + vis[0].int()
    assert int(starts.max()) == 1


def test_documents_without_groups_keep_their_order() -> None:
    docs = [6, 5]
    splits = [2, 2, 2, 5]
    modes = ["causal", "group", "causal", "causal"]
    perm = vision_group_order(docs, splits, modes)
    assert perm.tolist() == [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 10]
