# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Interval-based needed KV blocks match the block classification."""

from __future__ import annotations

import pytest
import torch

from usp_attention.fa4.flash4_attn import HAS_FLASH4, build_fa4_sparse_mask
from usp_attention.layouts import build_layout
from usp_attention.ring.group_order import vision_group_order
from usp_attention.ring.halo import _needed_blocks, _visible_kv_blocks, needed_key_masks
from usp_attention.sliding_window import segment_end_positions

pytestmark = pytest.mark.skipif(
    not HAS_FLASH4, reason="FA4 metadata helpers unavailable"
)

N = 8192
BLOCK = 128


@pytest.mark.parametrize("name", ["docs", "mixed", "vlm", "video", "vlm_group"])
@pytest.mark.parametrize("window", [None, 300, 1024])
@pytest.mark.parametrize("order", ["identity", "shuffled_blocks", "group"])
def test_needed_blocks_match_classification(name, window, order) -> None:
    ring = 4
    docs, splits, modes = build_layout(name, N, ring_size=ring, seed=2)
    if order == "group":
        perm = vision_group_order(docs, splits, modes)
        if perm is None:
            pytest.skip("no vision groups")
    elif order == "shuffled_blocks":
        g = torch.Generator().manual_seed(0)
        blocks = torch.randperm(N // BLOCK, generator=g)
        perm = (blocks[:, None] * BLOCK + torch.arange(BLOCK)).flatten()
    else:
        perm = None
    aux = build_fa4_sparse_mask(
        docs, splits, modes, device="cpu", build_block_sparse=False
    ).aux_tensors
    pos = torch.arange(N)
    seg = segment_end_positions(aux[0], pos)
    if perm is not None:
        aux = [t[perm].contiguous() for t in aux]
        pos, seg = pos[perm].contiguous(), seg[perm].contiguous()
    masks = needed_key_masks(
        docs,
        splits,
        modes,
        ring_size=ring,
        token_permutation=perm,
        sliding_window=window,
    )
    local = N // ring
    for r in range(ring):
        for o in range(ring):
            if r == o:
                continue
            ref = _visible_kv_blocks(
                aux, pos, seg, q_rank=r, kv_rank=o, local_len=local,
                window=window, block_size=BLOCK,
            )  # fmt: skip
            got = _needed_blocks(
                masks, q_rank=r, kv_rank=o, local_len=local, block_size=BLOCK
            )
            assert set(ref.tolist()) <= set(got.tolist()), "a visible block is missing"
            assert set(got.tolist()) == set(ref.tolist())
