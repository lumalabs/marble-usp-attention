# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Merging Q blocks of FA4 block-sparse lists (SM100-class layout), on CPU."""

from __future__ import annotations

import torch

from usp_attention.fa4.flash4_block_sparse import (
    _dense_to_ordered,
    _ordered_to_dense,
    coarsen_q_blocks,
)


def _dense(cnt: torch.Tensor, idx: torch.Tensor, cols: int) -> torch.Tensor:
    return _ordered_to_dense(cnt, idx, num_cols=cols).bool()


def test_coarsen_matches_pairwise_rule() -> None:
    torch.manual_seed(0)
    q_blocks, kv_blocks = 7, 6  # odd: the last coarse block has one member
    cat = torch.randint(0, 3, (1, 1, q_blocks, kv_blocks))  # 0 empty, 1 mask, 2 full
    full, mask = cat == 2, cat == 1
    out = coarsen_q_blocks(
        *_dense_to_ordered(full),
        *_dense_to_ordered(mask),
        num_kv_blocks=kv_blocks,
        factor=2,
    )
    f_full = _dense(out[0], out[1], kv_blocks)
    f_mask = _dense(out[2], out[3], kv_blocks)
    for qb in range(4):
        rows = cat[0, 0, 2 * qb : 2 * qb + 2]
        if rows.shape[0] == 1:  # missing member counts as empty
            rows = torch.cat([rows, torch.zeros_like(rows)])
        expect_full = (rows == 2).all(0)
        expect_mask = (rows > 0).any(0) & ~expect_full
        assert torch.equal(f_full[0, 0, qb], expect_full)
        assert torch.equal(f_mask[0, 0, qb], expect_mask)
    # Backward lists are the transpose of the forward ones.
    b_full = _dense(out[4], out[5], 4)
    b_mask = _dense(out[6], out[7], 4)
    assert torch.equal(b_full, f_full.transpose(-2, -1))
    assert torch.equal(b_mask, f_mask.transpose(-2, -1))


def test_coarsen_kv_blocks_matches_tile_rule() -> None:
    torch.manual_seed(1)
    q_blocks, kv_blocks = 6, 7  # odd KV count: the last coarse KV block is half padding
    cat = torch.randint(0, 3, (1, 2, q_blocks, kv_blocks))
    full, mask = cat == 2, cat == 1
    out = coarsen_q_blocks(
        *_dense_to_ordered(full),
        *_dense_to_ordered(mask),
        num_kv_blocks=kv_blocks,
        factor=2,
        kv_factor=2,
    )
    f_full = _dense(out[0], out[1], 4)
    f_mask = _dense(out[2], out[3], 4)
    padded = torch.nn.functional.pad(cat, (0, 1))
    for h in range(2):
        for qb in range(3):
            for kb in range(4):
                tile = padded[0, h, 2 * qb : 2 * qb + 2, 2 * kb : 2 * kb + 2]
                expect_full = bool((tile == 2).all())
                expect_mask = bool((tile > 0).any()) and not expect_full
                assert bool(f_full[0, h, qb, kb]) == expect_full
                assert bool(f_mask[0, h, qb, kb]) == expect_mask
    b_full = _dense(out[4], out[5], 3)
    assert torch.equal(b_full, f_full.transpose(-2, -1))
