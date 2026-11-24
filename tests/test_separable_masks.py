# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""The separable mask forms used on SM90 partial tiles equal the dense mask."""

from __future__ import annotations

import pytest
import torch

from usp_attention.fa4.column_ranges import column_query_ranges
from usp_attention.layouts import build_layout
from usp_attention.reference import dense_mask_rows
from usp_attention.ring.flash4_ring_attn import (
    COLUMN_RANGE_SEPARABLE,
    POSITION_RANGE_SEPARABLE,
    cross_qkv_separable,
)
from usp_attention.ring.group_order import vision_group_order
from usp_attention.ring.packed_layout import build_packed_layout
from usp_attention.ring.ring_planning import build_cross_qkv_step_aux_tensors
from usp_attention.usp import _position_ranges

N = 8192


def _evaluate(separable, aux, q_len, kv_len):
    row_fn, col_fn, combine = separable
    rows = row_fn(torch.arange(q_len)[:, None], aux)
    cols = col_fn(torch.arange(kv_len)[None, :], aux)
    return combine(rows, cols)


@pytest.mark.parametrize(
    "name", ["causal", "docs", "mixed", "vlm", "video", "vlm_group"]
)
@pytest.mark.parametrize("window", [None, 100, 1024])
def test_cross_qkv_separable_matches_dense_mask(name: str, window: int | None) -> None:
    """Ring steps over a random 128-block permutation, every (q, kv) pair."""
    docs, splits, modes = build_layout(name, N, ring_size=4, seed=3)
    layout = build_packed_layout(docs, splits, modes)
    dense = dense_mask_rows(layout, 0, N, sliding_window=window)
    gen = torch.Generator().manual_seed(0)
    perm = (
        torch.randperm(N // 128, generator=gen)[:, None] * 128 + torch.arange(128)
    ).flatten()
    aux = [
        layout.full_and_noise_seq_id[perm],
        layout.noise_seq_id[perm],
        layout.document_id[perm],
    ]
    local = N // 4
    for q_start in range(0, N, local):
        for kv_start in range(0, N, local):
            step = build_cross_qkv_step_aux_tensors(
                global_aux_tensors=aux,
                q_start=q_start,
                q_len=local,
                kv_start=kv_start,
                kv_len=local,
                global_position_ids=perm,
            )
            got = _evaluate(cross_qkv_separable(window), step, local, local)
            q_ids = perm[q_start : q_start + local]
            kv_ids = perm[kv_start : kv_start + local]
            assert torch.equal(got, dense[q_ids][:, kv_ids])


@pytest.mark.parametrize("name", ["vlm_group", "video_group"])
def test_column_range_separable_matches_dense_mask(name: str) -> None:
    docs, splits, modes = build_layout(name, N, seed=3)
    perm = vision_group_order(docs, splits, modes)
    lo, hi = column_query_ranges(docs, splits, modes)
    aux = [torch.from_numpy(lo), torch.from_numpy(hi)]
    got = _evaluate(COLUMN_RANGE_SEPARABLE, aux, N, N)
    dense = dense_mask_rows(build_packed_layout(docs, splits, modes), 0, N)
    assert torch.equal(got, dense[perm][:, perm])


@pytest.mark.parametrize(
    "name", ["causal", "docs", "mixed", "vlm", "video", "vlm_group", "video_group"]
)
@pytest.mark.parametrize("window", [None, 100, 1024])
def test_position_ranges_match_dense_mask(name: str, window: int | None) -> None:
    """Ring-step column ranges under a random 128-block permutation, every pair."""
    docs, splits, modes = build_layout(name, N, ring_size=4, seed=3)
    if window is not None and "group" in modes:
        assert (
            _position_ranges(
                docs, splits, modes, None, None, device="cpu", sliding_window=window
            )
            is None
        )
        return
    gen = torch.Generator().manual_seed(1)
    perm = (
        torch.randperm(N // 128, generator=gen)[:, None] * 128 + torch.arange(128)
    ).flatten()
    tracks = _position_ranges(
        docs, splits, modes, perm, None, device="cpu", sliding_window=window
    )
    assert tracks is not None
    got = _evaluate(POSITION_RANGE_SEPARABLE, list(tracks), N, N)
    dense = dense_mask_rows(
        build_packed_layout(docs, splits, modes), 0, N, sliding_window=window
    )
    assert torch.equal(got, dense[perm][:, perm])


def test_position_ranges_need_default_positions() -> None:
    docs, splits, modes = build_layout("mixed", N, seed=3)
    shifted = torch.arange(N) + 1
    assert _position_ranges(docs, splits, modes, None, shifted, device="cpu") is None
