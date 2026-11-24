# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""The dense reference agrees with PyTorch SDPA under the same mask."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from usp_attention.layouts import LAYOUT_NAMES, build_layout
from usp_attention.reference import (
    dense_mask_rows,
    reference_attention,
    visible_key_counts,
)
from usp_attention.ring.packed_layout import build_packed_layout


@pytest.mark.parametrize("name", LAYOUT_NAMES)
def test_reference_matches_sdpa(name: str) -> None:
    seq_len, heads, kv_heads, dim = 4096, 4, 2, 32
    docs, splits, modes = build_layout(name, seq_len, ring_size=2)
    torch.manual_seed(0)
    q = torch.randn(1, seq_len, heads, dim, dtype=torch.float64, requires_grad=True)
    k = torch.randn(1, seq_len, kv_heads, dim, dtype=torch.float64, requires_grad=True)
    v = torch.randn(1, seq_len, kv_heads, dim, dtype=torch.float64, requires_grad=True)

    out, lse = reference_attention(
        q,
        k,
        v,
        document_lens=docs,
        split_lens=splits,
        attn_modes=modes,
        dtype=torch.float64,
        chunk_size=512,
    )
    layout = build_packed_layout(docs, splits, modes)
    mask = dense_mask_rows(layout, 0, seq_len)
    want = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=mask,
        enable_gqa=True,
    ).transpose(1, 2)
    torch.testing.assert_close(out, want)
    assert torch.isfinite(lse).all()

    grad = torch.randn_like(out)
    got = torch.autograd.grad(out, (q, k, v), grad)
    ref = torch.autograd.grad(want, (q, k, v), grad)
    for g, r in zip(got, ref):
        torch.testing.assert_close(g, r)


@pytest.mark.parametrize("name", LAYOUT_NAMES)
def test_visible_key_counts_match_dense_mask(name: str) -> None:
    docs, splits, modes = build_layout(name, 8192, ring_size=4)
    layout = build_packed_layout(docs, splits, modes)
    dense = dense_mask_rows(layout, 0, layout.seq_len).sum(dim=1)
    torch.testing.assert_close(visible_key_counts(docs, splits, modes), dense)
