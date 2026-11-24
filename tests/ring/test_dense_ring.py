# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Dense varlen ring steps reproduce the packed mask exactly (CPU simulation)."""

from __future__ import annotations

import pytest
import torch

from usp_attention.layouts import build_layout
from usp_attention.reference import dense_mask_rows
from usp_attention.ring.dense_ring import (
    build_dense_ring_plan,
    chunk_assignment,
    dense_ring_supported,
    rank_work,
)
from usp_attention.ring.packed_layout import build_packed_layout


def _varlen(q, k, v, call):
    """fp64 reference of one dense varlen call on gathered rows."""
    out = torch.zeros(len(call.q_index), q.shape[1], dtype=q.dtype)
    lse = torch.full((len(call.q_index),), float("-inf"), dtype=q.dtype)
    qg, kg, vg = q[call.q_index], k[call.k_index], v[call.k_index]
    for b in range(len(call.cu_q) - 1):
        qs, qe = int(call.cu_q[b]), int(call.cu_q[b + 1])
        ks, ke = int(call.cu_k[b]), int(call.cu_k[b + 1])
        s = qg[qs:qe] @ kg[ks:ke].T
        if call.anti:
            qi = torch.arange(qe - qs)[:, None]
            ki = torch.arange(ke - ks)[None, :]
            s = s.masked_fill(ki <= qi, float("-inf"))
        elif call.causal or call.strict:
            qi = torch.arange(qe - qs)[:, None] + (ke - ks) - (qe - qs)
            ki = torch.arange(ke - ks)[None, :]
            s = s.masked_fill(ki > qi if call.causal else ki >= qi, float("-inf"))
        lse[qs:qe] = torch.logsumexp(s, dim=-1)
        out[qs:qe] = torch.exp(s - lse[qs:qe, None]).nan_to_num(0.0) @ vg[ks:ke]
    return out, lse


@pytest.mark.parametrize(("ring", "per_rank"), [(2, 1), (4, 1), (2, 4), (4, 2)])
def test_dense_ring_matches_mask(ring: int, per_rank: int) -> None:
    spec = build_layout("mixed", 8192, seed=1)
    assert dense_ring_supported(*spec)
    n = sum(spec[0])
    local = n // ring
    chunks = chunk_assignment(*spec, ring_size=ring, chunks_per_rank=per_rank)
    perm = torch.cat([torch.arange(lo, hi) for rank in chunks for lo, hi in rank])
    torch.manual_seed(0)
    q, k, v = (torch.randn(n, 8, dtype=torch.float64) for _ in range(3))
    expect = dense_mask_rows(build_packed_layout(*spec), 0, n)
    s = (q @ k.T).masked_fill(~expect, float("-inf"))
    ref = torch.softmax(s, dim=-1).nan_to_num(0.0) @ v
    for r in range(ring):
        plan = build_dense_ring_plan(
            *spec, ring_rank=r, ring_size=ring, device="cpu", chunks=chunks
        )
        q_r = q[perm[r * local : (r + 1) * local]]
        acc = torch.zeros(local, 8, dtype=torch.float64)
        acc_lse = torch.full((local,), float("-inf"), dtype=torch.float64)
        for step, calls in enumerate(plan.steps):
            j = (r - step) % ring
            own = perm[j * local : (j + 1) * local]
            k_j, v_j = k[own], v[own]
            for call in calls:
                out, lse = _varlen(q_r, k_j, v_j, call)
                idx = call.q_index
                new = torch.logaddexp(acc_lse[idx], lse)
                acc[idx] = (
                    acc[idx] * torch.exp(acc_lse[idx] - new).nan_to_num(0.0)[:, None]
                    + out * torch.exp(lse - new).nan_to_num(0.0)[:, None]
                )
                acc_lse[idx] = new
        torch.testing.assert_close(
            acc, ref[perm[r * local : (r + 1) * local]], rtol=1e-10, atol=1e-10
        )


def test_rank_work_sums_to_visible_pairs() -> None:
    spec = build_layout("mixed", 8192, seed=1)
    assert len(rank_work(*spec, 2)) == 2
