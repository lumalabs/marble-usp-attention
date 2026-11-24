# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""The two-call interval decomposition reproduces the packed mask exactly (CPU)."""

from __future__ import annotations

import pytest
import torch

from usp_attention.fa4.interval_mask import _batched, build_interval_plan
from usp_attention.layouts import build_layout
from usp_attention.reference import dense_mask_rows
from usp_attention.ring.packed_layout import build_packed_layout


def _attend(q, k, v, allowed):
    s = (q @ k.T).masked_fill(~allowed, float("-inf"))
    lse = torch.logsumexp(s, dim=-1)
    p = torch.exp(s - lse[:, None]).nan_to_num(0.0)
    return p @ v, lse


@pytest.mark.parametrize("layout", ["mixed", "full"])
def test_interval_decomposition_matches_mask(layout: str) -> None:
    spec = build_layout(layout, 8192, seed=3)
    plan = build_interval_plan(*spec, device="cpu")
    assert plan is not None
    n = sum(spec[0])
    torch.manual_seed(0)
    q, k, v = (torch.randn(n, 16, dtype=torch.float64) for _ in range(3))
    expect, _ = _attend(q, k, v, dense_mask_rows(build_packed_layout(*spec), 0, n))

    # Call A: causal within each document.
    doc = torch.repeat_interleave(torch.arange(len(spec[0])), torch.tensor(spec[0]))
    pos = torch.arange(n)
    allow_a = (doc[:, None] == doc[None, :]) & (pos[None, :] <= pos[:, None])
    out_a, lse_a = _attend(q, k, v, allow_a)
    # Call B: reversed spans, strictly causal within each span.
    idx = plan.span_index
    span = torch.repeat_interleave(
        torch.arange(len(plan.cu_spans) - 1), plan.cu_spans.diff().long()
    )
    local = torch.arange(idx.numel())
    allow_b = (span[:, None] == span[None, :]) & (local[None, :] < local[:, None])
    out_b, lse_b = _attend(q[idx], k[idx], v[idx], allow_b)

    lse_rows = torch.logaddexp(lse_a[idx], lse_b)
    out = out_a.clone()
    out[idx] = (
        out_a[idx] * torch.exp(lse_a[idx] - lse_rows)[:, None]
        + out_b * torch.exp(lse_b - lse_rows).nan_to_num(0.0)[:, None]
    )
    torch.testing.assert_close(out, expect, rtol=1e-10, atol=1e-10)


def test_batched_offsets_repeat_the_plan_per_sequence() -> None:
    spec = build_layout("mixed", 8192, seed=3)
    plan = build_interval_plan(*spec, device="cpu")
    assert plan is not None
    batch, seq = 3, sum(spec[0])
    cu_docs, index, cu_spans = _batched(plan, batch, seq)

    def repeated(cu: torch.Tensor) -> torch.Tensor:
        lens = cu.diff().repeat(batch)
        return torch.cat([cu[:1], lens.cumsum(0).to(cu.dtype)])

    torch.testing.assert_close(cu_docs, repeated(plan.cu_docs))
    torch.testing.assert_close(cu_spans, repeated(plan.cu_spans))
    torch.testing.assert_close(
        index, torch.cat([plan.span_index + b * seq for b in range(batch)])
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_batched_offsets_do_not_sync() -> None:
    spec = build_layout("mixed", 8192, seed=3)
    plan = build_interval_plan(*spec, device="cuda")
    assert plan is not None
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        _batched(plan, 2, sum(spec[0]))
    finally:
        torch.cuda.set_sync_debug_mode("default")


def test_noise_not_last_is_rejected() -> None:
    assert build_interval_plan([4], [2, 2], ["noise", "causal"], device="cpu") is None
    assert build_interval_plan([4], [4], ["causal"], device="cpu") is None
