# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest
import torch

from usp_attention.ring.dense_merge import accumulate_rows, merge_into


def _reference(acc_out, acc_lse, index, out, lse):
    acc_out, acc_lse = acc_out.double().clone(), acc_lse.double().clone()
    rows = torch.arange(acc_out.shape[0]) if index is None else index.cpu()
    old = acc_lse[:, rows]
    new = torch.logaddexp(old, lse.double().cpu())
    w_old = torch.exp(old - new).nan_to_num(0.0).T.unsqueeze(-1)
    w_new = torch.exp(lse.double().cpu() - new).nan_to_num(0.0).T.unsqueeze(-1)
    part = out.double().cpu().nan_to_num(0.0)
    acc_out[rows] = acc_out[rows] * w_old + part * w_new
    acc_lse[:, rows] = new
    return acc_out, acc_lse


def _case(device, *, n=300, m=170, heads=7, head_dim=128, indexed=True, seed=0):
    g = torch.Generator().manual_seed(seed)
    acc_out = torch.randn(n, heads, head_dim, generator=g)
    acc_lse = torch.randn(heads, n, generator=g)
    acc_lse[:, :20] = float("-inf")  # rows nothing has reached yet
    acc_out[:20] = 0.0
    if not indexed:
        m = n
    out = torch.randn(m, heads, head_dim, generator=g).bfloat16()
    lse = torch.randn(heads, m, generator=g)
    lse[:, -15:] = float("-inf")  # rows with no keys in this call
    out[-15:] = float("nan")
    index = None
    if indexed:
        # Unique rows, the first five empty on both sides.
        rest = torch.randperm(n - 5, generator=g)[: m - 5] + 5
        index = torch.cat([torch.arange(5), rest])
    return (
        acc_out.to(device),
        acc_lse.to(device),
        None if index is None else index.to(device),
        out.to(device),
        lse.to(device),
    )


def _check(device, monkeypatch, backend, **kw):
    monkeypatch.setenv("USP_DENSE_MERGE", backend)
    acc_out, acc_lse, index, out, lse = _case(device, **kw)
    want_out, want_lse = _reference(acc_out.cpu(), acc_lse.cpu(), index, out, lse)
    merge_into(acc_out, acc_lse, index, out, lse)
    torch.testing.assert_close(acc_out.cpu().double(), want_out, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(acc_lse.cpu().double(), want_lse, atol=1e-5, rtol=1e-5)
    assert not acc_out.isnan().any()


@pytest.mark.parametrize("indexed", [True, False])
def test_merge_torch(monkeypatch, indexed):
    _check("cpu", monkeypatch, "torch", indexed=indexed)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("indexed", [True, False])
@pytest.mark.parametrize("heads", [1, 7, 8, 32])
def test_merge_triton(monkeypatch, indexed, heads):
    _check("cuda", monkeypatch, "triton", indexed=indexed, heads=heads)


@pytest.mark.parametrize("backend", ["torch", "triton"])
@pytest.mark.parametrize("indexed", [True, False])
@pytest.mark.parametrize("heads", [1, 7])
def test_accumulate_rows(monkeypatch, backend, indexed, heads):
    device = "cuda" if backend == "triton" else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    monkeypatch.setenv("USP_DENSE_MERGE", backend)
    g = torch.Generator().manual_seed(1)
    acc = torch.randn(300, heads, 128, generator=g)
    src = torch.randn(170, heads, 128, generator=g).bfloat16()
    # Repeated rows, as when several query segments share keys.
    index = torch.randint(0, 120, (170,), generator=g) if indexed else None
    want = acc.double().clone()
    rows = index if indexed else torch.arange(170)
    want.index_add_(0, rows, src.double())
    acc, src = acc.to(device), src.to(device)
    accumulate_rows(acc, None if index is None else index.to(device), src)
    torch.testing.assert_close(acc.cpu().double(), want, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_merge_into_in_place_lse_has_no_race():
    # The kernel stores the new LSE in place; a warp re-loading its old LSE
    # after a neighbour stored the new one weighted the output wrong in about
    # 3% of calls at 32 heads before the LSE store moved behind a barrier.
    torch.manual_seed(0)
    heads, n = 32, 8192
    for _ in range(40):
        acc_out = torch.randn(n, heads, 128, device="cuda")
        acc_lse = 4 * torch.randn(heads, n, device="cuda")
        out = torch.randn(n, heads, 128, device="cuda").bfloat16()
        lse = 4 * torch.randn(heads, n, device="cuda")
        new = torch.logaddexp(acc_lse.double(), lse.double())
        ref = acc_out.double() * (acc_lse.double() - new).exp().T.unsqueeze(-1)
        ref += out.double() * (lse.double() - new).exp().T.unsqueeze(-1)
        merge_into(acc_out, acc_lse, None, out, lse)
        torch.testing.assert_close(acc_out.double(), ref, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(acc_lse.double(), new, atol=1e-5, rtol=1e-5)
