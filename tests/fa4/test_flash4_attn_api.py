# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""FA4 wrapper contracts: ``deterministic`` and the LSE gradient reach FA4's
backward, and ``flash4_attn_dense`` rejects a non-causal window on every path.

The plumbing test swaps FA4's kernels for recorders, so it runs on CPU wherever
``flash_attn.cute`` imports; the reference comparisons need an H100.
"""

from __future__ import annotations

import pytest
import torch

import usp_attention.fa4.flash4_attn as fa4
from usp_attention.fa4.flash4_attn import FA4SparseMask, flash4_attn, flash4_attn_dense

_needs_h100 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="needs an H100 for FA4 block sparsity",
)


@pytest.mark.parametrize("varlen", [False, True])
def test_dense_window_requires_causal(varlen: bool) -> None:
    q = torch.zeros(1, 8, 1, 16, dtype=torch.bfloat16)
    kw = (
        {"cu_seqlens": torch.tensor([0, 4, 8], dtype=torch.int32), "max_seqlen": 4}
        if varlen
        else {}
    )
    with pytest.raises(ValueError, match="sliding_window requires causal"):
        flash4_attn_dense(q, q, q, causal=False, sliding_window=2, **kw)


@pytest.mark.skipif(not fa4.HAS_FLASH4, reason="needs flash_attn.cute")
@pytest.mark.parametrize("loss_on", ["out", "lse", "both"])
@pytest.mark.parametrize("deterministic", [False, True])
def test_flags_reach_fa4_backward(
    monkeypatch: pytest.MonkeyPatch, loss_on: str, deterministic: bool
) -> None:
    from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch

    calls: list[dict] = []

    def fake_fwd(q, k, v, **kwargs):
        b, s, h, _ = q.shape
        return q.new_zeros(b, s, h, v.shape[-1]), torch.zeros(b, h, s)

    def fake_bwd(q, k, v, out, dout, lse, *args, **kwargs):
        calls.append({"dout": dout, **kwargs})
        return torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)

    monkeypatch.setattr(fa4, "_flash_attn_fwd", fake_fwd)
    monkeypatch.setattr(fa4, "_flash_attn_bwd", fake_bwd)
    seq = 8
    mask = FA4SparseMask(
        block_sparse=BlockSparseTensorsTorch(
            mask_block_cnt=torch.ones(1, 1, 1, dtype=torch.int32),
            mask_block_idx=torch.zeros(1, 1, 1, 1, dtype=torch.int32),
        ),
        block_sparse_bwd=None,
        aux_tensors=[
            torch.full((seq,), -1.0),
            torch.full((seq,), -1.0),
            torch.ones(seq, dtype=torch.int64),
        ],
        mask_mod=fa4.SPARSE_MASK_MOD,
        sparsity=None,
    )
    q, k, v = (
        torch.randn(1, seq, h, 16, dtype=torch.bfloat16, requires_grad=True)
        for h in (2, 1, 1)
    )
    out, lse = flash4_attn(q, k, v, mask, return_lse=True, deterministic=deterministic)

    if not fa4._FA4_BWD_TAKES_DLSE:
        # No LSE gradient in this FA4: the LSE is an explicit constant.
        assert not lse.requires_grad
        if loss_on == "lse":
            pytest.skip("the LSE is a constant with this FA4")
        loss_on = "out"
    loss = {
        "out": lambda: out.float().sum(),
        "lse": lambda: (2.0 * lse).sum(),
        "both": lambda: out.float().sum() + (2.0 * lse).sum(),
    }[loss_on]()
    loss.backward()

    (call,) = calls
    assert call["deterministic"] is deterministic
    if loss_on == "out":
        assert "dlse" not in call
    else:
        torch.testing.assert_close(call["dlse"], torch.full_like(lse, 2.0))
    if loss_on == "lse":
        # Only the LSE reached the loss: the output gradient is zeros.
        assert not call["dout"].any()


def _mixed_inputs(seq: int = 4096):
    from usp_attention.layouts import build_layout

    docs, splits, modes = build_layout("mixed", seq, seed=1)
    gen = torch.Generator(device="cuda").manual_seed(0)
    q, k, v, dout = (
        torch.randn(1, seq, h, 128, device="cuda", dtype=torch.bfloat16, generator=gen)
        for h in (8, 4, 4, 8)
    )
    # Scaled so the LSE term is as large as the output term in the gradients.
    dlse = 16.0 * torch.randn(1, seq, 8, device="cuda", generator=gen)
    return (docs, splits, modes), (q, k, v), dout, dlse


@pytest.mark.gpu
@_needs_h100
def test_lse_gradient_matches_reference() -> None:
    if not fa4._FA4_BWD_TAKES_DLSE:
        pytest.skip("this FA4 backward takes no dlse; the LSE is a constant")
    from usp_attention.fa4.flash4_attn import build_fa4_sparse_mask
    from usp_attention.reference import reference_attention

    (docs, splits, modes), qkv, dout, dlse = _mixed_inputs()
    mask = build_fa4_sparse_mask(docs, splits, modes, device="cuda")
    results = []
    for use_fa4 in (True, False):
        leaves = [t.clone().requires_grad_(True) for t in qkv]
        if use_fa4:
            out, lse = flash4_attn(*leaves, mask, return_lse=True)
            lse = lse.transpose(1, 2)  # (B, H, S) -> the reference's (B, S, H)
        else:
            out, lse = reference_attention(
                *leaves, document_lens=docs, split_lens=splits, attn_modes=modes
            )
        ((out.float() * dout.float()).sum() + (lse.float() * dlse).sum()).backward()
        results.append([out.float(), lse.float(), *(t.grad.float() for t in leaves)])
    for got, want in zip(*results):
        rel = (got - want).abs().mean() / want.abs().mean()
        assert rel < 1e-2


@pytest.mark.gpu
@_needs_h100
def test_compiled_matches_eager_with_lse_loss() -> None:
    """The op's autograd (LSE gradient or constant LSE) traces under compile."""
    from usp_attention.fa4.flash4_attn import build_fa4_sparse_mask

    (docs, splits, modes), qkv, dout, dlse = _mixed_inputs()
    mask = build_fa4_sparse_mask(docs, splits, modes, device="cuda")

    def loss_fn(q, k, v):
        out, lse = flash4_attn(q, k, v, mask, return_lse=True)
        loss = (out.float() * dout.float()).sum()
        if lse.requires_grad:
            loss = loss + (lse.transpose(1, 2) * dlse).sum()
        return loss

    results = []
    for fn in (loss_fn, torch.compile(loss_fn)):
        leaves = [t.clone().requires_grad_(True) for t in qkv]
        fn(*leaves).backward()
        results.append([t.grad.float() for t in leaves])
    for got, want in zip(*results):
        assert (got - want).abs().mean() / want.abs().mean() < 1e-3


@pytest.mark.gpu
@_needs_h100
def test_deterministic_backward_is_reproducible() -> None:
    """Bitwise-equal gradients over two runs, or FA4 refuses deterministic mode.

    FA4 releases whose deterministic block-sparse backward needs dQ write-order
    metadata (which ``build_fa4_sparse_mask`` does not build) raise instead;
    the flag must never be dropped silently.
    """
    from usp_attention.fa4.flash4_attn import build_fa4_sparse_mask

    (docs, splits, modes), qkv, dout, _dlse = _mixed_inputs()
    mask = build_fa4_sparse_mask(docs, splits, modes, device="cuda")

    def grads() -> list[torch.Tensor]:
        leaves = [t.clone().requires_grad_(True) for t in qkv]
        flash4_attn(*leaves, mask, deterministic=True).backward(dout)
        return [t.grad for t in leaves]

    try:
        first = grads()
    except (AssertionError, ValueError) as exc:
        assert "determinis" in str(exc)
        pytest.skip(f"this FA4 cannot run a deterministic block-sparse backward: {exc}")
    for a, b in zip(first, grads()):
        assert torch.equal(a, b)
