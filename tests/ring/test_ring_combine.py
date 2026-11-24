# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from usp_attention.ring.ring_combine import (
    _ensure_lse_bwd_layout,
    _LocalRingResultMerger,
    _normalize_step_lse,
    _use_fused_merge,
)

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


class TestLocalRingResultMerger:
    def test_single_step_is_identity(self):
        out = torch.randn(1, 4, 8)
        lse = torch.randn(1, 4)
        merger = _LocalRingResultMerger()
        merger.step(out, lse)
        result_out, result_lse = merger.result()
        torch.testing.assert_close(result_out, out)

    def test_two_equal_steps(self):
        out = torch.randn(1, 4, 8)
        lse = torch.zeros(1, 4)
        merger = _LocalRingResultMerger()
        merger.step(out.clone(), lse.clone())
        merger.step(out.clone(), lse.clone())
        result_out, _ = merger.result()
        torch.testing.assert_close(result_out, out, atol=1e-5, rtol=1e-5)

    def test_large_lse_dominates(self):
        out_a = torch.ones(1, 4, 2)
        lse_a = torch.full((1, 4), 100.0)
        out_b = torch.zeros(1, 4, 2)
        lse_b = torch.full((1, 4), -100.0)

        merger = _LocalRingResultMerger()
        merger.step(out_a, lse_a)
        merger.step(out_b, lse_b)
        result_out, _ = merger.result()
        torch.testing.assert_close(result_out, out_a, atol=1e-5, rtol=1e-5)

    def test_numerical_reference_three_steps(self):
        """Compare merger against explicit softmax-weighted combination."""
        torch.manual_seed(42)
        B, S, D = 1, 4, 2
        outs = [torch.randn(B, S, D) for _ in range(3)]
        lses = [torch.randn(B, S) for _ in range(3)]

        merger = _LocalRingResultMerger()
        for o, l in zip(outs, lses):
            merger.step(o, l)
        merged_out, _ = merger.result()

        lse_stack = torch.stack(lses, dim=0)
        weights = torch.softmax(lse_stack, dim=0).unsqueeze(-1)
        out_stack = torch.stack(outs, dim=0)
        reference = (weights * out_stack).sum(dim=0)
        torch.testing.assert_close(merged_out, reference, atol=1e-4, rtol=1e-4)

    def test_result_before_step_raises(self):
        merger = _LocalRingResultMerger()
        with pytest.raises(RuntimeError, match="No ring steps were merged"):
            merger.result()

    def test_lse_trailing_one_squeezed(self):
        out = torch.randn(1, 4, 8)
        lse = torch.randn(1, 4, 1)
        merger = _LocalRingResultMerger()
        merger.step(out, lse)
        _, result_lse = merger.result()
        assert result_lse.shape == (1, 4)


def _fp32_steps(head_dim: int, *, steps: int = 4, nan_empty: bool = True):
    """(B, S, H, D) fp32 step outputs and (B, S, H) LSEs with empty rows.

    Rows 0-6 have no keys at any step, rows 7-14 none at step 0 only, and rows
    15-19 keys at step 0 only. Empty rows carry NaN outputs if ``nan_empty``.
    """
    g = torch.Generator().manual_seed(head_dim)
    shape = (2, 300, 5)
    outs = [torch.randn(*shape, head_dim, generator=g) for _ in range(steps)]
    lses = [3 * torch.randn(*shape, generator=g) for _ in range(steps)]
    empty = [slice(0, 15)] + [[*range(7), *range(15, 20)]] * (steps - 1)
    for o, lse, rows in zip(outs, lses, empty):
        lse[:, rows] = float("-inf")
        o[:, rows] = float("nan") if nan_empty else 0.0
    return outs, lses


def _merge_reference(outs, lses):
    """fp64 softmax-weighted merge; rows no step reaches stay 0 with LSE -inf."""
    lse = torch.stack(lses).double().logsumexp(dim=0)
    out = torch.zeros_like(outs[0], dtype=torch.float64)
    for o, part in zip(outs, lses):
        w = (part.double() - lse).exp().nan_to_num(0.0).unsqueeze(-1)
        out += torch.where(w > 0, o.double() * w, 0.0)
    return out, lse


@pytest.mark.gpu
@needs_cuda
class TestFusedFp32Merge:
    @pytest.mark.parametrize("head_dim", [64, 96, 128, 256])
    def test_matches_fp64_reference(self, head_dim):
        outs, lses = _fp32_steps(head_dim)
        want_out, want_lse = _merge_reference(outs, lses)
        cuda = [(o.cuda(), lse.cuda()) for o, lse in zip(outs, lses)]
        first = tuple(t.clone() for t in cuda[0])
        merger = _LocalRingResultMerger()
        for o, lse in cuda:
            assert _use_fused_merge(o, lse.unsqueeze(-1), o, lse.unsqueeze(-1))
            merger.step(o, lse)
        got_out, got_lse = merger.result()
        assert got_out.dtype == torch.float32 and got_lse.shape == lses[0].shape
        tol = {"atol": 1e-5, "rtol": 1e-5}
        torch.testing.assert_close(got_out.cpu().double(), want_out, **tol)
        torch.testing.assert_close(got_lse.cpu().double(), want_lse, **tol)
        assert got_lse[:, :7].isneginf().all() and not got_lse[:, 7:].isinf().any()
        # The first step's tensors belong to the caller and stay untouched.
        torch.testing.assert_close(cuda[0], first, equal_nan=True)

    def test_as_accurate_as_the_python_merge(self):
        g = torch.Generator().manual_seed(1)
        shape = (2, 1024, 32, 128)  # many rows: a racy write would likely show
        outs = [torch.randn(shape, generator=g) for _ in range(4)]
        lses = [4 * torch.randn(shape[:-1], generator=g) for _ in range(4)]
        want_out, want_lse = _merge_reference(outs, lses)
        fused, python = _LocalRingResultMerger(), _LocalRingResultMerger()
        for o, lse in zip(outs, lses):
            fused.step(o.cuda(), lse.cuda())
            python.step(o, lse)
        got, ref = fused.result(), python.result()
        torch.testing.assert_close(got[0].cpu(), ref[0], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(got[1].cpu(), ref[1], atol=1e-5, rtol=1e-5)

        def err(out):
            return (out.cpu().double() - want_out).abs().mean() / want_out.abs().mean()

        assert err(got[0]) <= 1.1 * err(ref[0])
        assert (got[1].cpu().double() - want_lse).abs().max() <= 1e-5

    def test_bf16_steps_keep_the_python_merge(self):
        """``combine_mode="python"`` (bf16 steps) keeps its exact numerics."""
        outs, lses = _fp32_steps(128, nan_empty=False)
        lses = [lse.clamp_min(-30.0).cuda() for lse in lses]
        outs = [o.cuda().bfloat16() for o in outs]
        lse0, lse1 = lses[0].unsqueeze(-1), lses[1].unsqueeze(-1)
        assert not _use_fused_merge(outs[0], lse0, outs[1], lse1)
        merger = _LocalRingResultMerger()
        for o, lse in zip(outs, lses):
            merger.step(o, lse)
        out, lse = outs[0], lses[0].unsqueeze(-1)
        for o, part in zip(outs[1:], lses[1:]):
            part = part.unsqueeze(-1)
            out = out - torch.sigmoid(part - lse) * (out - o)
            lse = lse - F.logsigmoid(lse - part)
        got_out, got_lse = merger.result()
        assert torch.equal(got_out, out) and torch.equal(got_lse, lse.squeeze(-1))


def test_cpu_fp32_steps_use_the_python_merge():
    outs, lses = _fp32_steps(64, nan_empty=False)
    lse0, lse1 = lses[0].unsqueeze(-1), lses[1].unsqueeze(-1)
    assert not _use_fused_merge(outs[0], lse0, outs[1], lse1)


class TestNormalizeStepLSE:
    def test_passthrough_correct_shape(self):
        out = torch.randn(2, 4, 8)
        lse = torch.randn(2, 4)
        result = _normalize_step_lse(out, lse)
        torch.testing.assert_close(result, lse)

    def test_trailing_one_squeezed(self):
        out = torch.randn(2, 4, 8)
        lse = torch.randn(2, 4, 1)
        result = _normalize_step_lse(out, lse)
        assert result.shape == (2, 4)

    def test_transposed_layout(self):
        out = torch.randn(2, 4, 8, 16)
        lse = torch.randn(2, 8, 4)
        result = _normalize_step_lse(out, lse)
        assert result.shape == (2, 4, 8)

    def test_incompatible_raises(self):
        out = torch.randn(2, 4, 8)
        lse = torch.randn(3, 5)
        with pytest.raises(ValueError):
            _normalize_step_lse(out, lse)


class TestEnsureLSEBwdLayout:
    def test_already_bhs(self):
        out = torch.randn(2, 4, 8, 16)
        lse = torch.randn(2, 8, 4)
        result = _ensure_lse_bwd_layout(lse, out)
        torch.testing.assert_close(result, lse)

    def test_bsh_transposed(self):
        out = torch.randn(2, 4, 8, 16)
        lse_bsh = torch.randn(2, 4, 8)
        result = _ensure_lse_bwd_layout(lse_bsh, out)
        assert result.shape == (2, 8, 4)
        torch.testing.assert_close(result, lse_bsh.transpose(-1, -2))

    def test_bad_shape_raises(self):
        out = torch.randn(2, 4, 8, 16)
        lse = torch.randn(2, 3, 5)
        with pytest.raises(ValueError, match="Cannot reshape LSE"):
            _ensure_lse_bwd_layout(lse, out)
