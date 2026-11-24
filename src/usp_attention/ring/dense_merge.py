# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Log-sum-exp merge of one attention call into an fp32 accumulator.

The dense ring and interval paths run several FA4 calls over subsets of the
query rows and merge each call's ``(out, lse)`` into running fp32 results. In
plain PyTorch one merge is about ten elementwise, gather and scatter kernels;
:func:`merge_into` does it in one Triton pass on CUDA and falls back to PyTorch
elsewhere (or with ``USP_DENSE_MERGE=torch``). :func:`accumulate_rows` does the
same for the backward's scattered fp32 gradient sums.
"""

from __future__ import annotations

import os

import torch
from torch import Tensor

try:
    # [MLR-02.08] Triton ships with CUDA builds of PyTorch only.
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CPU-only installs
    triton = None


def _merge_torch(
    acc_out: Tensor, acc_lse: Tensor, index: Tensor | None, out: Tensor, lse: Tensor
) -> None:
    if index is None:
        new = torch.logaddexp(acc_lse, lse)
        acc_out.mul_(torch.exp(acc_lse - new).nan_to_num(0.0).T.unsqueeze(-1))
        acc_out.add_(
            out.float().nan_to_num(0.0)
            * torch.exp(lse - new).nan_to_num(0.0).T.unsqueeze(-1)
        )
        acc_lse.copy_(new)
        return
    old = acc_lse[:, index]
    new = torch.logaddexp(old, lse)
    w_old = torch.exp(old - new).nan_to_num(0.0).T.unsqueeze(-1)
    w_new = torch.exp(lse - new).nan_to_num(0.0).T.unsqueeze(-1)
    acc_out[index] = acc_out[index] * w_old + out.float().nan_to_num(0.0) * w_new
    acc_lse[:, index] = new


if triton is not None:

    @triton.jit
    def _merge_kernel(
        acc_out, acc_lse, index, out, lse,
        heads,
        s_ao_n, s_ao_h, s_al_h, s_o_m, s_o_h, s_l_h, s_l_m,
        HAS_INDEX: tl.constexpr, BLOCK_H: tl.constexpr, HEAD_DIM: tl.constexpr,
    ):  # fmt: skip
        i = tl.program_id(0).to(tl.int64)
        row = tl.load(index + i).to(tl.int64) if HAS_INDEX else i
        hh = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
        dd = tl.arange(0, HEAD_DIM)
        hmask = hh < heads
        old = tl.load(acc_lse + hh * s_al_h + row, mask=hmask, other=float("-inf"))
        part = tl.load(lse + hh * s_l_h + i * s_l_m, mask=hmask, other=float("-inf"))
        top = tl.maximum(old, part)
        empty = top == float("-inf")
        top = tl.where(empty, 0.0, top)
        new = top + tl.log(tl.exp(old - top) + tl.exp(part - top))
        w_old = tl.where(empty, 0.0, tl.exp(old - new))
        w_new = tl.where(empty, 0.0, tl.exp(part - new))
        new = tl.where(empty, float("-inf"), new)
        mask2 = hmask[:, None]
        ao_ptr = acc_out + row * s_ao_n + hh[:, None] * s_ao_h + dd[None, :]
        acc = tl.load(ao_ptr, mask=mask2, other=0.0)
        o = tl.load(
            out + i * s_o_m + hh[:, None] * s_o_h + dd[None, :], mask=mask2, other=0.0
        ).to(tl.float32)
        o = tl.where(w_new[:, None] > 0, o * w_new[:, None], 0.0)
        tl.store(ao_ptr, acc * w_old[:, None] + o, mask=mask2)
        # The LSE goes last, behind a barrier: the compiler may re-load ``old``
        # from acc_lse in the 2-D layout above, and a warp must never read a
        # neighbour's new LSE as its old one.
        tl.debug_barrier()
        tl.store(acc_lse + hh * s_al_h + row, new, mask=hmask)


if triton is not None:

    @triton.jit
    def _accumulate_kernel(
        acc, index, src, n_cols, s_acc, s_src, scale,
        HAS_INDEX: tl.constexpr, BLOCK: tl.constexpr,
    ):  # fmt: skip
        i = tl.program_id(0).to(tl.int64)
        for start in range(0, n_cols, BLOCK):
            cols = start + tl.arange(0, BLOCK)
            mask = cols < n_cols
            x = tl.load(src + i * s_src + cols, mask=mask, other=0.0).to(tl.float32)
            x = x * scale
            if HAS_INDEX:
                # Key rows can repeat within one call (several query segments
                # over the same keys), so indexed rows add atomically.
                row = tl.load(index + i).to(tl.int64)
                tl.atomic_add(acc + row * s_acc + cols, x, mask=mask, sem="relaxed")
            else:
                a = tl.load(acc + i * s_acc + cols, mask=mask, other=0.0)
                tl.store(acc + i * s_acc + cols, a + x, mask=mask)


def _use_triton(acc: Tensor) -> bool:
    return (
        triton is not None
        and acc.is_cuda
        and os.environ.get("USP_DENSE_MERGE", "triton") != "torch"
    )


def accumulate_rows(
    acc: Tensor, index: Tensor | None, src: Tensor, scale: float = 1.0
) -> None:
    """``acc[index] += scale * src`` in fp32, in place.

    Args:
        acc: ``(n, ...)`` fp32 accumulator, contiguous.
        index: Accumulator rows of ``src``'s rows (may repeat), or ``None``
            for the leading rows in order.
        src: ``(m, ...)`` rows to add, any float dtype.
        scale: Factor applied in fp32 (e.g. undoing a power-of-two scaling).
    """
    m = src.shape[0]
    if m == 0:
        return
    if not (_use_triton(acc) and acc.is_contiguous() and src.is_contiguous()):
        x = src.to(acc.dtype) * scale if scale != 1.0 else src
        if index is None:
            acc[:m] += x
        else:
            acc.index_add_(0, index, x.to(acc.dtype))
        return
    n_cols = src[0].numel()
    _accumulate_kernel[(m,)](
        acc, index if index is not None else acc, src, n_cols,
        acc.stride(0), src.stride(0), float(scale),
        HAS_INDEX=index is not None, BLOCK=min(1024, triton.next_power_of_2(n_cols)),
    )  # fmt: skip


def merge_into(
    acc_out: Tensor, acc_lse: Tensor, index: Tensor | None, out: Tensor, lse: Tensor
) -> None:
    """Merge one call's rows into the fp32 running ``(out, lse)`` in place.

    Args:
        acc_out: ``(n, heads, head_dim)`` fp32 accumulator, contiguous.
        acc_lse: ``(heads, n)`` fp32 log-sum-exp, contiguous.
        index: Accumulator rows of the call's rows (unique), or ``None`` when
            the call covers every row in order.
        out: ``(m, heads, head_dim)`` call output.
        lse: ``(heads, m)`` call log-sum-exp; ``-inf`` for rows with no keys.
    """
    use_triton = (
        _use_triton(acc_out)
        and acc_out.is_contiguous()
        and acc_lse.is_contiguous()
        and out.stride(-1) == 1
        and acc_out.shape[-1] in (64, 128, 256)
    )
    if not use_triton:
        _merge_torch(acc_out, acc_lse, index, out, lse)
        return
    m, heads, head_dim = out.shape
    if m == 0:
        return
    block_h = min(triton.next_power_of_2(heads), 8)
    grid = (m, triton.cdiv(heads, block_h))
    _merge_kernel[grid](
        acc_out, acc_lse, index if index is not None else acc_lse, out, lse,
        heads,
        acc_out.stride(0), acc_out.stride(1), acc_lse.stride(0),
        out.stride(0), out.stride(1), lse.stride(0), lse.stride(1),
        HAS_INDEX=index is not None, BLOCK_H=block_h, HEAD_DIM=head_dim,
    )  # fmt: skip
