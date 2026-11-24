# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Packed multimodal masks as two dense varlen FA4 calls instead of block sparsity.

In a packed multimodal sequence every query sees one contiguous range of keys:
from the start of its document to itself (a causal token) or to the end of its
split (a full or noise token), as long as a noise split is the last split of
its document (nothing later may see it). That mask is the union of

- A: causal attention within each document, and
- B: for every full or noise split, each query's keys *after* it inside the
  split. Reversing the split's token order turns B into strictly causal
  attention (keys before the query, diagonal excluded).

Both are dense varlen FA4 calls, so they run at FA4's dense speed (on Blackwell
with its 2-CTA backward, which block sparsity turns off). The two partial
results are merged by log-sum-exp in fp32. The backward runs both calls with
the merged output and log-sum-exp, as ring attention does, and sums the
gradients in fp32.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from usp_attention.ring.dense_merge import accumulate_rows, merge_into


@dataclass(kw_only=True, slots=True)
class IntervalPlan:
    """Varlen metadata for the two calls of :func:`interval_attention`.

    Attributes:
        cu_docs: Document boundaries ``[0, ..., seq_len]`` (int32).
        max_doc: Longest document.
        span_index: Token indices of every full / noise split, each split in
            reverse order, concatenated (int64).
        cu_spans: Split boundaries within ``span_index`` (int32).
        max_span: Longest full / noise split.
        sliding_window: Sliding window of call A in keys, ``None`` for none.
    """

    cu_docs: Tensor
    max_doc: int
    span_index: Tensor
    cu_spans: Tensor
    max_span: int
    sliding_window: int | None = None


def build_interval_plan(
    document_lens: list[int],
    split_lens: list[int],
    attn_modes: list[str],
    *,
    device: torch.device | str,
    sliding_window: int | None = None,
) -> IntervalPlan | None:
    """Plan the two-call decomposition, or ``None`` when the layout does not fit.

    The layout fits when it has at least one full or noise split, every
    noise split is the last split of its document, and there are no group
    splits. With a sliding window, call A becomes local attention over the
    last ``sliding_window`` keys, which is exact as long as no full or noise
    split is longer than the window (a span token must see its whole span).
    """
    spans: list[tuple[int, int]] = []
    split_iter = iter(zip(split_lens, attn_modes, strict=True))
    offset = 0
    for doc_len in document_lens:
        end = offset + doc_len
        pos = offset
        while pos < end:
            length, mode = next(split_iter)
            if mode == "noise" and pos + length != end:
                return None
            if mode == "group":
                # A group token also sees later group splits past causal
                # text it cannot see: no longer one contiguous key range.
                return None
            if mode in ("full", "noise"):
                if sliding_window and length > sliding_window:
                    return None
                spans.append((pos, pos + length))
            pos += length
        offset = end
    if not spans:
        return None
    index = torch.cat(
        [torch.arange(e - 1, s - 1, -1, dtype=torch.int64) for s, e in spans]
    )
    span_lens = [e - s for s, e in spans]
    cu_docs = torch.tensor([0, *document_lens], dtype=torch.int64).cumsum(0)
    cu_spans = torch.tensor([0, *span_lens], dtype=torch.int64).cumsum(0)
    return IntervalPlan(
        cu_docs=cu_docs.to(device=device, dtype=torch.int32),
        max_doc=max(document_lens),
        span_index=index.to(device),
        cu_spans=cu_spans.to(device=device, dtype=torch.int32),
        max_span=max(span_lens),
        sliding_window=sliding_window or None,
    )


def _batched(plan: IntervalPlan, batch: int, seq: int) -> tuple[Tensor, Tensor, Tensor]:
    """Boundaries and span indices for ``batch`` sequences flattened back to back."""
    if batch == 1:
        return plan.cu_docs, plan.span_index, plan.cu_spans

    def offsets(cu: Tensor, step: int) -> Tensor:
        shift = torch.arange(batch, device=cu.device, dtype=cu.dtype)[:, None] * step
        return torch.cat([(cu[:-1] + shift).flatten(), cu[-1:] + (batch - 1) * step])

    shift = torch.arange(batch, device=plan.span_index.device)[:, None] * seq
    index = (plan.span_index[None, :] + shift).flatten()
    # span_index holds every span token once, so its length is cu_spans[-1]
    # without reading the device tensor back (a host sync every forward).
    return (
        offsets(plan.cu_docs, seq),
        index,
        offsets(plan.cu_spans, plan.span_index.numel()),
    )


def _left(plan: IntervalPlan) -> int | None:
    """FA4's ``window_size_left`` for call A (keys ``q - left .. q``)."""
    return plan.sliding_window - 1 if plan.sliding_window else None


def _right(plan: IntervalPlan) -> int | None:
    return 0 if plan.sliding_window else None


class _IntervalAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, plan: IntervalPlan, softmax_scale, deterministic):
        # [MLR-02.08] flash_attn.cute is an optional, platform-specific dependency.
        from flash_attn.cute.interface import _flash_attn_fwd

        batch, seq = q.shape[:2]
        cu_docs, index, cu_spans = _batched(plan, batch, seq)
        qf, kf, vf = (t.reshape(batch * seq, *t.shape[2:]) for t in (q, k, v))
        out_a, lse_a, *_ = _flash_attn_fwd(
            qf, kf, vf,
            cu_seqlens_q=cu_docs, cu_seqlens_k=cu_docs,
            max_seqlen_q=plan.max_doc, max_seqlen_k=plan.max_doc,
            softmax_scale=softmax_scale, causal=True, return_lse=True,
            window_size_left=_left(plan), window_size_right=_right(plan),
        )  # fmt: skip
        out_b, lse_b, *_ = _flash_attn_fwd(
            qf[index], kf[index], vf[index],
            cu_seqlens_q=cu_spans, cu_seqlens_k=cu_spans,
            max_seqlen_q=plan.max_span, max_seqlen_k=plan.max_span,
            softmax_scale=softmax_scale, causal=False, window_size_right=-1,
            return_lse=True,
        )  # fmt: skip
        # Merge B into A's rows in fp32; B rows with no keys have lse -inf.
        lse = lse_a.clone(memory_format=torch.contiguous_format)
        out = out_a.float()
        merge_into(out, lse, index, out_b, lse_b)
        out = out.to(q.dtype)
        ctx.save_for_backward(q, k, v, out, lse, cu_docs, index, cu_spans)
        ctx.plan, ctx.softmax_scale, ctx.deterministic = (
            plan,
            softmax_scale,
            deterministic,
        )
        ctx.mark_non_differentiable(lse)
        return out.reshape(q.shape), lse

    @staticmethod
    def backward(ctx, dout, _dlse):
        # [MLR-02.08] flash_attn.cute is an optional, platform-specific dependency.
        from flash_attn.cute.interface import _flash_attn_bwd

        q, k, v, out, lse, cu_docs, index, cu_spans = ctx.saved_tensors
        plan = ctx.plan
        batch, seq = q.shape[:2]
        qf, kf, vf, doutf = (
            t.reshape(batch * seq, *t.shape[2:]) for t in (q, k, v, dout.contiguous())
        )
        dq, dk, dv = _flash_attn_bwd(
            qf, kf, vf, out, doutf, lse, ctx.softmax_scale, True,
            window_size_left=_left(plan), window_size_right=_right(plan),
            cu_seqlens_q=cu_docs, cu_seqlens_k=cu_docs,
            max_seqlen_q=plan.max_doc, max_seqlen_k=plan.max_doc,
            deterministic=ctx.deterministic,
        )  # fmt: skip
        dq_b, dk_b, dv_b = _flash_attn_bwd(
            qf[index], kf[index], vf[index], out[index], doutf[index],
            lse[:, index].contiguous(), ctx.softmax_scale, False,
            window_size_right=-1,
            cu_seqlens_q=cu_spans, cu_seqlens_k=cu_spans,
            max_seqlen_q=plan.max_span, max_seqlen_k=plan.max_span,
            deterministic=ctx.deterministic,
        )  # fmt: skip
        grads = []
        for full, part in ((dq, dq_b), (dk, dk_b), (dv, dv_b)):
            acc = full.float()
            accumulate_rows(acc, index, part)
            grads.append(acc.to(q.dtype).reshape(q.shape[0], seq, *full.shape[1:]))
        return *grads, None, None, None


def interval_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    plan: IntervalPlan,
    *,
    softmax_scale: float | None = None,
    deterministic: bool = False,
    return_lse: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Packed multimodal attention as two dense varlen FA4 calls.

    Args:
        q: ``(batch, seq, heads, head_dim)`` queries; every sequence shares the
            layout of ``plan``.
        k: ``(batch, seq, kv_heads, head_dim)`` keys.
        v: Values, same shape as ``k``.
        plan: From :func:`build_interval_plan`.
        softmax_scale: Defaults to ``1 / sqrt(head_dim)``.
        deterministic: Use FA4's deterministic backward.
        return_lse: Also return the ``(batch, heads, seq)`` log-sum-exp.

    Returns:
        The attention output, and the LSE when ``return_lse`` is set.
    """
    out, lse = _IntervalAttention.apply(q, k, v, plan, softmax_scale, deterministic)
    if not return_lse:
        return out
    batch, seq = q.shape[:2]
    return out, lse.reshape(lse.shape[0], batch, seq).transpose(0, 1).contiguous()
