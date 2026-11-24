# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Mask-aware KV gather on dense varlen FA4 calls.

The dense counterpart of :mod:`usp_attention.ring.halo` for packed multimodal
layouts that satisfy the interval condition
(:func:`~usp_attention.ring.dense_ring.dense_ring_supported`), for GPUs and
head dims where FA4 has no block sparsity (head_dim 256 on H100). Ring ranks
hold contiguous chunks (:func:`~usp_attention.ring.dense_ring.chunk_assignment`).
At plan time every rank lists the keys its queries can see: for each query
piece, its document from the start up to the piece's end, plus the rest of
each full / noise split it overlaps. One variable-size all-to-all over the ring
group brings in the remote ones, then at most two dense varlen calls attend
over ``local KV ++ halo`` in original token order:

- causal: each run of a document's non-noise queries against the document's
  non-noise keys up to the run's end (aligned bottom-right);
- later keys: each full piece's queries, reversed, against the rest of the
  split, reversed and strictly causal;
- noise: each noise piece's queries against the document's non-noise keys
  before its span followed by the whole span (full attention). A noise span
  can sit anywhere in its document (BAGEL-style interleaved generation: noised
  latents of a frame, then its clean copy); its keys are hidden from every
  other query. Prefixes are gathered per span, so these calls are split to
  bound the gathered keys.

The two calls merge by log-sum-exp in fp32. The backward adds every call's dK
and dV (and dq, unless each query row comes from one call) into fp32 buffers,
returns the halo's share to its owners with the reverse all-to-all, and rounds
once. What each call contributes depends on where FA4 accumulates: gradients
it accumulates in fp32 (dq, and dK / dV with grouped-query attention) come out
in fp16 with a power-of-two scale, three mantissa bits more than bf16
(:func:`~usp_attention.ring.fp32_grad.fp32_grad_outputs`); without GQA FA4
writes each call's dK / dV directly in the input dtype, so those are rounded to
bf16 per call before the fp32 sum (as is every gradient of FA4's head_dim-256
backward on Blackwell).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor

from usp_attention.fa4.flash4_attn import _fa3_flash_attn_func
from usp_attention.ring.dense_merge import accumulate_rows, merge_into
from usp_attention.ring.dense_ring import DenseCall, _call_bwd, _call_fwd, _segments
from usp_attention.ring.fp32_grad import as_scaled, fp32_grad_outputs
from usp_attention.ring.halo import _exchange

if TYPE_CHECKING:
    from jaxtyping import Float


@dataclass(kw_only=True, slots=True)
class DenseGatherPlan:
    """Per-rank metadata for :func:`dense_gather_attention`.

    Attributes:
        local_len: Tokens per ring rank.
        send_index: Local rows sent to other ring ranks, grouped by destination
            in ring-rank order.
        send_counts: Rows sent to each ring rank.
        recv_counts: Rows received from each ring rank.
        calls: Dense calls; ``k_index`` rows index ``local KV ++ halo``.
        use_fa3: Run the single causal call with FlashAttention-3 (H100).
        rows_disjoint: Every query row comes from exactly one call, so outputs
            and dq are written in place instead of merged / accumulated.
    """

    local_len: int
    send_index: Tensor
    send_counts: list[int]
    recv_counts: list[int]
    calls: list[DenseCall]
    use_fa3: bool = False
    rows_disjoint: bool = False

    @property
    def halo_len(self) -> int:
        """Remote keys this rank receives."""
        return sum(self.recv_counts)


def _needed(chunks, docs, spans, noise, seq_len: int) -> np.ndarray:
    """Keys (global positions) that the queries in ``chunks`` can see."""
    need = np.zeros(seq_len, dtype=bool)
    for a, b in chunks:
        for d0, d1 in docs:
            hi = min(b, d1)
            if hi > max(a, d0):
                need[d0:hi] = True
        for s0, s1 in spans:
            lo = max(a, s0)
            if min(b, s1) > lo:
                need[lo:s1] = True
    for s0, s1 in noise:
        # A noise span is visible only to its own queries, and to all of them.
        need[s0:s1] = any(a < s1 and b > s0 for a, b in chunks)
    return need


def dense_gather_supported(
    document_lens: list[int], split_lens: list[int], attn_modes: list[str]
) -> bool:
    """Check whether dense calls support the packed attention layout.

    Args:
        document_lens: Lengths of the packed documents, in tokens.
        split_lens: Lengths of the attention spans, in document order.
        attn_modes: Attention mode for each span; must match split_lens.

    Returns:
        True when the layout has no vision-group spans.
    """
    return "group" not in attn_modes


def _make_call(q_parts, k_parts, *, causal, strict, device) -> DenseCall | None:
    if not q_parts:
        return None
    q_index = torch.from_numpy(np.concatenate(q_parts))
    k_index = torch.from_numpy(np.concatenate(k_parts))
    q_lens = [len(p) for p in q_parts]
    k_lens = [len(p) for p in k_parts]

    def cu(lens):
        return (
            torch.tensor([0, *lens], dtype=torch.int64)
            .cumsum(0)
            .to(device=device, dtype=torch.int32)
        )

    return DenseCall(
        q_index=q_index.to(device),
        k_index=k_index.to(device),
        cu_q=cu(q_lens),
        cu_k=cu(k_lens),
        max_q=max(q_lens),
        max_k=max(k_lens),
        causal=causal,
        strict=strict,
        q_identity=torch.equal(q_index, torch.arange(q_index.numel())),
        k_identity=torch.equal(k_index, torch.arange(k_index.numel())),
    )


def build_dense_gather_plan(
    document_lens: list[int],
    split_lens: list[int],
    attn_modes: list[str],
    *,
    ring_rank: int,
    ring_size: int,
    chunks: list[list[tuple[int, int]]],
    device: torch.device | str,
    allow_fa3: bool = False,
    max_gathered_keys: int = 1 << 21,
    exact_output: bool = False,
) -> DenseGatherPlan:
    """Plan the gather and the dense calls of ``ring_rank``.

    Every rank derives the same send / receive lists from the layout, so
    planning needs no communication.

    Args:
        document_lens: Lengths of the packed documents, in tokens.
        split_lens: Lengths of the attention spans, in document order.
        attn_modes: Attention mode for each span; must match split_lens.
        ring_rank: Index of the calling rank within the ring.
        ring_size: Number of ranks in the ring.
        device: Device on which to allocate plan tensors.
        chunks: Global token ranges per ring rank, in local order (from
            :func:`~usp_attention.ring.dense_ring.chunk_assignment`).
        allow_fa3: Run the plan with FlashAttention-3 when it is one causal
            call (packed causal documents) and FA3 is installed. On H100 its
            varlen causal kernel is about 20% faster than FA4's at head_dim 256
            (Qwen3.8 shapes), but its dK / dV come out rounded, and a key's
            contributions from different ranks are then summed from bf16 (1.4x
            the single-GPU dK / dV error).
        max_gathered_keys: Keys gathered per call (memory bound): prefixes are
            gathered once per query sequence, so calls are split.
        exact_output: Answer full-span queries like noise queries (visible
            prefix plus the whole span in one call), so every query row comes
            from exactly one FA4 call. By default full spans take a causal and
            a reversed later-keys call, and noise spans share one call over
            their document's keys before this rank's chunk (one gather of the
            prefix instead of one per span); the forward then runs in fp16 and
            merges partials in fp32, so the output is at least as accurate as
            one bf16 FA4 call (0.68x its error), and gradients match
            (1.0-1.02x).

    Returns:
        A DenseGatherPlan containing local query calls, halo transfers and the
        partial-output merge requirement for the calling rank.
    """
    seq_len = sum(document_lens)
    local = seq_len // ring_size
    segs = list(_segments(document_lens, split_lens, attn_modes))
    docs = sorted({(d0, d1) for d0, d1, _, _, _ in segs})
    spans = [(s0, s1) for _, _, s0, s1, m in segs if m == "full"]
    noise_spans = [(d0, s0, s1) for d0, _, s0, s1, m in segs if m == "noise"]
    noise = [(s0, s1) for _, s0, s1 in noise_spans]
    is_noise = np.zeros(seq_len, dtype=bool)
    for s0, s1 in noise:
        is_noise[s0:s1] = True
    # Rows answered by one full call over gathered keys (no partial merge).
    is_full = is_noise.copy()
    for s0, s1 in spans:
        is_full[s0:s1] = True
    docs_start = np.array([d0 for d0, _ in docs], dtype=np.int64)
    docs_end = np.array([d1 for _, d1 in docs], dtype=np.int64)

    owner = np.empty(seq_len, dtype=np.int64)
    local_row = np.empty(seq_len, dtype=np.int64)
    for rank, ranges in enumerate(chunks):
        off = 0
        for lo, hi in ranges:
            owner[lo:hi] = rank
            local_row[lo:hi] = np.arange(off, off + hi - lo)
            off += hi - lo
    r = ring_rank
    need = [_needed(chunks[o], docs, spans, noise, seq_len) for o in range(ring_size)]
    recv_pos = [
        np.nonzero(need[r] & (owner == o))[0] if o != r else np.empty(0, np.int64)
        for o in range(ring_size)
    ]
    send_pos = [
        np.nonzero(need[o] & (owner == r))[0] if o != r else np.empty(0, np.int64)
        for o in range(ring_size)
    ]
    buf_row = np.full(seq_len, -1, dtype=np.int64)
    mine = owner == r
    buf_row[mine] = local_row[mine]
    off = local
    for pos in recv_pos:
        buf_row[pos] = np.arange(off, off + len(pos))
        off += len(pos)

    def visible_keys(d0: int, end: int) -> np.ndarray:
        """Buffer rows of the document's non-noise keys in ``[d0, end)``."""
        pos = np.arange(d0, end)
        return buf_row[pos[~is_noise[d0:end]]]

    causal_q, causal_k, later_q, later_k = [], [], [], []
    prefix_q: list[np.ndarray] = []
    prefix_k: list[np.ndarray] = []
    noise_parts: list[tuple[np.ndarray, np.ndarray]] = []
    q_off = 0
    for a, b in chunks[r]:
        for d0, d1 in docs:
            lo, hi = max(a, d0), min(b, d1)
            if hi <= lo:
                continue
            # Runs of non-noise queries: each is the tail of its key list.
            run_noise = (is_full if exact_output else is_noise)[lo:hi]
            edges = np.flatnonzero(np.diff(np.r_[1, run_noise.astype(np.int8), 1]))
            for start, stop in zip(edges[::2], edges[1::2], strict=True):
                if stop > start:
                    causal_q.append(np.arange(lo + start, lo + stop) - a + q_off)
                    causal_k.append(visible_keys(d0, lo + stop))
        for s0, s1 in spans:
            lo, hi = max(a, s0), min(b, s1)
            if hi <= lo:
                continue
            if exact_output:
                # Like a noise span: visible prefix plus the whole span, one call.
                d0 = docs_start[np.searchsorted(docs_end, s0, side="right")]
                keys = np.concatenate([visible_keys(d0, s0), buf_row[s0:s1]])
                noise_parts.append((np.arange(lo, hi) - a + q_off, keys))
            else:
                later_q.append(np.arange(hi - 1, lo - 1, -1) - a + q_off)
                later_k.append(buf_row[lo:s1][::-1].copy())
        shared: dict[int, list[np.ndarray]] = {}
        for d0, s0, s1 in noise_spans:
            lo, hi = max(a, s0), min(b, s1)
            if hi <= lo:
                continue
            q_rows = np.arange(lo, hi) - a + q_off
            if exact_output:
                keys = np.concatenate([visible_keys(d0, s0), buf_row[s0:s1]])
                noise_parts.append((q_rows, keys))
                continue
            # The document's keys before this chunk are shared by all its
            # spans here: one call over them; each span adds the rest.
            start = max(a, d0)
            shared.setdefault(d0, []).append(q_rows)
            keys = np.concatenate([visible_keys(start, s0), buf_row[s0:s1]])
            noise_parts.append((q_rows, keys))
        for d0, rows in shared.items():
            start = max(a, d0)
            if start > d0:
                prefix_q.append(np.concatenate(rows))
                prefix_k.append(visible_keys(d0, start))
        q_off += b - a
    assert all((k >= 0).all() for k in causal_k + later_k)
    assert all((k >= 0).all() for _, k in noise_parts)

    def bounded(q_parts, k_parts, **kind) -> list[DenseCall | None]:
        """Calls over ``(q, k)`` sequences, at most ``max_gathered_keys`` keys each."""
        out, gq, gk, total = [], [], [], 0
        for qp, kp in zip(q_parts, k_parts, strict=True):
            if gk and total + len(kp) > max_gathered_keys:
                out.append(_make_call(gq, gk, device=device, **kind))
                gq, gk, total = [], [], 0
            gq.append(qp)
            gk.append(kp)
            total += len(kp)
        out.append(_make_call(gq, gk, device=device, **kind))
        return out

    calls = [
        *bounded(causal_q, causal_k, causal=True, strict=False),
        *bounded(later_q, later_k, causal=False, strict=True),
        *bounded(
            [q for q, _ in noise_parts],
            [k for _, k in noise_parts],
            causal=False,
            strict=False,
        ),
        *bounded(prefix_q, prefix_k, causal=False, strict=False),
    ]
    calls = [c for c in calls if c is not None]
    use_fa3 = (
        allow_fa3
        and len(calls) == 1
        and calls[0].causal
        and _fa3_flash_attn_func(torch.device(device), varlen=True) is not None
    )
    return DenseGatherPlan(
        local_len=local,
        send_index=torch.from_numpy(local_row[np.concatenate(send_pos)]).to(device),
        send_counts=[len(p) for p in send_pos],
        recv_counts=[len(p) for p in recv_pos],
        calls=calls,
        use_fa3=use_fa3,
        # Noise queries can appear in both a shared-prefix call and a
        # span call even when the layout contains no full spans/later_q.
        rows_disjoint=exact_output or not ((later_q and causal_q) or prefix_q),
    )


def _fa3_fwd(call: DenseCall, q, k, v, scale) -> tuple[Tensor, Tensor]:
    # [MLR-02.08] flash-attn-3 is optional and Hopper-only.
    from flash_attn_interface import _flash_attn_forward

    out, lse, *_ = _flash_attn_forward(
        q if call.q_identity else q[call.q_index],
        k[call.k_index],
        v[call.k_index],
        cu_seqlens_q=call.cu_q,
        cu_seqlens_k=call.cu_k,
        max_seqlen_q=call.max_q,
        max_seqlen_k=call.max_k,
        softmax_scale=scale,
        causal=True,
    )
    return out, lse


def _fa3_bwd(call: DenseCall, q, k, v, out, dout, lse, scale):
    # [MLR-02.08] flash-attn-3 is optional and Hopper-only.
    from flash_attn_interface import _flash_attn_backward

    q_c = q if call.q_identity else q[call.q_index]
    k_c, v_c = k[call.k_index], v[call.k_index]
    dq, dk, dv = (torch.empty_like(t) for t in (q_c, k_c, v_c))
    _flash_attn_backward(
        dout if call.q_identity else dout[call.q_index],
        q_c, k_c, v_c,
        out if call.q_identity else out[call.q_index],
        lse if call.q_identity else lse[:, call.q_index].contiguous(),
        call.cu_q, call.cu_k, None, None, call.max_q, call.max_k,
        dq, dk, dv, scale, True,
    )  # fmt: skip
    return dq, dk, dv


def _amaxes(*xs: Tensor) -> Tensor:
    """``max|x|`` of each tensor as one fp32 vector, without a host sync."""
    return torch.stack(
        [
            torch.linalg.vector_norm(x, ord=float("inf")).float()
            if x.numel()
            else x.new_zeros((), dtype=torch.float32)
            for x in xs
        ]
    )


def _f16_factor(amax: float) -> float:
    """Power of two putting ``amax`` near 2**13 (1 for zero)."""
    if amax == 0.0:
        return 1.0
    return 2.0 ** max(min(math.floor(math.log2(2.0**13 / amax)), 60), -60)


def _scaled_f16(x: Tensor, f: float) -> Tensor:
    """``(x.float() * f).half()`` in one kernel, without the fp32 temporary.

    The product is computed in fp32; for a power-of-two ``f`` it is exact in
    ``x``'s dtype wherever fp16 can hold it, so the result is the same.
    """
    if f == 1.0:
        return x.half()
    return torch.mul(x, f, out=torch.empty_like(x, dtype=torch.float16))


def _to_f16(*xs: Tensor) -> list[tuple[Tensor, float]]:
    """``((x * f).half(), f)`` per tensor, ``f`` a power of two putting ``max|x|`` near 2**13.

    bf16 values convert to fp16 exactly in fp16's normal range, so scaled
    tensors keep their values while FA4 computes, and writes, in fp16 (three
    more mantissa bits than bf16 for partial outputs and gradients). One host
    sync covers all tensors.
    """
    peaks = _amaxes(*xs).tolist()
    return [(_scaled_f16(x, f), f) for x, f in zip(xs, map(_f16_factor, peaks))]


class _DenseGatherAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, group, plan: DenseGatherPlan, scale, regather):
        q2, k2, v2 = q[0], k[0], v[0]
        n, h = q2.shape[0], q2.shape[1]
        halo = _exchange(
            (k2[plan.send_index], v2[plan.send_index]),
            send_counts=plan.send_counts,
            recv_counts=plan.recv_counts,
            group=group,
        )
        k_buf = torch.cat([k2, halo[0]])
        v_buf = torch.cat([v2, halo[1]])
        if plan.use_fa3 and scale is None:
            scale = q2.shape[-1] ** -0.5
        fwd = _fa3_fwd if plan.use_fa3 else _call_fwd
        if plan.rows_disjoint:
            # Each row comes from one call: write outputs in place.
            out = torch.empty_like(q2)
            acc_lse = torch.empty((h, n), device=q.device, dtype=torch.float32)
            for call in plan.calls:
                o, lse = fwd(call, q2, k_buf, v_buf, scale)
                if call.q_identity:
                    out[: o.shape[0]] = o
                    acc_lse[:, : o.shape[0]] = lse
                else:
                    out.index_copy_(0, call.q_index, o)
                    acc_lse.index_copy_(1, call.q_index, lse)
        else:
            # Rows merge partials: run the forward calls in fp16 (exactly
            # converted, power-of-two scaled; softmax probabilities and outputs
            # cannot overflow), so partials keep fp16 precision and the merged
            # output rounds to bf16 once, as one FA4 call would.
            (q16, fq), (k16, fk), (v16, fv) = _to_f16(q2, k_buf, v_buf)
            scale16 = (q2.shape[-1] ** -0.5 if scale is None else scale) / (fq * fk)
            acc_out: Tensor | None = None
            acc_lse = None
            for call in plan.calls:
                o, lse = fwd(call, q16, k16, v16, scale16)
                full = call.q_identity and call.q_index.numel() == n
                if acc_out is None and full:
                    acc_out, acc_lse = o.float(), lse.contiguous()
                    continue
                if acc_out is None:
                    acc_out = torch.zeros(
                        n, h, q2.shape[2], device=q.device, dtype=torch.float32
                    )
                    acc_lse = torch.full(
                        (h, n), float("-inf"), device=q.device, dtype=torch.float32
                    )
                merge_into(acc_out, acc_lse, None if full else call.q_index, o, lse)
            out = acc_out.div_(fv).to(q.dtype)
        if regather:
            # Keep only the local KV; the backward exchanges the halo again.
            ctx.save_for_backward(q2, k2, v2, out, acc_lse)
        else:
            ctx.save_for_backward(q2, k_buf, v_buf, out, acc_lse)
        ctx.group, ctx.plan, ctx.scale, ctx.regather = group, plan, scale, regather
        return out.unsqueeze(0)

    @staticmethod
    def backward(ctx, dout):
        q2, k_buf, v_buf, out, lse = ctx.saved_tensors
        plan: DenseGatherPlan = ctx.plan
        if ctx.regather:
            halo = _exchange(
                (k_buf[plan.send_index], v_buf[plan.send_index]),
                send_counts=plan.send_counts,
                recv_counts=plan.recv_counts,
                group=ctx.group,
            )
            k_buf = torch.cat([k_buf, halo[0]])
            v_buf = torch.cat([v_buf, halo[1]])
        dout2 = dout[0].contiguous()
        disjoint = plan.rows_disjoint
        # The backward runs on the bf16 inputs (FA4 casts dS to the input
        # dtype, which fp16 scaling could overflow); its fp32 gradient
        # capture is already exact. Only the forward's partials needed fp16.
        dq = (
            torch.empty_like(q2)
            if disjoint
            else torch.zeros_like(q2, dtype=torch.float32)
        )
        dk_buf = torch.zeros_like(k_buf, dtype=torch.float32)
        dv_buf = torch.zeros_like(v_buf, dtype=torch.float32)
        for call in plan.calls:
            if plan.use_fa3:
                # dK / dV come back rounded (see ``allow_fa3``).
                g_q, g_k, g_v = _fa3_bwd(
                    call, q2, k_buf, v_buf, out, dout2, lse, ctx.scale
                )
                s_q = s_k = s_v = 1.0
            else:
                # Gradients FA4 accumulates in fp32 (dq; dK / dV with GQA) come
                # back in scaled fp16 instead of bf16; MHA dK / dV are FA4's
                # bf16 output. dq rows come from one call when rows are disjoint.
                with fp32_grad_outputs(skip_dq=disjoint) as exact:
                    grads = _call_bwd(
                        call, q2, k_buf, v_buf, out, dout2, lse, ctx.scale
                    )
                (g_q, s_q), (g_k, s_k), (g_v, s_v) = (
                    as_scaled(exact, g) for g in grads
                )
            if disjoint:
                if call.q_identity:
                    dq[: g_q.shape[0]] = g_q
                else:
                    dq.index_copy_(0, call.q_index, g_q.to(dq.dtype))
            else:
                accumulate_rows(
                    dq, None if call.q_identity else call.q_index, g_q,
                    s_q,
                )  # fmt: skip
            k_rows = None if call.k_identity else call.k_index
            accumulate_rows(dk_buf, k_rows, g_k, s_k)
            accumulate_rows(dv_buf, k_rows, g_v, s_v)
        n = plan.local_len
        back = _exchange(
            (dk_buf[n:], dv_buf[n:]),
            send_counts=plan.recv_counts,
            recv_counts=plan.send_counts,
            group=ctx.group,
        )
        dk, dv = dk_buf[:n], dv_buf[:n]
        dk.index_add_(0, plan.send_index, back[0])
        dv.index_add_(0, plan.send_index, back[1])
        return (
            dq.to(dout.dtype).unsqueeze(0),
            dk.to(dout.dtype).unsqueeze(0),
            dv.to(dout.dtype).unsqueeze(0),
            None,
            None,
            None,
            None,
        )


def dense_gather_attention(
    q: Float[Tensor, " batch local_tokens num_heads head_dim"],
    k: Float[Tensor, " batch local_tokens kv_heads head_dim"],
    v: Float[Tensor, " batch local_tokens kv_heads head_dim"],
    *,
    group: dist.ProcessGroup,
    plan: DenseGatherPlan,
    softmax_scale: float | None = None,
    regather: bool = False,
) -> Float[Tensor, " batch local_tokens num_heads head_dim"]:
    """Attention of a ring rank's chunks over its local KV and gathered halo.

    Args:
        q: ``(batch, local_len, heads, head_dim)`` local queries; every
            sequence shares the plan's layout.
        k: ``(batch, local_len, kv_heads, head_dim)`` local keys.
        v: Values, same shape as ``k``.
        group: Ring process group.
        plan: From :func:`build_dense_gather_plan` for this rank.
        softmax_scale: Defaults to ``1 / sqrt(head_dim)``.
        regather: Save only the local KV for the backward and exchange the halo
            again there, instead of keeping the gathered KV per layer.

    Returns:
        ``(batch, local_len, heads, head_dim)`` attention output.
    """
    return torch.cat(
        [
            _DenseGatherAttention.apply(
                q[i : i + 1],
                k[i : i + 1],
                v[i : i + 1],
                group,
                plan,
                softmax_scale,
                regather,
            )
            for i in range(q.shape[0])
        ]
    )
