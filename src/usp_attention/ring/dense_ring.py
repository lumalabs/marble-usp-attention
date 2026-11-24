# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Ring attention for packed multimodal masks on dense varlen FA4 calls.

With contiguous ring shards, every ring step of a packed multimodal mask
(see :mod:`usp_attention.fa4.interval_mask`) splits into dense varlen calls:

- the step against a rank's own keys is the two-call interval decomposition
  of the shard's slice of the layout (causal within each document piece, plus
  the later keys inside each full / noise piece, reversed and strictly causal);
- a step against an earlier shard is full attention from each document's
  queries here to the same document's keys there;
- a step against a later shard is full attention from each full / noise
  split's queries here to the same split's keys there.

Every call gathers its query and key rows into packed varlen order, so FA4
runs its dense kernels: on Blackwell that keeps the 2-CTA backward and the
dense schedulers, which block sparsity turns off. Step results merge by
log-sum-exp in fp32; the backward adds every call's dq, dK and dV into fp32
accumulators (one dK / dV accumulator per visited key shard) and
reduce-scatters the dK / dV once at the end, so ranks do not wait on each
other between steps. Each call's gradients are FA4's outputs in the input
dtype, so they are rounded to bf16 per call before the fp32 sum; only the sums
round once. Contiguous shards are balanced enough for small rings on packed
masks (rank work within about 10% at ring 2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch import Tensor

from usp_attention.ring.dense_merge import accumulate_rows, merge_into

if TYPE_CHECKING:
    from jaxtyping import Float, Int


@dataclass(kw_only=True, slots=True)
class DenseCall:
    """One dense varlen FA4 call of a ring step, on gathered rows.

    Attributes:
        q_index: Local query rows in packed varlen order (int64).
        k_index: Key rows of the visiting shard in packed varlen order (int64).
        cu_q: Query sequence boundaries within ``q_index`` (int32).
        cu_k: Key sequence boundaries within ``k_index`` (int32).
        max_q: Longest query sequence.
        max_k: Longest key sequence.
        causal: Causal (aligned bottom-right) instead of full attention.
        strict: Strictly causal (diagonal excluded); implies a causal window.
        q_identity: ``q_index`` is ``0, 1, ..., len - 1`` (no gather needed).
        k_identity: ``k_index`` is ``0, 1, ..., len - 1``.
        anti: Anti-causal: each query sees only the keys after it (FA4 local
            window with left bound -1); ``q_index`` / ``k_index`` are then in
            natural order.
        q_starts: In-place query sequences: start offset of each sequence in
            the local rows, then the end of the last (int32), with
            ``q_used`` lengths. ``None`` gathers ``q_index`` instead.
        q_used: Query sequence lengths for ``q_starts``.
        q_mask: Local rows covered by the in-place sequences (bool).
        k_inplace: Keys use the same in-place sequences as the queries.
    """

    q_index: Tensor
    k_index: Tensor
    cu_q: Tensor
    cu_k: Tensor
    max_q: int
    max_k: int
    causal: bool = False
    strict: bool = False
    q_identity: bool = False
    k_identity: bool = False
    anti: bool = False
    q_starts: Tensor | None = None
    q_used: Tensor | None = None
    q_mask: Tensor | None = None
    k_inplace: bool = False


@dataclass(kw_only=True, slots=True)
class DenseRingPlan:
    """Per-rank dense calls for every ring step.

    Attributes:
        ring_rank: This rank's position in the ring.
        ring_size: Ring degree.
        local_len: Tokens per ring shard.
        steps: ``steps[s]`` are the calls against the shard of ring rank
            ``(ring_rank - s) % ring_size``.
        chunks: Global token ranges per ring rank, in local order.
        dkv_overlap: Send remote gradient contributions after each backward step.
            Captured in the plan so backward uses the same execution options.
    """

    ring_rank: int
    ring_size: int
    local_len: int
    steps: list[list[DenseCall]]
    chunks: list[list[tuple[int, int]]]
    dkv_overlap: bool = False


def _segments(document_lens, split_lens, attn_modes):
    """Yield ``(doc_start, doc_end, split_start, split_end, mode)`` per split."""
    split_iter = iter(zip(split_lens, attn_modes, strict=True))
    offset = 0
    for doc_len in document_lens:
        end = offset + doc_len
        pos = offset
        while pos < end:
            length, mode = next(split_iter)
            yield offset, end, pos, pos + length, mode
            pos += length
        offset = end


def dense_ring_supported(
    document_lens: list[int], split_lens: list[int], attn_modes: list[str]
) -> bool:
    """Check whether the layout supports the interval dense ring.

    Args:
        document_lens: Lengths of the packed documents, in tokens.
        split_lens: Lengths of the attention spans, in document order.
        attn_modes: Attention mode for each span; must match split_lens.

    Returns:
        True when no span is a vision group and each noise span ends its
        document.
    """
    return all(
        mode != "group" and (mode != "noise" or s_end == d_end)
        for _, d_end, _, s_end, mode in _segments(document_lens, split_lens, attn_modes)
    )


def _call(
    pairs, *, causal=False, strict=False, reverse=False, anti=False, device
) -> DenseCall | None:
    """Build a call from ``[(q_lo, q_hi, k_lo, k_hi), ...]`` local ranges."""
    pairs = [p for p in pairs if p[1] > p[0] and p[3] > p[2]]
    if not pairs:
        return None
    if reverse:
        q_index = torch.cat(
            [torch.arange(hi - 1, lo - 1, -1) for lo, hi, _, _ in pairs]
        )
        k_index = torch.cat(
            [torch.arange(hi - 1, lo - 1, -1) for _, _, lo, hi in pairs]
        )
    else:
        q_index = torch.cat([torch.arange(lo, hi) for lo, hi, _, _ in pairs])
        k_index = torch.cat([torch.arange(lo, hi) for _, _, lo, hi in pairs])
    q_lens = [hi - lo for lo, hi, _, _ in pairs]
    k_lens = [hi - lo for _, _, lo, hi in pairs]
    cu = (
        lambda lens: torch.tensor([0, *lens], dtype=torch.int64)
        .cumsum(0)
        .to(device=device, dtype=torch.int32)
    )  # noqa: E731
    return DenseCall(
        q_index=q_index.to(device),
        k_index=k_index.to(device),
        cu_q=cu(q_lens),
        cu_k=cu(k_lens),
        max_q=max(q_lens),
        max_k=max(k_lens),
        causal=causal,
        strict=strict,
        anti=anti,
        q_identity=torch.equal(q_index, torch.arange(q_index.numel())),
        k_identity=torch.equal(k_index, torch.arange(k_index.numel())),
    )


def _with_inplace_q(
    call: DenseCall, ranges, local: int, *, k_inplace: bool
) -> DenseCall:
    """Run ``call`` on the local query rows in place (sorted, disjoint ``ranges``).

    Opt-in (``USPConfig.dense_ring_inplace=True``): on H100 the full-size merges and
    masks it needs cost more than the gathers it saves (Qwen2.5-7B layout, 256k, USP 2x4:
    158 vs 152 ms), so gathering stays the default.
    """
    device = call.cu_q.device
    starts = [lo for lo, _ in ranges] + [ranges[-1][1]]
    used = [hi - lo for lo, hi in ranges]
    mask = torch.zeros(local, dtype=torch.bool)
    for lo, hi in ranges:
        mask[lo:hi] = True
    call.q_starts = torch.tensor(starts, dtype=torch.int32, device=device)
    call.q_used = torch.tensor(used, dtype=torch.int32, device=device)
    call.q_mask = mask.to(device)
    call.k_inplace = k_inplace
    return call


def _full_call(entries, *, device) -> DenseCall:
    """Full attention; each entry is ``(q_lo, q_hi, [(k_lo, k_hi), ...])``."""
    q_index = torch.cat([torch.arange(lo, hi) for lo, hi, _ in entries])
    k_index = torch.cat([torch.arange(lo, hi) for _, _, ks in entries for lo, hi in ks])
    q_lens = [hi - lo for lo, hi, _ in entries]
    k_lens = [sum(hi - lo for lo, hi in ks) for _, _, ks in entries]

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
    )


def _full_calls(pairs, *, device, local: int, inplace: bool = False) -> list[DenseCall]:
    """Full-attention calls for ``(q_lo, q_hi, k_lo, k_hi)`` pairs.

    Pairs that share a query range become one sequence over all their keys.
    One call's results are merged row by row, so query ranges that still
    overlap (a split's rows seeing a later chunk, and its document's rows an
    earlier one) go to separate calls.
    """
    by_q: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for q_lo, q_hi, k_lo, k_hi in pairs:
        if q_hi > q_lo and k_hi > k_lo:
            by_q.setdefault((q_lo, q_hi), []).append((k_lo, k_hi))
    groups: list[list[tuple[int, int, list]]] = []
    for (q_lo, q_hi), ks in sorted(by_q.items()):
        for group in groups:
            if all(q_hi <= g[0] or q_lo >= g[1] for g in group):
                group.append((q_lo, q_hi, ks))
                break
        else:
            groups.append([(q_lo, q_hi, ks)])
    return [
        _with_inplace_q(
            _full_call(group, device=device),
            [(lo, hi) for lo, hi, _ in group],
            local,
            k_inplace=False,
        )
        if inplace
        else _full_call(group, device=device)
        for group in groups
    ]


def chunk_assignment(
    document_lens: list[int],
    split_lens: list[int],
    attn_modes: list[str],
    *,
    ring_size: int,
    chunks_per_rank: int,
    counts: Int[Tensor, " tokens"] | None = None,
) -> list[list[tuple[int, int]]]:
    """Global token ranges held by each ring rank.

    With one chunk per rank the shards are contiguous. With more, the sequence
    is cut into ``ring_size * chunks_per_rank`` equal chunks, dealt to ranks in
    snake order by visible-pair count (heaviest first), so every rank holds the
    same number of tokens and nearly the same work; each rank keeps its chunks
    in sequence order. ``counts`` gives each token's visible keys when the
    tokens are stored in another order than the layout's (default: the layout's
    own counts).

    Args:
        document_lens: Lengths of the packed documents, in tokens.
        split_lens: Lengths of the attention spans, in document order.
        attn_modes: Attention mode for each span; must match split_lens.
        ring_size: Number of ring ranks.
        chunks_per_rank: Number of equal-length chunks assigned to each rank.
        counts: Optional per-token visible-key counts in stored token order;
            None computes them from the layout.

    Returns:
        Per-rank lists of half-open global token ranges, sorted by position.
        Each rank owns the same number of tokens.
    """
    seq_len = sum(document_lens)
    n = ring_size * chunks_per_rank
    size = seq_len // n
    ranges = [(i * size, (i + 1) * size) for i in range(n)]
    if chunks_per_rank == 1:
        return [[rg] for rg in ranges]
    # [MLR-02.08] local import keeps this module importable without the reference helpers.
    from usp_attention.reference import visible_key_counts

    if counts is None:
        counts = visible_key_counts(document_lens, split_lens, attn_modes)
    work = counts.view(n, size).sum(1)
    order = sorted(range(n), key=lambda i: -int(work[i]))
    owners: list[list[int]] = [[] for _ in range(ring_size)]
    for pos, chunk in enumerate(order):
        lap, idx = divmod(pos, ring_size)
        owners[idx if lap % 2 == 0 else ring_size - 1 - idx].append(chunk)
    return [[ranges[c] for c in sorted(chunks)] for chunks in owners]


def build_dense_ring_plan(
    document_lens: list[int],
    split_lens: list[int],
    attn_modes: list[str],
    *,
    ring_rank: int,
    ring_size: int,
    device: torch.device | str,
    chunks: list[list[tuple[int, int]]] | None = None,
    inplace: bool = False,
    dkv_overlap: bool = False,
) -> DenseRingPlan:
    """Dense calls for every ring step of ``ring_rank``.

    Args:
        document_lens: Lengths of the packed documents, in tokens.
        split_lens: Lengths of the attention spans, in document order.
        attn_modes: Attention mode for each span; must match split_lens.
        ring_rank: Index of the calling rank within the ring.
        ring_size: Number of ranks in the ring.
        device: Device on which to allocate plan tensors.
        chunks: Global token ranges per ring rank, in local order (from
            :func:`chunk_assignment`); contiguous shards when ``None``.
        inplace: Attend on query rows in place instead of gathering them.
        dkv_overlap: Overlap backward gradient transfers with subsequent steps.

    Returns:
        A DenseRingPlan containing per-step calls and the requested execution
        options for the calling rank.
    """
    seq_len = sum(document_lens)
    local = seq_len // ring_size
    if chunks is None:
        chunks = [[(r * local, (r + 1) * local)] for r in range(ring_size)]
    segs = list(_segments(document_lens, split_lens, attn_modes))
    docs = sorted({(d0, d1) for d0, d1, _, _, _ in segs})
    spans = [(s0, s1) for _, _, s0, s1, m in segs if m != "causal"]

    def with_offsets(ranges):
        out, off = [], 0
        for lo, hi in ranges:
            out.append((lo, hi, off))
            off += hi - lo
        return out

    def clip(lo, hi, chunk):
        a, b, off = chunk
        lo, hi = max(lo, a), min(hi, b)
        return (lo - a + off, hi - a + off) if hi > lo else (0, 0)

    r = ring_rank
    mine = with_offsets(chunks[r])
    steps: list[list[DenseCall]] = []
    for s in range(ring_size):
        j = (r - s) % ring_size
        theirs = with_offsets(chunks[j])
        causal_pairs, strict_pairs, full_pairs = [], [], []
        for qc in mine:
            for kc in theirs:
                if (qc[0], qc[1]) == (kc[0], kc[1]):
                    # Same chunk: the two-call interval decomposition.
                    for d0, d1 in docs:
                        lo, hi = clip(d0, d1, qc)
                        causal_pairs.append((lo, hi, lo, hi))
                    for s0, s1 in spans:
                        lo, hi = clip(s0, s1, qc)
                        strict_pairs.append((lo, hi, lo, hi))
                elif kc[1] <= qc[0]:
                    # Earlier keys: all of each shared document.
                    for d0, d1 in docs:
                        full_pairs.append((*clip(d0, d1, qc), *clip(d0, d1, kc)))
                else:
                    # Later keys: the rest of each full / noise split.
                    for s0, s1 in spans:
                        full_pairs.append((*clip(s0, s1, qc), *clip(s0, s1, kc)))
        spans_here = [(p[0], p[1]) for p in strict_pairs if p[1] > p[0]]
        if not inplace or not spans_here:
            later = _call(strict_pairs, strict=True, reverse=True, device=device)
        else:
            # Keys after each query inside its split: anti-causal, in place.
            later = _with_inplace_q(
                _call(strict_pairs, anti=True, device=device),
                spans_here,
                local,
                k_inplace=True,
            )
        calls = [
            _call(causal_pairs, causal=True, device=device),
            later,
            *_full_calls(full_pairs, device=device, local=local, inplace=inplace),
        ]
        steps.append([c for c in calls if c is not None])
    return DenseRingPlan(
        ring_rank=r,
        ring_size=ring_size,
        local_len=local,
        steps=steps,
        chunks=chunks,
        dkv_overlap=dkv_overlap,
    )


def rank_work(
    document_lens: list[int],
    split_lens: list[int],
    attn_modes: list[str],
    ring_size: int,
    chunks: list[list[tuple[int, int]]] | None = None,
    counts: Int[Tensor, " tokens"] | None = None,
) -> list[int]:
    """Visible (query, key) pairs per ring rank (contiguous shards by default).

    ``counts`` gives each stored token's visible keys, as in
    :func:`chunk_assignment`.

    Args:
        document_lens: Lengths of the packed documents, in tokens.
        split_lens: Lengths of the attention spans, in document order.
        attn_modes: Attention mode for each span; must match split_lens.
        ring_size: Number of ring ranks.
        chunks: Optional per-rank token ranges; None uses contiguous equal
            shards.
        counts: Optional visible-key counts in stored token order.

    Returns:
        A list of visible query/key pair counts, one per ring rank.
    """
    # [MLR-02.08] local import keeps this module importable without the reference helpers.
    from usp_attention.reference import visible_key_counts

    if counts is None:
        counts = visible_key_counts(document_lens, split_lens, attn_modes)
    if chunks is None:
        return [int(c.sum()) for c in counts.chunk(ring_size)]
    return [sum(int(counts[lo:hi].sum()) for lo, hi in rank) for rank in chunks]


# ---------------------------------------------------------------------------
# Execution.


def _fa4():
    # [MLR-02.08] flash_attn.cute is an optional, platform-specific dependency.
    from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd

    return _flash_attn_fwd, _flash_attn_bwd


def _rows(x: Tensor, index: Tensor, identity: bool) -> Tensor:
    """``x[index]``, as a view when the index is a leading range."""
    return x[: index.numel()] if identity else x[index]


def _window(call: DenseCall) -> dict:
    if call.anti:
        return {"window_size_left": -1, "window_size_right": None}
    return {"window_size_right": -1 if call.strict else None}


def _kv_args(call: DenseCall, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor, dict]:
    if call.k_inplace:
        return (
            k,
            v,
            {
                "cu_seqlens_k": call.q_starts,
                "seqused_k": call.q_used,
                "max_seqlen_k": call.max_k,
            },
        )
    ki = call.k_identity
    return (
        _rows(k, call.k_index, ki),
        _rows(v, call.k_index, ki),
        {"cu_seqlens_k": call.cu_k, "max_seqlen_k": call.max_k},
    )


def _call_fwd(
    call: DenseCall, q: Tensor, k: Tensor, v: Tensor, scale
) -> tuple[Tensor, Tensor]:
    fwd, _ = _fa4()
    if call.q_starts is not None:
        # In place: rows outside the sequences keep out = 0, lse = -inf.
        k_arg, v_arg, kv_kw = _kv_args(call, k, v)
        out = torch.zeros_like(q)
        lse = torch.full(
            (q.shape[1], q.shape[0]),
            float("-inf"),
            device=q.device,
            dtype=torch.float32,
        )
        fwd(
            q, k_arg, v_arg,
            cu_seqlens_q=call.q_starts, seqused_q=call.q_used, max_seqlen_q=call.max_q,
            softmax_scale=scale, causal=call.causal, return_lse=True, out=out, lse=lse,
            **kv_kw, **_window(call),
        )  # fmt: skip
        return out, lse
    out, lse, *_ = fwd(
        _rows(q, call.q_index, call.q_identity),
        _rows(k, call.k_index, call.k_identity),
        _rows(v, call.k_index, call.k_identity),
        cu_seqlens_q=call.cu_q, cu_seqlens_k=call.cu_k,
        max_seqlen_q=call.max_q, max_seqlen_k=call.max_k,
        softmax_scale=scale, causal=call.causal,
        return_lse=True,
        **_window(call),
    )  # fmt: skip
    return out, lse


def _call_bwd(call, q, k, v, out, dout, lse, scale):
    _, bwd = _fa4()
    if call.q_starts is not None:
        k_arg, v_arg, kv_kw = _kv_args(call, k, v)
        g_q, g_k, g_v = bwd(
            q, k_arg, v_arg, out, dout, lse, scale, call.causal,
            cu_seqlens_q=call.q_starts, seqused_q=call.q_used, max_seqlen_q=call.max_q,
            **kv_kw, **_window(call),
        )  # fmt: skip
        # Rows outside the sequences are left uninitialized by FA4.
        keep = call.q_mask[:, None, None]
        g_q = torch.where(keep, g_q, 0)
        if call.k_inplace:
            g_k, g_v = torch.where(keep, g_k, 0), torch.where(keep, g_v, 0)
        return g_q, g_k, g_v
    qi, ki = call.q_identity, call.k_identity
    return bwd(
        _rows(q, call.q_index, qi), _rows(k, call.k_index, ki), _rows(v, call.k_index, ki),
        _rows(out, call.q_index, qi), _rows(dout, call.q_index, qi),
        lse[:, : call.q_index.numel()] if qi else lse[:, call.q_index].contiguous(),
        scale, call.causal,
        cu_seqlens_q=call.cu_q, cu_seqlens_k=call.cu_k,
        max_seqlen_q=call.max_q, max_seqlen_k=call.max_k,
        **_window(call),
    )  # fmt: skip


class _Rotation:
    """Asynchronous send to ring rank ``rank + shift``, receive from ``rank - shift``."""

    def __init__(
        self, tensors: list[Tensor], group, rank: int, size: int, shift: int = 1
    ):
        self._recv = [torch.empty_like(t) for t in tensors]
        dst = dist.get_global_rank(group, (rank + shift) % size)
        src = dist.get_global_rank(group, (rank - shift) % size)
        ops = [dist.P2POp(dist.isend, t, dst, group) for t in tensors]
        ops += [dist.P2POp(dist.irecv, t, src, group) for t in self._recv]
        self._works = dist.batch_isend_irecv(ops)

    def wait(self) -> list[Tensor]:
        for work in self._works:
            work.wait()
        return self._recv


class _DenseRingAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, group, plan: DenseRingPlan, scale):
        # (B=1) local shards, (S_local, H, D) after squeezing the batch.
        # KV travels by NCCL point-to-point, which needs dense tensors (a
        # model may pass views of a fused projection).
        q2, k2, v2 = q[0], k[0].contiguous(), v[0].contiguous()
        n, h = q2.shape[0], q2.shape[1]
        acc_out: Tensor | None = None
        acc_lse: Tensor | None = None
        kv = [k2, v2]
        for s, calls in enumerate(plan.steps):
            # The next block moves while this step computes.
            nxt = None
            if s + 1 < plan.ring_size:
                nxt = _Rotation(kv, group, plan.ring_rank, plan.ring_size)
            for call in calls:
                out, lse = _call_fwd(call, q2, kv[0], kv[1], scale)
                full = call.q_starts is not None or (
                    call.q_identity and call.q_index.numel() == n
                )
                if acc_out is None and full and call.q_starts is None:
                    # A call over every row in order starts the accumulator.
                    acc_out, acc_lse = out.float(), lse.contiguous()
                    continue
                if acc_out is None:
                    acc_out = torch.zeros(
                        n, h, q2.shape[2], device=q.device, dtype=torch.float32
                    )
                    acc_lse = torch.full(
                        (h, n), float("-inf"), device=q.device, dtype=torch.float32
                    )
                merge_into(acc_out, acc_lse, None if full else call.q_index, out, lse)
            if nxt is not None:
                kv = nxt.wait()
        out = acc_out.to(q.dtype)
        # The last step's KV starts the backward, which runs the steps in reverse.
        ctx.save_for_backward(q2, k2, v2, out, acc_lse, kv[0], kv[1])
        ctx.group, ctx.plan, ctx.scale = group, plan, scale
        return out.unsqueeze(0)

    @staticmethod
    def backward(ctx, dout):
        q2, k2, v2, out, lse, k_last, v_last = ctx.saved_tensors
        group, plan, scale = ctx.group, ctx.plan, ctx.scale
        dout2 = dout[0].contiguous()
        dq = torch.zeros_like(q2, dtype=torch.float32)
        size = plan.ring_size
        # Steps run last to first, starting from the KV the forward ended with.
        # By default every owner's dK / dV accumulates locally and is
        # reduce-scattered once at the end. dense_ring_dkv_overlap sends
        # each remote owner's share as soon as its step is done instead; that
        # hides the reduce-scatter but its NCCL kernel then shares SMs with the
        # remaining backward steps (slower on H100 at 1M tokens).
        overlap = plan.dkv_overlap
        if overlap:
            dk = torch.zeros_like(k2, dtype=torch.float32)
            dv = torch.zeros_like(v2, dtype=torch.float32)
        else:
            shape = (size, *k2.shape)
            dk_all = torch.zeros(shape, device=k2.device, dtype=torch.float32)
            dv_all = torch.zeros_like(dk_all)
        kv = [k_last, v_last]
        sent: list[_Rotation] = []
        for s in reversed(range(size)):
            nxt = None
            if s > 0:
                nxt = _Rotation(kv, group, plan.ring_rank, size, shift=-1)
            if not overlap:
                owner = (plan.ring_rank - s) % size
                g_k_acc, g_v_acc = dk_all[owner], dv_all[owner]
            elif s == 0:
                g_k_acc, g_v_acc = dk, dv
            else:
                g_k_acc = torch.zeros_like(dk)
                g_v_acc = torch.zeros_like(dv)
            for call in plan.steps[s]:
                g_q, g_k, g_v = _call_bwd(
                    call, q2, kv[0], kv[1], out, dout2, lse, scale
                )
                q_rows = (
                    None
                    if call.q_starts is not None or call.q_identity
                    else call.q_index
                )
                k_rows = None if call.k_inplace or call.k_identity else call.k_index
                accumulate_rows(dq, q_rows, g_q)
                accumulate_rows(g_k_acc, k_rows, g_k)
                accumulate_rows(g_v_acc, k_rows, g_v)
            if overlap and s > 0:
                # Step s used the KV of ring rank (rank - s); that rank sends
                # its contribution for this rank's KV at its own step s.
                sent.append(
                    _Rotation([g_k_acc, g_v_acc], group, plan.ring_rank, size, -s)
                )
            if nxt is not None:
                kv = nxt.wait()
        if overlap:
            for exchange in sent:
                r_k, r_v = exchange.wait()
                dk += r_k
                dv += r_v
        else:
            dk = torch.empty_like(k2, dtype=torch.float32)
            dv = torch.empty_like(v2, dtype=torch.float32)
            dist.reduce_scatter_tensor(dk, dk_all.flatten(0, 1), group=group)
            dist.reduce_scatter_tensor(dv, dv_all.flatten(0, 1), group=group)
        return (
            dq.to(q2.dtype).unsqueeze(0),
            dk.to(k2.dtype).unsqueeze(0),
            dv.to(v2.dtype).unsqueeze(0),
            None,
            None,
            None,
        )


def dense_ring_attention(
    q: Float[Tensor, " batch local_tokens num_heads head_dim"],
    k: Float[Tensor, " batch local_tokens kv_heads head_dim"],
    v: Float[Tensor, " batch local_tokens kv_heads head_dim"],
    *,
    group: dist.ProcessGroup,
    plan: DenseRingPlan,
    softmax_scale: float | None = None,
) -> Float[Tensor, " batch local_tokens num_heads head_dim"]:
    """Ring attention over contiguous shards with dense varlen FA4 steps.

    Args:
        q: ``(batch, local_len, heads, head_dim)`` local query shard; every
            sequence shares the plan's layout.
        k: ``(batch, local_len, kv_heads, head_dim)`` local key shard.
        v: Values, same shape as ``k``.
        group: Ring process group.
        plan: From :func:`build_dense_ring_plan` for this rank.
        softmax_scale: Defaults to ``1 / sqrt(head_dim)``.

    Returns:
        ``(batch, local_len, heads, head_dim)`` attention output.
    """
    if q.shape[0] == 1:
        return _DenseRingAttention.apply(q, k, v, group, plan, softmax_scale)
    return torch.cat(
        [
            _DenseRingAttention.apply(
                q[i : i + 1], k[i : i + 1], v[i : i + 1], group, plan, softmax_scale
            )
            for i in range(q.shape[0])
        ]
    )
