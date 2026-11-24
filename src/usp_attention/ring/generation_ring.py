# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Exact ring attention on dense varlen calls, for any packed mask without groups.

Ring ranks hold contiguous chunks balanced at every ring step
(:func:`ring_chunk_assignment`);
every ring step attends the visiting KV shard with dense varlen FA4 calls, so
FA4 runs its dense kernels (on H100 1.36x, on B300 about 1.6x faster than
block sparsity on long generation masks), while the ring keeps one remote KV
shard in flight (bounded memory) and overlaps its rotation with compute.
Masks may place noise spans anywhere in a document (BAGEL-style interleaved
generation: each frame's noised latents, then its clean copy):

- shard before the queries' chunk: every query of a document attends that
  document's non-noise keys in the shard, one full-attention sequence per
  document (no duplicated keys); queries of a noise span that crosses into
  the shard also attend their span's keys there;
- the queries' own chunk: causal over the document's non-noise tokens,
  reversed strictly causal later keys inside full spans, and each noise span
  over the chunk's non-noise prefix plus its own tokens;
- shard after the chunk: the rest of each full / noise span crossing into it.

Numerics: the forward runs in fp16 (bf16 converts exactly after power-of-two
scaling, with scales shared by the ring group) and merges partials in fp32,
so the output is at least as accurate as one bf16 FA4 call. The backward runs
in bf16, adds every call's gradients into fp32 accumulators, and rotates each
shard's fp32 dK / dV accumulator around the ring with its KV, so it arrives
home complete and memory stays at one shard. Gradients FA4 accumulates in fp32
(dq, and dK / dV with grouped-query attention) come out of each call in fp16
with a power-of-two scale, three mantissa bits more than bf16
(:func:`~usp_attention.ring.fp32_grad.fp32_grad_outputs`); without GQA FA4
writes each call's dK / dV directly in bf16, so those are rounded per call
before the fp32 sum (as is every gradient of FA4's head_dim-256 backward on
Blackwell).
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor

from usp_attention.ring.dense_gather import (
    _amaxes,
    _f16_factor,
    _make_call,
    _scaled_f16,
)
from usp_attention.ring.dense_merge import accumulate_rows, merge_into
from usp_attention.ring.dense_ring import (
    DenseCall,
    _call_bwd,
    _call_fwd,
    _Rotation,
    _segments,
)
from usp_attention.ring.fp32_grad import as_scaled, fp32_grad_outputs

if TYPE_CHECKING:
    from jaxtyping import Float, Int


def _visible_spans(document_lens, split_lens, attn_modes):
    """Per query token: document start, end of its non-noise key range, and
    (noise queries) its own span, plus the non-noise key mask."""
    seq_len = sum(document_lens)
    doc0 = np.empty(seq_len, np.int64)
    end = np.arange(1, seq_len + 1, dtype=np.int64)
    span_lo = np.zeros(seq_len, np.int64)
    span_hi = np.zeros(seq_len, np.int64)
    is_noise = np.zeros(seq_len, bool)
    for d0, d1, s0, s1, mode in _segments(document_lens, split_lens, attn_modes):
        doc0[s0:s1] = d0
        if mode == "full":
            end[s0:s1] = s1
        elif mode == "noise":
            end[s0:s1] = s0
            span_lo[s0:s1], span_hi[s0:s1] = s0, s1
            is_noise[s0:s1] = True
    return doc0, end, span_lo, span_hi, is_noise


def chunk_pair_counts(
    document_lens: list[int],
    split_lens: list[int],
    attn_modes: list[str],
    *,
    n_chunks: int,
    samples_per_chunk: int = 64,
) -> Int[np.ndarray, " num_chunks num_chunks"]:
    """Estimate visible query/key pair counts between equal chunks.

    The sequence is cut into ``n_chunks`` equal chunks. About
    ``samples_per_chunk`` evenly spaced queries stand for each chunk's queries.
    A query sees at most two key intervals, its document's non-noise keys up to
    its own end and, for a noise query, its own span, so its count against a
    key chunk is the overlap with that chunk: two partial chunks plus a run of
    whole chunks, accumulated with a difference array in O(samples +
    n_chunks^2). Counts are integers, so every rank that derives a layout from
    them agrees on it exactly.

    Args:
        document_lens: Lengths of the packed documents, in tokens.
        split_lens: Lengths of the attention spans, in document order.
        attn_modes: Attention mode for each span; must match split_lens.
        n_chunks: Number of equal-length chunks dividing the packed sequence.
        samples_per_chunk: Target number of evenly spaced query samples per
            chunk; every query is counted when the chunk is small enough.

    Returns:
        An int64 array shaped (n_chunks, n_chunks), indexed by query chunk and
        key chunk. Query sampling approximates counts for large chunks.
    """
    doc0, end, span_lo, span_hi, is_noise = _visible_spans(
        document_lens, split_lens, attn_modes
    )
    seq_len = doc0.shape[0]
    n = n_chunks
    size = seq_len // n
    step = max(1, size // samples_per_chunk)
    offs = np.arange(step // 2, size, step)
    x = (np.arange(n)[:, None] * size + offs[None, :]).ravel()
    q_chunk = x // size
    edges = np.arange(n + 1) * size

    def overlap(lo: np.ndarray, hi: np.ndarray, cum: np.ndarray) -> np.ndarray:
        out = np.zeros((n, n + 1), np.int64)
        cover = np.zeros((n, n + 1), np.int64)
        keep = hi > lo
        q, lo, hi = q_chunk[keep], lo[keep], hi[keep]
        c0, c1 = lo // size, (hi - 1) // size
        one = c0 == c1
        np.add.at(out, (q[one], c0[one]), cum[hi[one]] - cum[lo[one]])
        two = ~one
        q, lo, hi, c0, c1 = q[two], lo[two], hi[two], c0[two], c1[two]
        np.add.at(out, (q, c0), cum[edges[c0 + 1]] - cum[lo])
        np.add.at(out, (q, c1), cum[hi] - cum[edges[c1]])
        np.add.at(cover, (q, c0 + 1), 1)
        np.add.at(cover, (q, c1), -1)
        whole = np.cumsum(cover, axis=1)[:, :n] * (cum[edges[1:]] - cum[edges[:-1]])
        return out[:, :n] + whole

    clean = np.concatenate([[0], np.cumsum(~is_noise, dtype=np.int64)])
    every = np.arange(seq_len + 1, dtype=np.int64)
    pairs = overlap(doc0[x], np.maximum(end[x], doc0[x]), clean)
    pairs += overlap(span_lo[x], span_hi[x], every)
    return pairs * step


def _step_cost(w: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """``sum_s max_r w[..., r, (r - s) % R]``: each ring step lasts as long as
    its slowest rank."""
    return w[..., rows, cols].max(axis=-1).sum(axis=-1)


def _range_counts(owner: np.ndarray, ring_size: int) -> np.ndarray:
    """Contiguous ranges per rank once adjacent chunks of a rank merge."""
    starts = np.ones(owner.shape[0], bool)
    starts[1:] = owner[1:] != owner[:-1]
    return np.bincount(owner[starts], minlength=ring_size)


def _rank_sums(
    pairs: np.ndarray, owner: np.ndarray, ring_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``rq[q, j]`` (chunk ``q``'s queries against rank ``j``'s keys),
    ``ck[k, r]`` (rank ``r``'s queries against chunk ``k``'s keys) and
    ``W[r, j]``, by summing chunk rows and columns per rank: O(n^2) integer
    adds instead of O(n^2 R) for a one-hot matmul. Every rank owns a chunk."""
    order = np.argsort(owner, kind="stable")
    offsets = np.searchsorted(owner[order], np.arange(ring_size))
    by_rank = pairs[order]
    rq = np.add.reduceat(pairs[:, order], offsets, axis=1)
    ck = np.add.reduceat(by_rank, offsets, axis=0).T
    w = np.add.reduceat(rq[order], offsets, axis=0)
    return rq, np.ascontiguousarray(ck), w


def layout_cost(
    pairs: Int[np.ndarray, " num_chunks num_chunks"],
    owner: Int[np.ndarray, " num_chunks"],
    ring_size: int,
    range_cost: int = 0,
) -> int:
    """Score chunk owners using step work and range overhead.

    Args:
        pairs: Square array of query/key visible-pair counts by chunk.
        owner: One-dimensional array mapping each chunk to its ring rank.
        ring_size: Number of ring ranks.
        range_cost: Cost per merged range per ring step; zero scores only work.

    Returns:
        The sum of each ring step's maximum rank work plus range_cost *
        ring_size * the largest per-rank merged-range count.
    """
    rows = np.broadcast_to(np.arange(ring_size), (ring_size, ring_size))
    cols = (rows - np.arange(ring_size)[:, None]) % ring_size
    step = int(_step_cost(_rank_sums(pairs, owner, ring_size)[2], rows, cols))
    return step + range_cost * ring_size * int(_range_counts(owner, ring_size).max())


def _swap_range_delta(
    owner: np.ndarray, q1: np.ndarray, q2: np.ndarray, ring_size: int
) -> np.ndarray:
    """Change in each rank's merged-range count when chunks ``q1`` and ``q2``
    (on different ranks) swap owners, one row per candidate swap.

    A swap can only start or end a range at ``q1``, ``q1 + 1``, ``q2`` and
    ``q2 + 1``, so only those range starts are recounted.
    """
    n, b = owner.shape[0], q1.shape[0]
    idx = np.sort(np.stack([q1, q1 + 1, q2, q2 + 1], axis=1), axis=1)
    valid = idx < n
    valid[:, 1:] &= idx[:, 1:] != idx[:, :-1]
    i = np.minimum(idx, n - 1)
    prev = np.maximum(i - 1, 0)
    r1, r2 = owner[q1][:, None], owner[q2][:, None]

    def after(j: np.ndarray) -> np.ndarray:
        o = owner[j]
        o = np.where(j == q1[:, None], r2, o)
        return np.where(j == q2[:, None], r1, o)

    start_before = valid & ((i == 0) | (owner[i] != owner[prev]))
    o_after = after(i)
    start_after = valid & ((i == 0) | (o_after != after(prev)))
    rows = np.repeat(np.arange(b), 4)
    delta = np.zeros((b, ring_size), np.int64)
    np.add.at(delta, (rows, owner[i].ravel()), -start_before.ravel().astype(np.int64))
    np.add.at(delta, (rows, o_after.ravel()), start_after.ravel().astype(np.int64))
    return delta


def refine_chunk_owners(
    pairs: Int[np.ndarray, " num_chunks num_chunks"],
    owner: Int[np.ndarray, " num_chunks"],
    ring_size: int,
    *,
    range_cost: int = 0,
    batch: int = 256,
    patience: int = 12,
    max_rounds: int = 4000,
    seed: int = 0,
) -> tuple[Int[np.ndarray, " num_chunks"], int]:
    """Swap chunks between ring ranks to minimize the layout cost.

    The cost is :func:`layout_cost`: ``sum_s max_r W[r, (r - s) % R]`` plus
    ``range_cost`` per merged range of the most fragmented rank and ring step.
    ``W[r, j]`` (rank ``r``'s queries against rank ``j``'s keys) and each
    rank's range count are kept exactly; each round scores ``batch`` random
    swaps of two chunks on different ranks from O(R) updates of ``W`` and the
    range starts next to the two chunks, applies the best one if it lowers the
    cost, and stops after ``patience`` rounds without one. Every rank holds as
    many chunks as before. Integer counts and a fixed seed make the result
    identical on every rank.

    Args:
        pairs: Square array of query/key visible-pair counts by chunk.
        owner: Initial rank assignment per chunk; copied, not modified in
            place.
        ring_size: Number of ring ranks.
        range_cost: Merged-range overhead per step; zero refines attention work
            only.
        batch: Number of candidate random swaps sampled in each round.
        patience: Stop after this many consecutive rounds without improvement.
        max_rounds: Maximum number of swap-search rounds.
        seed: Random seed shared by all ranks for deterministic planning.

    Returns:
        A tuple (owners, cost) containing the refined chunk-owner array and its
        layout_cost. Each rank retains its initial number of chunks.
    """
    n, size_r = owner.shape[0], ring_size
    owner = owner.copy()
    rq, ck, w = _rank_sums(pairs, owner, size_r)
    rows = np.broadcast_to(np.arange(size_r), (size_r, size_r))
    cols = (rows - np.arange(size_r)[:, None]) % size_r
    ranges = _range_counts(owner, size_r)
    best = int(_step_cost(w, rows, cols)) + range_cost * size_r * int(ranges.max())
    rng = np.random.default_rng(seed)
    idle = 0
    for _ in range(max_rounds):
        q1, q2 = rng.integers(0, n, (2, batch))
        r1, r2 = owner[q1], owner[q2]
        keep = r1 != r2
        q1, q2, r1, r2 = q1[keep], q2[keep], r1[keep], r2[keep]
        b = np.arange(q1.shape[0])
        cand = np.repeat(w[None], b.shape[0], axis=0)
        d = rq[q2] - rq[q1]
        cand[b, r1] += d
        cand[b, r2] -= d
        d = ck[q2] - ck[q1]
        cand[b, :, r1] += d
        cand[b, :, r2] -= d
        moved = ((q1, r1, r2), (q2, r2, r1))
        for qa, a_old, a_new in moved:
            for qb, b_old, b_new in moved:
                p = pairs[qa, qb]
                cand[b, a_old, b_old] += p
                cand[b, a_new, b_new] += p
                cand[b, a_old, b_new] -= p
                cand[b, a_new, b_old] -= p
        costs = _step_cost(cand, rows, cols)
        if range_cost:
            cand_ranges = ranges + _swap_range_delta(owner, q1, q2, size_r)
            costs = costs + range_cost * size_r * cand_ranges.max(axis=1)
        i = int(np.argmin(costs)) if costs.size else -1
        if i < 0 or costs[i] >= best:
            idle += 1
            if idle >= patience:
                break
            continue
        idle = 0
        a, c, ra, rc = q1[i], q2[i], r1[i], r2[i]
        best, w = int(costs[i]), cand[i]
        if range_cost:
            ranges = cand_ranges[i]
        owner[a], owner[c] = rc, ra
        d = pairs[:, c] - pairs[:, a]
        rq[:, ra] += d
        rq[:, rc] -= d
        d = pairs[c] - pairs[a]
        ck[:, ra] += d
        ck[:, rc] -= d
    return owner, best


def ring_chunk_assignment(
    document_lens: list[int],
    split_lens: list[int],
    attn_modes: list[str],
    *,
    ring_size: int,
    block_size: int = 128,
    min_chunks_per_rank: int = 1,
    max_chunks_per_rank: int = 64,
    chunk_overhead: float = 6.4e6,
    refine: bool = True,
    price_ranges: bool = False,
) -> list[list[tuple[int, int]]]:
    """Contiguous chunks per ring rank that balance every ring step.

    PTRR (``ring_planning``) deals 128-token blocks so that every ring step
    gives every rank the same work. Dense steps need contiguous chunks, so this
    optimizes the same criterion, ``sum_s max_r W[r, (r - s) % R]``, at chunk
    granularity, plus ``chunk_overhead`` visible pairs per merged range of the
    most fragmented rank and ring step (more ranges mean more, shorter
    sequences per call; :func:`layout_cost`). For each power of two from
    ``min_chunks_per_rank`` to ``max_chunks_per_rank`` chunks per rank it
    starts from the better-balanced of the work-sorted snake deal and the
    head-tail (zigzag) deal and swaps chunks between ranks while that lowers
    the per-step work (:func:`refine_chunk_owners`, unless ``refine`` is off);
    the cheapest of the plain and refined deals over all chunk counts wins.
    With ``price_ranges``, swaps then continue while they lower the cost
    including ranges. It is off by default: the overhead was fitted on H100,
    where pricing ranges in swaps changed BAGEL plans by under 0.5% either way,
    and on B300 it picked a 1M-token Qwen2.5-7B-layout plan with 26 instead of
    14 ranges per rank that ran 2.4% slower at USP 4x2. ``chunk_overhead`` was
    fitted to nine BAGEL long-video runs at 1-8M tokens on 32-64 H100s (time
    per visible pair times step cost plus overhead, within 3% on eight of
    them). Balancing only each rank's total (``chunk_assignment``) can leave
    single steps 1.5x off; on BAGEL long video at 8M tokens and 16 ring ranks,
    the best deal is 1.11x above perfect per-step balance and refining brings
    it to 1.01-1.04x. Adjacent chunks on the same rank are merged into one
    range. Both plain deals and the refined deal compete on the complete cost,
    so refinement cannot win by reducing work at the expense of greater range
    overhead.

    Args:
        document_lens: Lengths of the packed documents, in tokens.
        split_lens: Lengths of the attention spans, in document order.
        attn_modes: Attention mode for each span; must match split_lens.
        ring_size: Number of ring ranks.
        block_size: Chunk-size alignment in tokens.
        min_chunks_per_rank: Smallest candidate number of chunks per rank.
        max_chunks_per_rank: Largest candidate number of chunks per rank;
            candidates double from min_chunks_per_rank while the sequence
            remains divisible.
        chunk_overhead: Visible-pair-equivalent cost per merged range and ring
            step.
        refine: Whether to search for work-reducing swaps from the better
            initial deal.
        price_ranges: Whether to run an additional refinement search that
            includes range overhead. False keeps swaps work-only; final
            candidate selection still accounts for range overhead.

    Returns:
        Per-rank lists of sorted half-open global token ranges for the
        lowest-cost candidate assignment.
    """
    seq_len = sum(document_lens)
    pers = []
    per = min_chunks_per_rank
    while per <= max_chunks_per_rank and seq_len % (ring_size * per * block_size) == 0:
        pers.append(per)
        per *= 2
    assert pers, "sequence length must divide into ring_size * block_size chunks"
    finest = chunk_pair_counts(
        document_lens, split_lens, attn_modes, n_chunks=ring_size * pers[-1]
    )
    range_cost = int(round(chunk_overhead))
    best = None
    for per in pers:
        n = ring_size * per
        f = finest.shape[0] // n
        pairs = finest.reshape(n, f, n, f).sum(axis=(1, 3))
        size = seq_len // n
        zigzag = np.array([
            (i % ring_size) if (i // ring_size) % 2 == 0
            else ring_size - 1 - (i % ring_size)
            for i in range(n)
        ])  # fmt: skip
        # Work-sorted snake deal, as in chunk_assignment.
        snake = np.empty(n, np.int64)
        for pos, chunk in enumerate(np.argsort(-pairs.sum(axis=1), kind="stable")):
            lap, idx = divmod(pos, ring_size)
            snake[chunk] = idx if lap % 2 == 0 else ring_size - 1 - idx
        deals = (snake, zigzag)
        candidates = [
            (layout_cost(pairs, owner, ring_size, range_cost), owner) for owner in deals
        ]
        if refine:
            # Balance work from the better-balanced deal; optionally let swaps
            # then trade work against ranges (never costlier than their start).
            step = [layout_cost(pairs, owner, ring_size) for owner in deals]
            work, _ = refine_chunk_owners(pairs, deals[int(np.argmin(step))], ring_size)
            if price_ranges:
                work, _ = refine_chunk_owners(
                    pairs, work, ring_size, range_cost=range_cost
                )
            candidates.append((layout_cost(pairs, work, ring_size, range_cost), work))
        for cost, owner in candidates:
            if best is None or cost < best[0]:
                best = (cost, _merged_ranges(owner, ring_size, size))
    assert best is not None
    return best[1]


def _merged_ranges(
    owner: np.ndarray, ring_size: int, size: int
) -> list[list[tuple[int, int]]]:
    """Each rank's chunks in sequence order, adjacent ones merged."""
    chunks: list[list[tuple[int, int]]] = [[] for _ in range(ring_size)]
    for i, r in enumerate(owner.tolist()):
        lo, hi = i * size, (i + 1) * size
        if chunks[r] and chunks[r][-1][1] == lo:
            chunks[r][-1] = (chunks[r][-1][0], hi)
        else:
            chunks[r].append((lo, hi))
    return chunks


@dataclass(kw_only=True, slots=True)
class GenerationRingPlan:
    """Per-rank dense calls for every ring step.

    Attributes:
        ring_rank: This rank's position in the ring.
        ring_size: Ring degree.
        local_len: Tokens per ring rank.
        steps: ``steps[s]`` are the calls against the shard of ring rank
            ``(ring_rank - s) % ring_size``; ``q_index`` rows are local rows,
            ``k_index`` rows index that shard.
        chunks: Global token ranges per ring rank, in local order.
    """

    ring_rank: int
    ring_size: int
    local_len: int
    steps: list[list[DenseCall]]
    chunks: list[list[tuple[int, int]]]


def build_generation_ring_plan(
    document_lens: list[int],
    split_lens: list[int],
    attn_modes: list[str],
    *,
    ring_rank: int,
    ring_size: int,
    chunks: list[list[tuple[int, int]]],
    device: torch.device | str,
    max_gathered_keys: int = 1 << 21,
) -> GenerationRingPlan:
    """Plan dense generation-attention calls for one ring rank.

    Args:
        document_lens: Lengths of the packed documents, in tokens.
        split_lens: Lengths of the attention spans, in document order.
        attn_modes: Attention mode for each span; must match split_lens.
        ring_rank: Index of the calling rank within the ring.
        ring_size: Number of ranks in the ring.
        device: Device on which to allocate plan tensors.
        chunks: Per-rank lists of half-open global token ranges, in local
            order.
        max_gathered_keys: Target limit on concatenated keys per call; calls
            are split between query/key sequences to bound gather memory.

    Returns:
        A GenerationRingPlan with per-step dense calls for a layout without
        vision-group spans. Every rank derives it without communication.
    """
    seq_len = sum(document_lens)
    local = seq_len // ring_size
    segs = list(_segments(document_lens, split_lens, attn_modes))
    docs = sorted({(d0, d1) for d0, d1, _, _, _ in segs})
    full = [(s0, s1) for _, _, s0, s1, m in segs if m == "full"]
    noise = [(d0, s0, s1) for d0, _, s0, s1, m in segs if m == "noise"]
    is_noise = np.zeros(seq_len, dtype=bool)
    for _, s0, s1 in noise:
        is_noise[s0:s1] = True
    # Documents, full spans and noise spans are each sorted and disjoint, so
    # the ones a chunk overlaps are a contiguous run found by bisection; the
    # chunk-pair loops below then only visit those (a few per chunk instead of
    # every span: 9k spans x 64^2 chunk pairs x 16 steps at BAGEL 8M).
    doc_s, doc_e = [d0 for d0, _ in docs], [d1 for _, d1 in docs]
    full_s, full_e = [s0 for s0, _ in full], [s1 for _, s1 in full]
    noise_s, noise_e = [s0 for _, s0, _ in noise], [s1 for _, _, s1 in noise]

    def overlapping(items, starts, ends, lo, hi):
        return items[bisect_right(ends, lo) : bisect_left(starts, hi)]

    def with_offsets(ranges):
        out, off = [], 0
        for lo, hi in ranges:
            out.append((lo, hi, off))
            off += hi - lo
        return out

    def rows(pos: np.ndarray, chunk) -> np.ndarray:
        lo, _, off = chunk
        return pos - lo + off

    def clean(lo: int, hi: int) -> np.ndarray:
        pos = np.arange(lo, hi)
        return pos[~is_noise[lo:hi]]

    def bounded(q_parts, k_parts, **kind) -> list[DenseCall]:
        out, gq, gk, total = [], [], [], 0
        for qp, kp in zip(q_parts, k_parts, strict=True):
            if len(qp) == 0 or len(kp) == 0:
                continue
            if gk and total + len(kp) > max_gathered_keys:
                out.append(_make_call(gq, gk, device=device, **kind))
                gq, gk, total = [], [], 0
            gq.append(qp)
            gk.append(kp)
            total += len(kp)
        out.append(_make_call(gq, gk, device=device, **kind))
        return [c for c in out if c is not None]

    mine = with_offsets(chunks[ring_rank])
    steps: list[list[DenseCall]] = []
    for s in range(ring_size):
        theirs = with_offsets(chunks[(ring_rank - s) % ring_size])
        causal_q, causal_k, later_q, later_k, own_q, own_k = [], [], [], [], [], []
        # Keys from every chunk of the shard, per (query chunk, document) and
        # per (query chunk, span): each query row once per call.
        doc_keys: dict[tuple[int, int], list[np.ndarray]] = {}
        doc_rows: dict[tuple[int, int], np.ndarray] = {}
        span_keys: dict[tuple[int, int], list[np.ndarray]] = {}
        span_rows: dict[tuple[int, int], np.ndarray] = {}
        for qi, qc in enumerate(mine):
            a, b, _ = qc
            q_docs = overlapping(docs, doc_s, doc_e, a, b)
            q_full = overlapping(full, full_s, full_e, a, b)
            q_noise = overlapping(noise, noise_s, noise_e, a, b)
            spans = [*q_full, *((s0, s1) for _, s0, s1 in q_noise)]
            for kc in theirs:
                c, e, _ = kc
                if (a, b) == (c, e):
                    # The queries' own chunk.
                    for d0, d1 in q_docs:
                        lo, hi = max(a, d0), min(b, d1)
                        if hi <= lo:
                            continue
                        pos = clean(lo, hi)
                        causal_q.append(rows(pos, qc))
                        causal_k.append(rows(pos, kc))
                    for s0, s1 in q_full:
                        lo, hi = max(a, s0), min(b, s1)
                        if hi > lo:
                            rev = np.arange(hi - 1, lo - 1, -1)
                            later_q.append(rows(rev, qc))
                            later_k.append(rows(rev, kc))
                    for d0, s0, s1 in q_noise:
                        lo, hi = max(a, s0), min(b, s1)
                        if hi > lo:
                            keys = np.concatenate(
                                [clean(max(a, d0), s0), np.arange(lo, hi)]
                            )
                            own_q.append(rows(np.arange(lo, hi), qc))
                            own_k.append(rows(keys, kc))
                elif e <= a:
                    # An earlier shard: whole documents see its non-noise keys,
                    # noise spans crossing into it their own tokens there.
                    for d0, d1 in q_docs:
                        lo, hi = max(a, d0), min(b, d1)
                        klo, khi = max(c, d0), min(e, d1)
                        if hi > lo and khi > klo:
                            doc_rows[(qi, d0)] = rows(np.arange(lo, hi), qc)
                            doc_keys.setdefault((qi, d0), []).append(
                                rows(clean(klo, khi), kc)
                            )
                    for _, s0, s1 in q_noise:
                        lo, hi = max(a, s0), min(b, s1)
                        klo, khi = max(c, s0), min(e, s1)
                        if hi > lo and khi > klo:
                            span_rows[(qi, s0)] = rows(np.arange(lo, hi), qc)
                            span_keys.setdefault((qi, s0), []).append(
                                rows(np.arange(klo, khi), kc)
                            )
                else:
                    # A later shard: the rest of spans that cross into it.
                    for s0, s1 in spans:
                        lo, hi = max(a, s0), min(b, s1)
                        klo, khi = max(c, s0), min(e, s1)
                        if hi > lo and khi > klo:
                            span_rows[(qi, s0)] = rows(np.arange(lo, hi), qc)
                            span_keys.setdefault((qi, s0), []).append(
                                rows(np.arange(klo, khi), kc)
                            )
        doc_q = [doc_rows[k] for k in doc_keys]
        doc_k = [np.concatenate(v) for v in doc_keys.values()]
        cross_q = [span_rows[k] for k in span_keys]
        cross_k = [np.concatenate(v) for v in span_keys.values()]
        steps.append(
            [
                *bounded(causal_q, causal_k, causal=True, strict=False),
                *bounded(later_q, later_k, causal=False, strict=True),
                *bounded(own_q, own_k, causal=False, strict=False),
                *bounded(doc_q, doc_k, causal=False, strict=False),
                *bounded(cross_q, cross_k, causal=False, strict=False),
            ]
        )
    return GenerationRingPlan(
        ring_rank=ring_rank,
        ring_size=ring_size,
        local_len=local,
        steps=steps,
        chunks=chunks,
    )


def _ring_factors(q: Tensor, k: Tensor, v: Tensor, group) -> tuple[float, float, float]:
    """fp16 scales for Q (local, as in ``_to_f16``) and for K and V, shared by
    the ring group (same on every shard); one host sync for all three.

    ``group`` is ``None`` for a one-rank ring: the K and V scales are then local.
    """
    q_amax = _amaxes(q)
    kv_amax = _amaxes(k, v)
    if group is not None:
        dist.all_reduce(kv_amax, op=dist.ReduceOp.MAX, group=group)
    q_peak, *kv_peaks = torch.cat([q_amax, kv_amax]).tolist()
    factors = []
    for m in kv_peaks:
        factors.append(
            1.0
            if m == 0
            else 2.0 ** max(min(int(np.floor(np.log2(2.0**13 / m))), 60), -60)
        )
    return _f16_factor(q_peak), factors[0], factors[1]


class _GenerationRingAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, group, plan: GenerationRingPlan, scale):
        q2, k2, v2 = q[0], k[0].contiguous(), v[0].contiguous()
        n, h, d = q2.shape
        scale = d**-0.5 if scale is None else scale
        fq, fk, fv = _ring_factors(q2, k2, v2, group)
        q16 = _scaled_f16(q2, fq)
        scale16 = scale / (fq * fk)
        acc_out = torch.zeros(n, h, d, device=q.device, dtype=torch.float32)
        acc_lse = torch.full(
            (h, n), float("-inf"), device=q.device, dtype=torch.float32
        )
        kv = [k2, v2]
        for s, calls in enumerate(plan.steps):
            nxt = None
            if s + 1 < plan.ring_size:
                nxt = _Rotation(kv, group, plan.ring_rank, plan.ring_size)
            if calls:
                # Per step: the backward needs the visiting bf16 KV, so the
                # ring rotates bf16 and converts here.
                k16 = _scaled_f16(kv[0], fk)
                v16 = _scaled_f16(kv[1], fv)
                for call in calls:
                    o, lse = _call_fwd(call, q16, k16, v16, scale16)
                    merge_into(
                        acc_out, acc_lse, None if call.q_identity and
                        call.q_index.numel() == n else call.q_index, o, lse,
                    )  # fmt: skip
            if nxt is not None:
                kv = nxt.wait()
        out = acc_out.div_(fv).to(q.dtype)
        ctx.save_for_backward(q2, k2, v2, out, acc_lse, kv[0], kv[1])
        ctx.group, ctx.plan, ctx.scale = group, plan, scale
        return out.unsqueeze(0)

    @staticmethod
    def backward(ctx, dout):
        q2, k2, v2, out, lse, k_last, v_last = ctx.saved_tensors
        group, plan, scale = ctx.group, ctx.plan, ctx.scale
        dout2 = dout[0].contiguous()
        dq = torch.zeros_like(q2, dtype=torch.float32)
        size, rank = plan.ring_size, plan.ring_rank
        kv = [k_last, v_last]
        incoming: _Rotation | None = None
        acc: list[Tensor] | None = None
        # Steps last to first. At step s this rank holds the KV of ring rank
        # (rank - s); that shard's fp32 dK / dV accumulator arrives from the
        # next rank, gets this rank's share, and moves on with the KV.
        for s in reversed(range(size)):
            nxt = _Rotation(kv, group, rank, size, shift=-1) if s > 0 else None
            g_k_acc = torch.zeros_like(kv[0], dtype=torch.float32)
            g_v_acc = torch.zeros_like(kv[1], dtype=torch.float32)
            for call in plan.steps[s]:
                with fp32_grad_outputs() as exact:
                    grads = _call_bwd(call, q2, kv[0], kv[1], out, dout2, lse, scale)
                (g_q, s_q), (g_k, s_k), (g_v, s_v) = (
                    as_scaled(exact, g) for g in grads
                )
                accumulate_rows(dq, None if call.q_identity else call.q_index, g_q, s_q)
                k_rows = None if call.k_identity else call.k_index
                accumulate_rows(g_k_acc, k_rows, g_k, s_k)
                accumulate_rows(g_v_acc, k_rows, g_v, s_v)
            if incoming is not None:
                acc = incoming.wait()
                acc[0].add_(g_k_acc)
                acc[1].add_(g_v_acc)
            else:
                acc = [g_k_acc, g_v_acc]
            if s > 0:
                incoming = _Rotation(acc, group, rank, size, shift=-1)
                kv = nxt.wait()
        assert acc is not None
        return (
            dq.to(q2.dtype).unsqueeze(0),
            acc[0].to(k2.dtype).unsqueeze(0),
            acc[1].to(v2.dtype).unsqueeze(0),
            None,
            None,
            None,
        )


def generation_ring_attention(
    q: Float[Tensor, " batch local_tokens num_heads head_dim"],
    k: Float[Tensor, " batch local_tokens kv_heads head_dim"],
    v: Float[Tensor, " batch local_tokens kv_heads head_dim"],
    *,
    group: dist.ProcessGroup,
    plan: GenerationRingPlan,
    softmax_scale: float | None = None,
) -> Float[Tensor, " batch local_tokens num_heads head_dim"]:
    """Ring attention over contiguous chunks with dense varlen FA4 steps.

    Args:
        q: ``(batch, local_len, heads, head_dim)`` local queries; every
            sequence shares the plan's layout.
        k: ``(batch, local_len, kv_heads, head_dim)`` local keys.
        v: Values, same shape as ``k``.
        group: Ring process group.
        plan: From :func:`build_generation_ring_plan` for this rank.
        softmax_scale: Defaults to ``1 / sqrt(head_dim)``.

    Returns:
        ``(batch, local_len, heads, head_dim)`` attention output.
    """
    return torch.cat(
        [
            _GenerationRingAttention.apply(
                q[i : i + 1], k[i : i + 1], v[i : i + 1], group, plan, softmax_scale
            )
            for i in range(q.shape[0])
        ]
    )
