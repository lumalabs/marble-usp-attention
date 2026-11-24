# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Choose a sequence-parallel strategy for a packed mask from a cost model.

Which strategy is fastest depends on the mask as much as on the model:

- Ulysses splits heads, so its degree is capped by the (replicated) KV heads,
  and its all-to-all is fast only inside a node;
- a ring rotates every KV shard past every rank and hides that traffic behind
  compute, at one shard of remote KV in memory: best for dense, long masks
  (long generation, very long documents);
- a mask-aware gather brings in only the KV a rank's queries can see, in one
  exchange and one kernel pass: best when the mask has locality (packed
  documents, images, sliding windows), and it costs that KV in memory.

:func:`rank_strategies` enumerates Ulysses x (ring | gather) layouts for a
world size and head counts, and scores each from the same quantities the
planners use: visible (query, key) pairs per rank (compute), Ulysses and
ring / gather bytes split into intra- and inter-node traffic, per-step
overheads, and the extra KV each strategy keeps in memory. The
:class:`Hardware` constants are fitted to measured runs
(``benchmarks/fit_strategy.py``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from usp_attention.fa4.flash4_attn import AttentionMode
from usp_attention.ring.dense_gather import dense_gather_supported
from usp_attention.ring.dense_ring import chunk_assignment, dense_ring_supported


@dataclass(kw_only=True, slots=True)
class Hardware:
    """Throughputs and fixed costs of one GPU type and interconnect.

    Attributes:
        gpus_per_node: GPUs sharing the fast intra-node fabric.
        sparse_tflops: Effective attention throughput (forward + backward
            FLOPs of visible pairs per second) of block-sparse kernels.
        dense_tflops: Same for FA4's dense varlen kernels (dense gather / ring).
        fa3_tflops: Same for FlashAttention-3's varlen causal kernel (H100).
        intra_gbps: Per-GPU all-to-all bandwidth inside a node (GB/s).
        inter_gbps: Per-GPU bandwidth across nodes (GB/s).
        ring_step_ms: Fixed cost of one ring step across nodes, forward +
            backward (per-step synchronization with the slowest hop).
        ring_step_intra_ms: Same for a ring inside one node.
        gather_ms: Fixed cost of one gather (planning-free exchange + launch).
        noise_tflops: ``sparse_tflops`` for pairs in documents with noise spans
            (generation masks), whose tiles take the kernel's slower
            mask-evaluation path.
        memory_gib: Memory available for attention's extra KV and buffers.
    """

    gpus_per_node: int = 8
    # Fitted on H100 32-128 GPU runs (benchmarks/results/strategy_selector).
    sparse_tflops: float = 540.0
    dense_tflops: float = 540.0
    fa3_tflops: float = 470.0
    intra_gbps: float = 177.0
    inter_gbps: float = 29.4
    ring_step_ms: float = 11.2
    ring_step_intra_ms: float = 1.0
    gather_ms: float = 1.25
    noise_tflops: float = 486.0
    memory_gib: float = 30.0


@dataclass(kw_only=True, slots=True, frozen=True)
class Strategy:
    """One way to split attention over ``ring_size * ulysses_size`` GPUs.

    Attributes:
        ring_size: Ring / gather group size (sequence split across groups).
        ulysses_size: Ulysses degree (heads split inside a group).
        replicate_kv: KV head replication so ``ulysses_size`` divides them.
        kv_exchange: ``"ring"`` or ``"gather"`` (no effect at ring size 1).
        balance: Token layout: ``"ptrr"`` blocks or contiguous ``"chunks"``;
            ``"chunked"`` is the chunked ring (dense varlen calls over
            contiguous chunks balanced per ring step,
            :mod:`usp_attention.ring.generation_ring`).
    """

    ring_size: int
    ulysses_size: int
    replicate_kv: int = 1
    kv_exchange: str = "ring"
    balance: str = "ptrr"

    @property
    def label(self) -> str:
        s = f"R{self.ring_size}U{self.ulysses_size}"
        if self.replicate_kv > 1:
            s += f"kv{self.replicate_kv}"
        if self.ring_size > 1:
            s += f" {self.kv_exchange}/{self.balance}"
        return s


@dataclass(kw_only=True, slots=True)
class Estimate:
    """Predicted cost of one strategy (milliseconds per forward + backward)."""

    strategy: Strategy
    compute_ms: float
    ulysses_ms: float
    exchange_ms: float
    total_ms: float
    extra_kv_gib: float
    fits: bool
    features: dict = field(default_factory=dict)


@dataclass(kw_only=True, slots=True)
class MaskStats:
    """Per-token visible-key counts and the layout, computed once per mask."""

    document_lens: list[int]
    split_lens: list[int]
    attn_modes: list[AttentionMode]
    counts: np.ndarray
    sliding_window: int | None = None
    noise_docs: np.ndarray | None = None

    @property
    def seq_len(self) -> int:
        return int(self.counts.shape[0])

    @property
    def pairs(self) -> int:
        return int(self.counts.sum())

    def block_work(self, block: int = 128) -> np.ndarray:
        """Per-token work in tiles: each query block pays for every KV tile it
        touches (the union of its rows' visible keys, rounded up to tiles,
        plus one for misalignment), spread evenly over the block's tokens."""
        c = self.counts.reshape(-1, block)
        tiles = np.ceil(c.max(axis=1) / block) + 1
        return np.repeat(tiles * block / 1.0, block).astype(np.float64)


def mask_stats(
    document_lens: list[int],
    split_lens: list[int],
    attn_modes: list[AttentionMode],
    *,
    sliding_window: int | None = None,
) -> MaskStats:
    """Visible keys per query token for a packed layout."""
    # [MLR-02.08] reference helpers import torch; keep this module's import light.
    from usp_attention.reference import visible_key_counts

    counts = visible_key_counts(
        document_lens, split_lens, attn_modes, sliding_window=sliding_window
    ).numpy()
    # Tokens of documents that contain a noise span (generation masks).
    noise_docs = np.zeros(len(counts), dtype=bool)
    pos, split = 0, 0
    for doc_len in document_lens:
        end, has_noise = pos + doc_len, False
        consumed = pos
        while consumed < end:
            has_noise |= attn_modes[split] == "noise"
            consumed += split_lens[split]
            split += 1
        noise_docs[pos:end] = has_noise
        pos = end
    return MaskStats(
        noise_docs=noise_docs,
        document_lens=list(document_lens),
        split_lens=list(split_lens),
        attn_modes=list(attn_modes),
        counts=counts,
        sliding_window=sliding_window,
    )


def _needed_remote(stats: MaskStats, chunks) -> list[int]:
    """Remote keys each rank's queries can see, for contiguous-chunk shards."""
    seq_len = stats.seq_len
    window = stats.sliding_window
    owner = np.empty(seq_len, dtype=np.int32)
    for rank, ranges in enumerate(chunks):
        for lo, hi in ranges:
            owner[lo:hi] = rank
    docs, spans, noise = [], [], []
    pos = 0
    split_iter = iter(zip(stats.split_lens, stats.attn_modes, strict=True))
    for doc_len in stats.document_lens:
        d0, d1 = pos, pos + doc_len
        docs.append((d0, d1))
        while pos < d1:
            length, mode = next(split_iter)
            if mode != "causal":
                spans.append((pos, pos + length, d0, d1, mode))
            if mode == "noise":
                noise.append((pos, pos + length))
            pos += length
    docs_arr = np.array(docs, dtype=np.int64)
    span_end = np.array([s[1] for s in spans], dtype=np.int64)
    groups: dict[int, list[tuple[int, int]]] = {}
    for s0, s1, d0, _, mode in spans:
        if mode == "group":
            groups.setdefault(d0, []).append((s0, s1))
    out = []
    for rank, ranges in enumerate(chunks):
        need = np.zeros(seq_len, dtype=bool)
        group_docs = set()
        for lo, hi in ranges:
            first = np.searchsorted(docs_arr[:, 1], lo, side="right")
            for d0, d1 in docs_arr[first:]:
                if d0 >= hi:
                    break
                end = min(hi, d1)
                start = d0 if not window else max(d0, max(lo, d0) - window + 1)
                need[start:end] = True
            # Spans are sorted and disjoint: from the first one ending after lo
            # until one starts at or after hi.
            for i in range(
                int(np.searchsorted(span_end, lo, side="right")), len(spans)
            ):
                s0, s1, d0, _, mode = spans[i]
                if s0 >= hi:
                    break
                need[s0:s1] = True
                if mode == "group":
                    group_docs.add(d0)
        for d0 in group_docs:
            # Every group split of the document is visible.
            for g0, g1 in groups[d0]:
                need[g0:g1] = True
        for a, b in noise:
            # A noise span is invisible outside itself.
            if not any(a < hi and b > lo for lo, hi in ranges):
                need[a:b] = False
        out.append(int((need & (owner != rank)).sum()))
    return out


def candidates(
    world: int, heads: int, kv_heads: int, *, chunked_ring: bool = False
) -> list[Strategy]:
    """Ulysses x (ring | gather) layouts for ``world`` GPUs and these heads,
    plus the chunked ring for each split when ``chunked_ring``."""
    out = []
    for u in range(1, min(world, heads) + 1):
        if world % u or heads % u:
            continue
        rep = u // math.gcd(u, kv_heads)
        r = world // u
        if r == 1:
            out.append(Strategy(ring_size=1, ulysses_size=u, replicate_kv=rep))
            continue
        out.append(
            Strategy(ring_size=r, ulysses_size=u, replicate_kv=rep, balance="ptrr")
        )
        out.append(
            Strategy(
                ring_size=r,
                ulysses_size=u,
                replicate_kv=rep,
                kv_exchange="gather",
                balance="chunks",
            )
        )
        if chunked_ring:
            out.append(
                Strategy(
                    ring_size=r, ulysses_size=u, replicate_kv=rep, balance="chunked"
                )
            )
    return out


# Fixed cost of one contiguous range per ring step, in visible pairs (the
# planner's tau, fitted on BAGEL long video; generation_ring.ring_chunk_assignment).
_RANGE_PAIRS = 6.4e6


def _chunked_ring_cost(stats: MaskStats, ring_size: int) -> tuple[float, int]:
    """The chunked ring's planned layout: ``sum_s max_r W`` (visible pairs) and
    the most contiguous ranges any rank holds (:mod:`generation_ring`)."""
    # [MLR-02.08] generation_ring imports torch; keep this module's import light.
    from usp_attention.ring.generation_ring import (
        _range_counts,
        chunk_pair_counts,
        layout_cost,
        ring_chunk_assignment,
    )

    seq = stats.seq_len
    chunks = ring_chunk_assignment(
        stats.document_lens, stats.split_lens, stats.attn_modes, ring_size=ring_size
    )
    per = 64
    while per > 1 and seq % (ring_size * per * 128):
        per //= 2
    n = ring_size * per
    size = seq // n
    owner = np.empty(n, dtype=np.int64)
    for rank, ranges in enumerate(chunks):
        for lo, hi in ranges:
            owner[lo // size : hi // size] = rank
    pairs = chunk_pair_counts(
        stats.document_lens, stats.split_lens, stats.attn_modes, n_chunks=n
    )
    step = layout_cost(pairs, owner, ring_size)
    return float(step), int(_range_counts(owner, ring_size).max())


def _off_node_share(group: int, stride: int, gpus_per_node: int) -> float:
    """Share of a rank's peers (``group`` ranks ``stride`` apart) on other nodes."""
    if group <= 1:
        return 0.0
    rank0_node = 0
    off = sum(1 for p in range(1, group) if (p * stride) // gpus_per_node != rank0_node)
    return off / (group - 1)


def features(
    strategy: Strategy,
    stats: MaskStats,
    *,
    heads: int,
    kv_heads: int,
    head_dim: int,
    gpus_per_node: int = 8,
) -> dict:
    """Hardware-independent quantities of one strategy on one mask."""
    r, u, rep = strategy.ring_size, strategy.ulysses_size, strategy.replicate_kv
    world = r * u
    seq = stats.seq_len
    kv_local = kv_heads * rep / u
    flops_per_pair = 4 * head_dim * (heads / u) * 3.5  # fwd 2 GEMMs, bwd 2.5x
    # Kernels pay for whole tiles: per-token work is the tile-rounded key count.
    work_tok = stats.block_work()
    total_pairs = float(work_tok.sum())
    noise_tok = work_tok * stats.noise_docs if stats.noise_docs is not None else None
    noise_share = float(noise_tok.sum() / total_pairs) if noise_tok is not None else 0.0
    if r == 1:
        rank_pairs = total_pairs
        rank_imbalance = 1.0
    elif strategy.balance == "chunked":
        # Dense calls pay for visible pairs, not tiles; each ring step lasts as
        # long as its slowest rank, plus a fixed cost per range and step.
        step_pairs, ranges = _chunked_ring_cost(stats, r)
        rank_pairs = step_pairs + _RANGE_PAIRS * r * ranges
        rank_imbalance = step_pairs / (stats.pairs / r)
    elif strategy.balance == "chunks":
        per_rank = 1
        best = None
        while per_rank <= 64 and seq % (r * per_rank * 128) == 0:
            chunks = chunk_assignment(
                stats.document_lens,
                stats.split_lens,
                stats.attn_modes,
                ring_size=r,
                chunks_per_rank=per_rank,
                counts=_as_tensor(stats.counts),
            )
            work = [
                sum(float(work_tok[lo:hi].sum()) for lo, hi in rank) for rank in chunks
            ]
            imb = max(work) / (sum(work) / len(work))
            if best is None or imb < best[0] - 1e-9:
                best = (imb, chunks, work)
            if imb <= 1.025:
                break
            per_rank *= 2
        rank_imbalance, chunks, work = best
        busiest = int(np.argmax(work))
        rank_pairs = work[busiest]
        if noise_tok is not None and rank_pairs > 0:
            noise_share = (
                sum(float(noise_tok[lo:hi].sum()) for lo, hi in chunks[busiest])
                / rank_pairs
            )
    else:
        rank_pairs = total_pairs / r
        rank_imbalance = 1.0
    feats = {
        "compute_flops": rank_pairs * flops_per_pair,
        "rank_imbalance": rank_imbalance,
        "noise_share": noise_share,
        "ring_steps": r if (r > 1 and strategy.kv_exchange == "ring") else 0,
        "gathers": 1 if (r > 1 and strategy.kv_exchange == "gather") else 0,
    }
    # Ulysses all-to-all: q, k, v, out forward; dout, dq, dk, dv backward.
    a2a = (u - 1) / u * (seq / world) * head_dim * 2 * (4 * heads + 4 * kv_heads * rep)
    off = _off_node_share(u, 1, gpus_per_node)
    feats["ulysses_intra_bytes"] = a2a * (1 - off)
    feats["ulysses_inter_bytes"] = a2a * off
    ring_off = 1.0 if world > gpus_per_node else 0.0
    feats["ring_bytes"] = 0.0
    feats["gather_bytes"] = 0.0
    feats["extra_kv_bytes"] = 0.0
    feats["exchange_inter"] = ring_off
    if r > 1 and strategy.kv_exchange == "ring":
        # K, V forward and backward (bf16), dK, dV in fp32.
        feats["ring_bytes"] = (r - 1) * (seq / r) * kv_local * head_dim * 16
        feats["extra_kv_bytes"] = 2 * (seq / r) * kv_local * head_dim * (4 + 8)
    elif r > 1:
        if strategy.balance == "chunks":
            halo = max(_needed_remote(stats, chunks))
        else:
            # Scattered blocks: every rank holds pieces of every document.
            halo = int(seq * (r - 1) / r)
        feats["halo_tokens"] = halo
        # K, V once (kept for the backward), dK, dV back in fp32.
        feats["gather_bytes"] = halo * kv_local * head_dim * 12
        feats["extra_kv_bytes"] = halo * kv_local * head_dim * (4 + 8)
    return feats


def _as_tensor(counts: np.ndarray):
    # [MLR-02.08] chunk_assignment takes torch counts.
    import torch

    return torch.from_numpy(counts)


def estimate(
    strategy: Strategy, feats: dict, hw: Hardware, *, kernel: str = "sparse"
) -> Estimate:
    """Predicted milliseconds per attention forward + backward.

    Args:
        kernel: ``"sparse"`` (block-sparse FA4), ``"dense"`` (FA4 varlen) or
            ``"fa3"`` (FlashAttention-3 varlen, H100).
    """
    tflops = {
        "sparse": hw.sparse_tflops,
        "dense": hw.dense_tflops,
        "fa3": hw.fa3_tflops,
    }[kernel]
    compute = feats["compute_flops"] / (tflops * 1e12) * 1e3
    if kernel == "sparse" and feats.get("noise_share"):
        # Noise-span documents run the slower mask-evaluation path.
        noise = feats["compute_flops"] * feats["noise_share"]
        compute += noise * (1 / hw.noise_tflops - 1 / tflops) / 1e12 * 1e3
    ulysses = (
        feats["ulysses_intra_bytes"] / (hw.intra_gbps * 1e9)
        + feats["ulysses_inter_bytes"] / (hw.inter_gbps * 1e9)
    ) * 1e3
    bw = hw.inter_gbps if feats["exchange_inter"] else hw.intra_gbps
    if feats["ring_steps"]:
        comm = feats["ring_bytes"] / (bw * 1e9) * 1e3
        # Rotation overlaps compute; each step adds a fixed cost.
        step = hw.ring_step_ms if feats["exchange_inter"] else hw.ring_step_intra_ms
        exchange = max(0.0, comm - compute) + feats["ring_steps"] * step
    elif feats["gathers"]:
        exchange = feats["gather_bytes"] / (bw * 1e9) * 1e3 + hw.gather_ms
    else:
        exchange = 0.0
    total = compute + ulysses + exchange
    extra = feats["extra_kv_bytes"] / 2**30
    return Estimate(
        strategy=strategy,
        compute_ms=compute,
        ulysses_ms=ulysses,
        exchange_ms=exchange,
        total_ms=total,
        extra_kv_gib=extra,
        fits=extra <= hw.memory_gib,
        features=feats,
    )


def rank_strategies(
    document_lens: list[int],
    split_lens: list[int],
    attn_modes: list[AttentionMode],
    *,
    world: int,
    heads: int,
    kv_heads: int,
    head_dim: int,
    hardware: Hardware | None = None,
    sliding_window: int | None = None,
    kernel: str | None = None,
    chunked_ring: bool = True,
) -> list[Estimate]:
    """Every candidate strategy, fastest predicted first (strategies that fit first).

    Args:
        kernel: Attention kernel family (see :func:`estimate`); ``None`` picks
            dense varlen kernels for head_dim > 128 when the layout allows it
            (FA4 has no block sparsity there on H100), else block-sparse.
        chunked_ring: Also score the chunked ring (always dense varlen calls)
            where its calls can express the mask (no vision groups, no window).
    """
    hw = hardware or Hardware()
    stats = mask_stats(
        document_lens, split_lens, attn_modes, sliding_window=sliding_window
    )
    if kernel is None:
        dense = head_dim > 128 and dense_ring_supported(
            document_lens, split_lens, attn_modes
        )
        kernel = "dense" if dense else "sparse"
    out = [
        estimate(
            s,
            features(
                s,
                stats,
                heads=heads,
                kv_heads=kv_heads,
                head_dim=head_dim,
                gpus_per_node=hw.gpus_per_node,
            ),
            hw,
            kernel="dense" if s.balance == "chunked" else kernel,
        )
        for s in candidates(
            world,
            heads,
            kv_heads,
            chunked_ring=chunked_ring
            and sliding_window is None
            and dense_gather_supported(document_lens, split_lens, attn_modes),
        )
    ]
    return sorted(out, key=lambda e: (not e.fits, e.total_ms))
