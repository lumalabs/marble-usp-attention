# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Dense generation-ring plans cover exactly the mask across all ring steps."""

from __future__ import annotations

import pytest
import torch

from usp_attention.layouts import build_layout
from usp_attention.reference import dense_mask_rows
from usp_attention.ring.dense_ring import chunk_assignment
from usp_attention.ring.generation_ring import build_generation_ring_plan
from usp_attention.ring.packed_layout import build_packed_layout

N = 8192


def _interleaved_generation(seed: int):
    """BAGEL-style documents: text, then frames as noise + clean (full) spans."""
    import random

    rng = random.Random(seed)
    docs, splits, modes = [], [], []
    while sum(docs) < N:
        doc_s, doc_m = [rng.randint(5, 90)], ["causal"]
        for _ in range(rng.randint(1, 8)):
            frame = rng.choice([64, 200, 333, 900])
            doc_s += [frame, frame]
            doc_m += ["noise", "full"]
            if rng.random() < 0.5:
                doc_s.append(rng.randint(3, 40))
                doc_m.append("causal")
        if sum(docs) + sum(doc_s) > N:
            doc_s, doc_m = [N - sum(docs)], ["causal"]
        docs.append(sum(doc_s))
        splits += doc_s
        modes += doc_m
    return docs, splits, modes


def _covered(call, q_global, k_global) -> torch.Tensor:
    out = torch.zeros(len(q_global), N, dtype=torch.bool)
    cu_q, cu_k = call.cu_q.tolist(), call.cu_k.tolist()
    for b in range(len(cu_q) - 1):
        rows = call.q_index[cu_q[b] : cu_q[b + 1]]
        keys = call.k_index[cu_k[b] : cu_k[b + 1]]
        lq, lk = len(rows), len(keys)
        i = torch.arange(lq)[:, None]
        j = torch.arange(lk)[None, :]
        shift = lk - lq - (1 if call.strict else 0)
        seen = (
            j <= i + shift
            if (call.causal or call.strict)
            else torch.ones(lq, lk, dtype=torch.bool)
        )
        for qi, row in enumerate(rows.tolist()):
            assert not out[row, k_global[keys[seen[qi]]]].any(), "pair twice in a call"
            out[row, k_global[keys[seen[qi]]]] = True
    return out


@pytest.mark.parametrize("name", ["docs", "mixed", "vlm", "video", "generation"])
@pytest.mark.parametrize(("ring_size", "per_rank"), [(1, 1), (4, 1), (4, 2), (8, 2)])
def test_generation_ring_covers_mask(name: str, ring_size: int, per_rank: int) -> None:
    if name == "generation":
        docs, splits, modes = _interleaved_generation(ring_size + per_rank)
    else:
        docs, splits, modes = build_layout(name, N, ring_size=ring_size, seed=5)
    mask = dense_mask_rows(build_packed_layout(docs, splits, modes), 0, N)
    chunks = chunk_assignment(
        docs, splits, modes, ring_size=ring_size, chunks_per_rank=per_rank
    )
    tokens = [torch.cat([torch.arange(lo, hi) for lo, hi in c]) for c in chunks]
    for r in range(ring_size):
        plan = build_generation_ring_plan(
            docs,
            splits,
            modes,
            ring_rank=r,
            ring_size=ring_size,
            chunks=chunks,
            device="cpu",
            max_gathered_keys=2048,
        )
        covered = torch.zeros(len(tokens[r]), N, dtype=torch.bool)
        for s, calls in enumerate(plan.steps):
            k_global = tokens[(r - s) % ring_size]
            for call in calls:
                pairs = _covered(call, tokens[r], k_global)
                assert not (pairs & covered).any(), "a pair is attended twice"
                covered |= pairs
        assert torch.equal(covered, mask[tokens[r]])


def test_chunk_pair_counts_match_the_mask() -> None:
    from usp_attention.ring.generation_ring import chunk_pair_counts

    docs, splits, modes = _interleaved_generation(3)
    mask = dense_mask_rows(build_packed_layout(docs, splits, modes), 0, N)
    n = 16
    # One sample per query: exact counts.
    pairs = chunk_pair_counts(docs, splits, modes, n_chunks=n, samples_per_chunk=N)
    exact = mask.view(n, N // n, n, N // n).sum(dim=(1, 3))
    assert torch.equal(torch.from_numpy(pairs), exact)


@pytest.mark.parametrize("refine", [False, True])
def test_chunk_choice_includes_range_overhead(refine: bool) -> None:
    import numpy as np

    from usp_attention.ring.generation_ring import (
        chunk_pair_counts,
        ring_chunk_assignment,
    )

    splits = [
        512,
        128,
        128,
        2048,
        2048,
        128,
        128,
        2048,
        128,
        2048,
        512,
        128,
        128,
        128,
        4096,
        2048,
    ]
    modes = [
        "full",
        "noise",
        "noise",
        "noise",
        "causal",
        "noise",
        "noise",
        "full",
        "full",
        "noise",
        "full",
        "noise",
        "causal",
        "full",
        "causal",
        "full",
    ]
    seq, ring, per = sum(splits), 2, 4
    chunks = ring_chunk_assignment(
        [seq],
        splits,
        modes,
        ring_size=ring,
        min_chunks_per_rank=per,
        max_chunks_per_rank=per,
        refine=refine,
    )
    size = seq // (ring * per)
    owner = np.empty(ring * per, dtype=np.int64)
    tokens = []
    for rank, ranges in enumerate(chunks):
        local = []
        for lo, hi in ranges:
            owner[lo // size : hi // size] = rank
            local.extend(range(lo, hi))
        assert len(local) == seq // ring
        tokens.extend(local)
    assert sorted(tokens) == list(range(seq))
    pairs = chunk_pair_counts([seq], splits, modes, n_chunks=ring * per)
    work = np.zeros((ring, ring), dtype=np.int64)
    for q in range(len(owner)):
        for k in range(len(owner)):
            work[owner[q], owner[k]] += pairs[q, k]
    compute = sum(
        max(work[r, (r - s) % ring] for r in range(ring)) for s in range(ring)
    )
    ranges = max(map(len, chunks))
    # The rejected refined deal has less compute (48,007,936), but four
    # ranges cost more overall (99,207,936) than this three-range deal.
    assert compute + 6_400_000 * ring * ranges == 89_675_136


@pytest.mark.parametrize(("seed", "ring_size"), [(1, 4), (2, 8)])
def test_refined_chunks_balance_steps_and_cover_mask(seed: int, ring_size: int) -> None:
    import numpy as np

    from usp_attention.ring.generation_ring import (
        _step_cost,
        chunk_pair_counts,
        refine_chunk_owners,
        ring_chunk_assignment,
    )

    docs, splits, modes = _interleaved_generation(seed)
    # Refinement keeps every rank's chunk count, never raises the per-step
    # cost, tracks W exactly, and is deterministic.
    n = ring_size * 8
    pairs = chunk_pair_counts(docs, splits, modes, n_chunks=n)
    start = np.arange(n) % ring_size
    owner, cost = refine_chunk_owners(pairs, start, ring_size)
    assert np.array_equal(np.bincount(owner, minlength=ring_size), [8] * ring_size)
    one_hot = np.eye(ring_size, dtype=np.int64)[owner]
    rows = np.broadcast_to(np.arange(ring_size), (ring_size, ring_size))
    cols = (rows - np.arange(ring_size)[:, None]) % ring_size
    assert cost == _step_cost(one_hot.T @ pairs @ one_hot, rows, cols)
    start_hot = np.eye(ring_size, dtype=np.int64)[start]
    assert cost <= _step_cost(start_hot.T @ pairs @ start_hot, rows, cols)
    again, _ = refine_chunk_owners(pairs, start, ring_size)
    assert np.array_equal(owner, again)

    priced = ring_chunk_assignment(
        docs, splits, modes, ring_size=ring_size, price_ranges=True
    )
    priced_tokens = [torch.cat([torch.arange(lo, hi) for lo, hi in c]) for c in priced]
    assert {len(t) for t in priced_tokens} == {N // ring_size}
    assert torch.equal(torch.cat(priced_tokens).sort().values, torch.arange(N))

    chunks = ring_chunk_assignment(docs, splits, modes, ring_size=ring_size)
    assert chunks == ring_chunk_assignment(docs, splits, modes, ring_size=ring_size)
    tokens = [torch.cat([torch.arange(lo, hi) for lo, hi in c]) for c in chunks]
    assert {len(t) for t in tokens} == {N // ring_size}
    assert torch.equal(torch.cat(tokens).sort().values, torch.arange(N))
    mask = dense_mask_rows(build_packed_layout(docs, splits, modes), 0, N)
    for r in range(ring_size):
        plan = build_generation_ring_plan(
            docs,
            splits,
            modes,
            ring_rank=r,
            ring_size=ring_size,
            chunks=chunks,
            device="cpu",
            max_gathered_keys=2048,
        )
        covered = torch.zeros(len(tokens[r]), N, dtype=torch.bool)
        for s, calls in enumerate(plan.steps):
            k_global = tokens[(r - s) % ring_size]
            for call in calls:
                pairs_rs = _covered(call, tokens[r], k_global)
                assert not (pairs_rs & covered).any(), "a pair is attended twice"
                covered |= pairs_rs
        assert torch.equal(covered, mask[tokens[r]])


def test_swap_range_delta_matches_recount() -> None:
    import numpy as np

    from usp_attention.ring.generation_ring import _range_counts, _swap_range_delta

    rng = np.random.default_rng(0)
    for ring_size, n in ((2, 8), (4, 32), (8, 64)):
        owner = rng.permutation(np.arange(n) % ring_size)
        q1, q2 = rng.integers(0, n, (2, 512))
        keep = owner[q1] != owner[q2]
        q1, q2 = q1[keep], q2[keep]
        delta = _swap_range_delta(owner, q1, q2, ring_size)
        base = _range_counts(owner, ring_size)
        for k in range(q1.shape[0]):
            swapped = owner.copy()
            swapped[q1[k]], swapped[q2[k]] = owner[q2[k]], owner[q1[k]]
            assert np.array_equal(base + delta[k], _range_counts(swapped, ring_size))


@pytest.mark.parametrize(("seed", "ring_size"), [(1, 4), (2, 8)])
def test_refinement_lowers_the_cost_with_range_overhead(
    seed: int, ring_size: int
) -> None:
    import numpy as np

    from usp_attention.ring.generation_ring import (
        chunk_pair_counts,
        layout_cost,
        refine_chunk_owners,
    )

    docs, splits, modes = _interleaved_generation(seed)
    n = ring_size * 8
    pairs = chunk_pair_counts(docs, splits, modes, n_chunks=n)
    start = np.arange(n) % ring_size
    range_cost = int(pairs.sum()) // (ring_size * n)
    owner, cost = refine_chunk_owners(pairs, start, ring_size, range_cost=range_cost)
    assert np.array_equal(np.bincount(owner, minlength=ring_size), [8] * ring_size)
    assert cost == layout_cost(pairs, owner, ring_size, range_cost)
    assert cost <= layout_cost(pairs, start, ring_size, range_cost)
    again, _ = refine_chunk_owners(pairs, start, ring_size, range_cost=range_cost)
    assert np.array_equal(owner, again)
