# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Dense KV-gather plans cover exactly the mask, and exchanges line up."""

from __future__ import annotations

import pytest
import torch

from usp_attention.layouts import build_layout
from usp_attention.reference import dense_mask_rows, visible_key_counts
from usp_attention.ring.dense_gather import (
    build_dense_gather_plan,
    dense_gather_supported,
)
from usp_attention.ring.dense_ring import chunk_assignment
from usp_attention.ring.group_order import vision_group_order
from usp_attention.ring.packed_layout import build_packed_layout

N = 8192


def _covered(call, k_global, n_local: int) -> torch.Tensor:
    """(local query, global key) pairs one dense call attends."""
    out = torch.zeros(n_local, N, dtype=torch.bool)
    cu_q, cu_k = call.cu_q.tolist(), call.cu_k.tolist()
    for b in range(len(cu_q) - 1):
        rows = call.q_index[cu_q[b] : cu_q[b + 1]]
        keys = call.k_index[cu_k[b] : cu_k[b + 1]]
        lq, lk = len(rows), len(keys)
        i = torch.arange(lq)[:, None]
        j = torch.arange(lk)[None, :]
        # Causal and strict calls are aligned bottom-right.
        shift = lk - lq - (1 if call.strict else 0)
        seen = (
            j <= i + shift
            if (call.causal or call.strict)
            else torch.ones(lq, lk, dtype=torch.bool)
        )
        for qi, row in enumerate(rows.tolist()):
            out[row, k_global[keys[seen[qi]]]] = True
    return out


def _interleaved_generation(seed: int):
    """BAGEL-style documents: text, then frames as noise + clean (full) spans."""
    import random

    rng = random.Random(seed)
    docs, splits, modes = [], [], []
    while sum(docs) < N:
        doc_s, doc_m = [rng.randint(5, 90)], ["causal"]
        for _ in range(rng.randint(1, 6)):
            frame = rng.choice([64, 200, 333])
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


@pytest.mark.parametrize("exact", [True, False])
@pytest.mark.parametrize("name", ["docs", "mixed", "vlm", "video", "generation"])
@pytest.mark.parametrize(("ring_size", "per_rank"), [(4, 1), (4, 2), (8, 2)])
def test_dense_gather_covers_mask(
    name: str, ring_size: int, per_rank: int, exact: bool
) -> None:
    if name == "generation":
        docs, splits, modes = _interleaved_generation(ring_size * per_rank)
    else:
        docs, splits, modes = build_layout(name, N, ring_size=ring_size, seed=5)
    if not dense_gather_supported(docs, splits, modes):
        pytest.skip("vision groups")
    mask = dense_mask_rows(build_packed_layout(docs, splits, modes), 0, N)
    chunks = chunk_assignment(
        docs, splits, modes, ring_size=ring_size, chunks_per_rank=per_rank
    )
    plans = [
        build_dense_gather_plan(
            docs,
            splits,
            modes,
            ring_rank=r,
            ring_size=ring_size,
            chunks=chunks,
            device="cpu",
            max_gathered_keys=4096,
            exact_output=exact,
        )
        for r in range(ring_size)
    ]
    for r, plan in enumerate(plans):
        for o, other in enumerate(plans):
            assert plan.send_counts[o] == other.recv_counts[r]
        local = torch.cat([torch.arange(lo, hi) for lo, hi in chunks[r]])
        # Rows of ``local KV ++ halo``: own tokens, then each sender's in order.
        halo = []
        for o, other in enumerate(plans):
            if o == r:
                continue
            starts = torch.cat([torch.arange(lo, hi) for lo, hi in chunks[o]])
            sent = torch.split(other.send_index, other.send_counts)[r]
            halo.append(starts[sent])
        k_global = torch.cat([local, *halo])
        covered = torch.zeros(len(local), N, dtype=torch.bool)
        for call in plan.calls:
            pairs = _covered(call, k_global, len(local))
            assert not (pairs & covered).any(), "a pair is attended twice"
            covered |= pairs
        assert torch.equal(covered, mask[local])


def test_group_order_chunks_balance() -> None:
    from usp_attention.usp import _balanced_chunks

    docs, splits, modes = build_layout("vlm_group", N, ring_size=4, seed=1)
    perm = vision_group_order(docs, splits, modes)
    counts = visible_key_counts(docs, splits, modes)[perm]
    order = _balanced_chunks(docs, splits, modes, 4, torch.device("cpu"), counts=counts)
    assert torch.equal(torch.sort(order).values, torch.arange(N))
    work = counts[order].view(4, -1).sum(1).float()
    assert work.max() / work.mean() < 1.15


def test_shared_noise_prefix_requires_merge_without_full_spans() -> None:
    # Each rank owns a quarter of one document. Noise rows on rank 1
    # attend both a shared remote prefix and their own noise span.
    docs, splits, modes = (
        [4096],
        [512, 1024, 512, 1024, 1024],
        ["causal", "noise", "causal", "noise", "causal"],
    )
    chunks = [[(1024 * r, 1024 * (r + 1))] for r in range(4)]
    plan = build_dense_gather_plan(
        docs,
        splits,
        modes,
        ring_rank=1,
        ring_size=4,
        chunks=chunks,
        device="cpu",
    )
    counts = torch.bincount(
        torch.cat([call.q_index for call in plan.calls]), minlength=1024
    )
    assert counts.max() == 2
    assert not plan.rows_disjoint
