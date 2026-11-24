# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""The candidate-pair block classifier matches the dense block grid exactly."""

from __future__ import annotations

import random

import pytest
import torch

from usp_attention.ring import block_sparse
from usp_attention.ring.block_sparse import build_block_sparse_from_cross_qkv_aux


def _aux(rng: random.Random, seq: int) -> list[torch.Tensor]:
    """Per-token ``[fn, noise, doc, pos]`` of a random packed layout."""
    fn, noise, doc = [], [], []
    seg = 0
    d = 0
    while len(doc) < seq:
        doc_len = min(seq - len(doc), rng.choice([64, 200, 700, 1500, 4000]))
        filled = 0
        while filled < doc_len:
            length = min(doc_len - filled, rng.choice([32, 100, 300, 900]))
            mode = rng.choice(["causal", "causal", "full", "noise"])
            fn += [seg if mode != "causal" else -1] * length
            noise += [seg if mode == "noise" else -1] * length
            seg += mode != "causal"
            filled += length
        doc += [d] * doc_len
        d += 1
    pos = list(range(seq))
    return [torch.tensor(x, dtype=torch.int32) for x in (fn, noise, doc, pos)]


def _cross(aux, perm, q, kv, n):
    fn, noise, doc, pos = (t[perm] for t in aux)
    qs, ks = slice(q * n, (q + 1) * n), slice(kv * n, (kv + 1) * n)
    return [fn[qs], fn[ks], noise[qs], noise[ks], doc[qs], doc[ks], pos[qs], pos[ks]]


def _check(cross, block_size=128):
    dense = build_block_sparse_from_cross_qkv_aux(cross, block_size)
    types = build_block_sparse_from_cross_qkv_aux(cross, block_size, types_only=True)
    pairs = build_block_sparse_from_cross_qkv_aux(cross, block_size, compact=True)
    assert torch.equal(types.block_type, dense.block_type)
    assert torch.equal(pairs.block_type, dense.block_type)
    for cnt, idx in (
        ("kv_num_blocks", "kv_indices"),
        ("full_kv_num_blocks", "full_kv_indices"),
        ("q_num_blocks", "q_indices"),
        ("full_q_num_blocks", "full_q_indices"),
    ):
        want_cnt, got_cnt = getattr(dense, cnt), getattr(pairs, cnt)
        assert torch.equal(want_cnt, got_cnt), cnt
        want, got = getattr(dense, idx), getattr(pairs, idx)
        assert got.shape[-1] <= want.shape[-1]
        for row in range(want_cnt.shape[-1]):
            c = int(want_cnt[0, 0, row])
            assert torch.equal(got[0, 0, row, :c], want[0, 0, row, :c]), (idx, row)


@pytest.mark.parametrize("seed", range(6))
def test_pairs_match_dense_on_ring_steps(seed):
    rng = random.Random(seed)
    seq, ranks = 8192, 4
    aux = _aux(rng, seq)
    n = seq // ranks
    contiguous = torch.arange(seq)
    blocks = torch.randperm(seq // 128, generator=torch.Generator().manual_seed(seed))
    scattered = (blocks[:, None] * 128 + torch.arange(128)).flatten()  # PTRR-like
    for perm in (contiguous, scattered):
        for q in range(ranks):
            for kv in range(ranks):
                _check(_cross(aux, perm, q, kv, n))


def test_too_many_ids_per_block_falls_back_to_the_dense_grid(monkeypatch):
    monkeypatch.setattr(block_sparse, "_MAX_ID_SLOTS", 0)
    aux = _aux(random.Random(9), 2048)
    cross = _cross(aux, torch.arange(2048), 0, 0, 2048)
    dense = build_block_sparse_from_cross_qkv_aux(cross)
    out = build_block_sparse_from_cross_qkv_aux(cross, compact=True)
    assert torch.equal(out.block_type, dense.block_type)
    assert torch.equal(out.kv_num_blocks, dense.kv_num_blocks)
    for row, c in enumerate(dense.kv_num_blocks[0, 0].tolist()):
        assert torch.equal(
            out.kv_indices[0, 0, row, :c], dense.kv_indices[0, 0, row, :c]
        )


@pytest.mark.parametrize("seed", range(3))
def test_dense_steps_match_through_the_grid_fallback(monkeypatch, seed):
    monkeypatch.setattr(block_sparse, "_MAX_PAIR_DENSITY", 0.0)  # always the grid
    aux = _aux(random.Random(seed), 4096)
    perm = torch.randperm(4096, generator=torch.Generator().manual_seed(seed))
    _check(_cross(aux, perm, 0, 1, 2048))
