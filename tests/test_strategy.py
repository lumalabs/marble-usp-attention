# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""The strategy cost model's candidates, including the chunked ring."""

from __future__ import annotations

from usp_attention.strategy import (
    Strategy,
    candidates,
    features,
    mask_stats,
    rank_strategies,
)

# Two generated videos (prompt, then noise / clean frame pairs) and one text
# sample: 16k tokens, divisible into 4 ring ranks x 32 chunks of 128 tokens.
GEN = dict(
    document_lens=[7168, 7168, 2048],
    split_lens=[1024, 1536, 1536, 1536, 1536, 1024, 1536, 1536, 1536, 1536, 2048],
    attn_modes=[
        "causal", "noise", "full", "noise", "full",
        "causal", "noise", "full", "noise", "full",
        "causal",
    ],
)  # fmt: skip


def test_candidates_add_the_chunked_ring_per_split():
    plain = candidates(8, 28, 4)
    with_chunked = candidates(8, 28, 4, chunked_ring=True)
    assert all(s.balance != "chunked" for s in plain)
    chunked = [s for s in with_chunked if s.balance == "chunked"]
    # One per split with a ring (U = 1, 2, 4; U = 7 does not divide 8 GPUs).
    assert {(s.ring_size, s.ulysses_size) for s in chunked} == {(8, 1), (4, 2), (2, 4)}
    assert all(s.kv_exchange == "ring" for s in chunked)


def test_chunked_ring_compute_is_its_per_step_cost():
    stats = mask_stats(**GEN)
    s = Strategy(ring_size=4, ulysses_size=1, balance="chunked")
    f = features(s, stats, heads=28, kv_heads=4, head_dim=128)
    # Each ring step lasts as long as its slowest rank: at least an even split.
    assert f["rank_imbalance"] >= 0.99
    assert f["ring_steps"] == 4 and f["ring_bytes"] > 0


def test_rank_strategies_offer_the_chunked_ring_only_where_dense_calls_apply():
    def chunked(**kw):
        return [
            e for e in rank_strategies(**kw, world=8, heads=28, kv_heads=4, head_dim=128)
            if e.strategy.balance == "chunked"
        ]  # fmt: skip

    assert chunked(**GEN)
    assert not chunked(**GEN, chunked_ring=False)
    assert not chunked(**GEN, sliding_window=512)
    grouped = dict(
        GEN, attn_modes=["group" if m == "full" else m for m in GEN["attn_modes"]]
    )
    assert not chunked(**grouped)
