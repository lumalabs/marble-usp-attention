# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""The automatic strategy: regimes, shortlists, trial schedule and locks (CPU)."""

from __future__ import annotations

import dataclasses
import pickle
from dataclasses import dataclass
from typing import NamedTuple

import pytest
import torch

import usp_attention.strategy as strategy_module
from usp_attention import USPConfig
from usp_attention.auto import (
    AutoStrategy,
    PlannedBatch,
    plan_to,
    regime_signature,
    strategy_config,
)
from usp_attention.strategy import Strategy

# Two generated videos and one text sample (as in test_strategy.py).
GEN = dict(
    document_lens=[7168, 7168, 2048],
    split_lens=[1024, 1536, 1536, 1536, 1536, 1024, 1536, 1536, 1536, 1536, 2048],
    attn_modes=[
        "causal", "noise", "full", "noise", "full",
        "causal", "noise", "full", "noise", "full",
        "causal",
    ],
)  # fmt: skip


def _auto(**kw) -> AutoStrategy:
    return AutoStrategy(
        world=8, heads=28, kv_heads=4, head_dim=128, config=USPConfig(), **kw
    )


def _batch(auto: AutoStrategy, step: int) -> PlannedBatch:
    """A planned batch without plans (selection only)."""
    regime = regime_signature(**GEN)
    shortlist = auto.shortlist(**GEN)
    return PlannedBatch(
        step=step,
        regime=regime,
        shortlist=shortlist,
        strategy=auto.decide(step, regime, shortlist),
        plan=None,  # type: ignore[arg-type]
        sliding_plan=None,
        layout=(GEN["document_lens"], GEN["split_lens"], GEN["attn_modes"]),
    )


def test_regime_signature_ignores_small_changes_but_not_new_modes():
    a = regime_signature(**GEN)
    # Move 64 tokens from a noise span to the next full span.
    splits = list(GEN["split_lens"])
    splits[1] -= 64
    splits[2] += 64
    shifted = dict(GEN, split_lens=splits)
    assert regime_signature(**shifted) == a
    no_noise = dict(
        GEN, attn_modes=["full" if m == "noise" else m for m in GEN["attn_modes"]]
    )
    assert regime_signature(**no_noise) != a
    assert regime_signature(**GEN, sliding_window=512) != a


def test_shortlist_is_top_k_and_offers_the_chunked_ring_on_generation_masks():
    auto = _auto(top_k=3)
    short = auto.shortlist(**GEN)
    assert len(short) == 3 and len(set(short)) == 3
    assert any(s.balance == "chunked" for s in _auto(top_k=99).shortlist(**GEN))
    assert auto.shortlist(**GEN) is short  # cached per regime


def test_trials_cycle_candidates_then_lock_after_the_horizon():
    auto = _auto(top_k=3, trial_steps=2, horizon=10)
    times = {}
    regime = regime_signature(**GEN)
    for step in range(9):
        b = _batch(auto, step)
        assert b.strategy == b.shortlist[step % 3]
        # The second candidate is the fastest; first (compile) steps are slow
        # and, for the first candidate, misleadingly fast.
        ms = 100.0 if b.strategy == b.shortlist[1] else 150.0
        if step < 3:
            ms = 1.0 if step == 0 else 5000.0
        times[b.strategy] = ms
        auto.record(b, ms)
        assert (regime in auto.locks) == (step == 8)
    best, start = auto.locks[regime]
    assert best == auto.shortlist(**GEN)[1] and start == 8 + 10
    # Steps before the lock applies keep the trial schedule (plans built ahead).
    assert _batch(auto, 17).strategy == auto.shortlist(**GEN)[17 % 3]
    assert _batch(auto, 18).strategy == best and _batch(auto, 99).strategy == best
    # Locked regimes ignore further timings.
    auto.record(_batch(auto, 99), 1.0)
    assert auto.locks[regime] == (best, 18)


def test_shared_lock_table_is_used_by_other_planners():
    shared: dict = {}
    trainer, worker = _auto(shared=shared, horizon=0), _auto(shared=shared)
    regime = regime_signature(**GEN)
    choice = trainer.shortlist(**GEN)[2]
    shared[regime] = (choice, 0)
    assert worker.decide(5, regime, worker.shortlist(**GEN)) == choice


def test_strategy_config_runs_exact_paths():
    base = USPConfig(combine_mode="online")
    chunked = strategy_config(
        Strategy(ring_size=4, ulysses_size=2, balance="chunked"), base, hopper=True
    )
    assert (
        chunked.dense_ring and chunked.exact_dense_ring and chunked.balance == "chunks"
    )
    gather = strategy_config(
        Strategy(ring_size=4, ulysses_size=2, kv_exchange="gather", balance="chunks"),
        base,
        hopper=True,
    )
    assert gather.kv_exchange == "gather" and gather.halo_overlap
    ring = strategy_config(Strategy(ring_size=8, ulysses_size=1), base, hopper=False)
    assert ring.balance == "ptrr" and not ring.halo_overlap
    assert ring.combine_mode == "online"


@dataclass(kw_only=True, slots=True)
class _Inner:
    t: torch.Tensor
    n: int


class _Pair(NamedTuple):
    a: torch.Tensor
    b: int


@dataclass(kw_only=True, slots=True)
class _Outer:
    inner: _Inner
    items: list
    by_key: dict
    pair: _Pair


def test_plan_to_moves_nested_tensors_and_keeps_the_rest():
    src = _Outer(
        inner=_Inner(t=torch.ones(2), n=3),
        items=[torch.zeros(1), "x"],
        by_key={"a": torch.arange(3)},
        pair=_Pair(a=torch.ones(1), b=7),
    )
    out = plan_to(src, torch.device("cpu"))
    assert out is not src and out.inner.n == 3 and out.items[1] == "x"
    assert torch.equal(out.by_key["a"], torch.arange(3))
    assert isinstance(out.pair, _Pair) and out.pair.b == 7


def _make_predicate():
    def mask_mod(q, k):
        return q >= k

    return mask_mod


def test_planned_batches_pickle_closure_predicates_by_reference(monkeypatch):
    # Like FA4's mask_mod singletons: module-level, but defined in a closure.
    predicate = _make_predicate()
    monkeypatch.setattr(strategy_module, "_TEST_MASK_MOD", predicate, raising=False)
    batch = dataclasses.replace(
        _batch(_auto(), 0),
        plan=_Outer(  # type: ignore[arg-type]
            inner=_Inner(t=torch.ones(2), n=1),
            items=[predicate],
            by_key={},
            pair=_Pair(a=torch.zeros(1), b=0),
        ),
    )
    with pytest.raises(pickle.PicklingError):
        pickle.dumps(
            dataclasses.replace(
                batch,
                plan=_Outer(  # type: ignore[arg-type]
                    inner=_Inner(t=torch.ones(2), n=1),
                    items=[_make_predicate()],
                    by_key={},
                    pair=_Pair(a=torch.zeros(1), b=0),
                ),
            )
        )
    back = pickle.loads(pickle.dumps(batch))
    assert back.plan.items[0] is predicate
    assert back.strategy == batch.strategy and back.regime == batch.regime
    assert torch.equal(back.plan.inner.t, torch.ones(2))
