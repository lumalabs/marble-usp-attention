# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Dense-ring options belong to each plan, independent of process environment."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import torch

from usp_attention import USPConfig, build_usp_plan
from usp_attention.distributed import USPGroups
from usp_attention.ring.dense_ring import build_dense_ring_plan


def _groups() -> USPGroups:
    return USPGroups(
        ring_size=2,
        ulysses_size=1,
        ring_rank=0,
        ulysses_rank=0,
        cp_rank=0,
        ring_group=None,
        ulysses_group=None,
        cp_group=None,
    )


@pytest.mark.parametrize("chunks", [0, -1, 1.5, True])
def test_invalid_chunks_rejected(chunks: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        USPConfig(dense_ring_chunks=chunks)


def test_forced_chunks_and_plan_scoped_execution_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("CHUNKS", "REFINE", "INPLACE", "DKV_OVERLAP"):
        monkeypatch.setenv(f"USP_DENSE_RING_{name}", "invalid-old-env")
    config = USPConfig(
        dense_ring=True,
        dense_ring_chunks=2,
        dense_ring_inplace=True,
        dense_ring_dkv_overlap=True,
    )
    spec = dict(
        document_lens=[2048, 2048],
        split_lens=[512, 1536, 512, 1536],
        attn_modes=["causal", "full", "causal", "full"],
    )
    plan = build_usp_plan(**spec, groups=_groups(), config=config, device="cpu")
    assert plan.dense_ring is not None
    assert len(plan.dense_ring.chunks[0]) == 2
    assert plan.dense_ring.dkv_overlap is True
    assert any(c.q_starts is not None for calls in plan.dense_ring.steps for c in calls)
    other = build_usp_plan(
        **spec,
        groups=_groups(),
        config=dataclasses.replace(
            config, dense_ring_inplace=False, dense_ring_dkv_overlap=False
        ),
        device="cpu",
    )
    assert other.dense_ring.dkv_overlap is False
    assert all(c.q_starts is None for calls in other.dense_ring.steps for c in calls)
    assert torch.equal(plan.token_permutation, other.token_permutation)


def test_incompatible_forced_chunk_count_rejected() -> None:
    with pytest.raises(ValueError, match="divide the sequence"):
        build_usp_plan(
            document_lens=[4096],
            split_lens=[1024, 3072],
            attn_modes=["causal", "full"],
            groups=_groups(),
            config=USPConfig(dense_ring=True, dense_ring_chunks=3),
            device="cpu",
        )


def test_inplace_planning_does_not_change_query_or_key_rows() -> None:
    args = ([1024], [256, 768], ["causal", "full"])
    gathered = build_dense_ring_plan(*args, ring_rank=0, ring_size=2, device="cpu")
    inplace = build_dense_ring_plan(
        *args, ring_rank=0, ring_size=2, device="cpu", inplace=True
    )
    for before, after in zip(gathered.steps, inplace.steps, strict=True):
        # Anti-causal in-place calls replace reversed strictly-causal calls.
        assert len(before) == len(after)
        for a, b in zip(before, after, strict=True):
            assert sorted(a.q_index.tolist()) == sorted(b.q_index.tolist())
            assert sorted(a.k_index.tolist()) == sorted(b.k_index.tolist())


@pytest.mark.parametrize("exchange", ["ring", "gather"])
def test_gpu_exactness_fixture_selects_dense_paths(exchange: str) -> None:
    from usp_attention.layouts import build_layout

    path = Path(__file__).resolve().parents[1] / "repro/layouts/exactness.json"
    spec = build_layout(f"file:{path}#0", 4096)
    plan = build_usp_plan(
        document_lens=spec[0],
        split_lens=spec[1],
        attn_modes=spec[2],
        groups=_groups(),
        device="cpu",
        config=USPConfig(
            dense_ring=True,
            exact_dense_ring=True,
            kv_exchange=exchange,
            dense_min_span=0,
        ),
    )
    assert (
        plan.generation_ring is not None
        if exchange == "ring"
        else plan.dense_gather is not None
    )


def test_benchmark_passes_recorded_execution_flags_into_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
    import bench_attention

    captured = []

    def capture_plan(**kwargs):
        captured.append(kwargs["config"])
        raise RuntimeError("stop before allocation")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_attention.py",
            "--ring",
            "2",
            "--ulysses",
            "1",
            "--seq-len",
            "4096",
            "--dense-ring-chunks",
            "2",
            "--no-dense-ring-refine",
            "--dense-ring-inplace",
            "--dense-ring-dkv-overlap",
            "--exact-dense-ring",
            "--gather-regather",
            "--gather-max-keys",
            "512",
        ],
    )
    monkeypatch.setattr(
        bench_attention, "init_distributed", lambda: torch.device("cpu")
    )
    monkeypatch.setattr(bench_attention.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(bench_attention.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(bench_attention.USPGroups, "create", lambda **kwargs: _groups())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(bench_attention, "build_usp_plan", capture_plan)
    with pytest.raises(RuntimeError, match="stop before allocation"):
        bench_attention.main()
    config = captured[0]
    assert config.dense_ring_chunks == 2
    assert not config.dense_ring_refine
    assert config.dense_ring_inplace and config.dense_ring_dkv_overlap
    assert config.exact_dense_ring and config.gather_regather
    assert config.gather_max_keys == 512


def test_head256_gate_selects_dense_gather_instead_of_native_ring() -> None:
    from usp_attention.layouts import docs_layout

    docs, splits, modes = docs_layout(4096)
    plan = build_usp_plan(
        document_lens=docs,
        split_lens=splits,
        attn_modes=modes,
        groups=_groups(),
        config=USPConfig(kv_exchange="gather", dense_ring=True),
        device="cpu",
    )
    assert plan.dense_gather is not None
    assert plan.step_meta is None
