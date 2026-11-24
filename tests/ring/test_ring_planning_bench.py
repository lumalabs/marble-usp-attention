# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Correctness and performance tests for _build_rank_plan vectorization.

Run with::

    cd projects/usp_attention
    source .venv/bin/activate
    pytest tests/ring/test_ring_planning_bench.py -v
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from usp_attention.ring.packed_layout import build_packed_layout
from usp_attention.ring.ring_planning import (
    RankRingPlan,
    _build_rank_plan,
    _build_rank_plan_vectorized,
    build_ring_plan,
    create_layout_block_mask,
)


def _make_mock_block_mask(
    q_blocks: int,
    max_kv_per_row: int,
    *,
    density: float = 0.5,
    include_full: bool = False,
    seed: int = 42,
) -> Any:
    """Build a mock BlockMask-like object for testing.

    The real BlockMask from flex_attention has kv_num_blocks, kv_indices,
    and optionally full_kv_num_blocks / full_kv_indices. We mock these
    with SimpleNamespace since the planning code only accesses those fields.
    """
    torch.manual_seed(seed)
    kv_indices = torch.randint(0, q_blocks, (1, 1, q_blocks, max_kv_per_row))
    kv_num_blocks = (torch.rand(1, 1, q_blocks) * max_kv_per_row * density).int()
    kv_num_blocks = kv_num_blocks.clamp(max=max_kv_per_row)

    full_kv_num_blocks = None
    full_kv_indices = None
    if include_full:
        full_kv_indices = torch.randint(0, q_blocks, (1, 1, q_blocks, max_kv_per_row))
        full_kv_num_blocks = (
            torch.rand(1, 1, q_blocks) * max_kv_per_row * density * 0.3
        ).int()
        full_kv_num_blocks = full_kv_num_blocks.clamp(max=max_kv_per_row)

    return SimpleNamespace(
        BLOCK_SIZE=(128, 128),
        kv_num_blocks=kv_num_blocks,
        kv_indices=kv_indices,
        full_kv_num_blocks=full_kv_num_blocks,
        full_kv_indices=full_kv_indices,
    )


def _plans_equal(a: RankRingPlan, b: RankRingPlan) -> bool:
    """Check structural equality of two RankRingPlans."""
    if a.rank != b.rank:
        return False
    if a.local_block_start != b.local_block_start:
        return False
    if a.local_block_end != b.local_block_end:
        return False
    if len(a.steps) != len(b.steps):
        return False
    for sa, sb in zip(a.steps, b.steps):
        if sa.step != sb.step:
            return False
        if sa.kv_owner_rank != sb.kv_owner_rank:
            return False
        if sa.estimated_block_intersections != sb.estimated_block_intersections:
            return False
        if not torch.equal(sa.active_local_q_blocks, sb.active_local_q_blocks):
            return False
    return True


class TestBuildRankPlanVectorizedCorrectness:
    """Verify vectorized version matches the original loop for various configs."""

    @pytest.mark.parametrize("world_size", [2, 4, 8])
    @pytest.mark.parametrize("q_blocks", [8, 16, 64])
    def test_sparse_only(self, world_size: int, q_blocks: int):
        if q_blocks % world_size != 0:
            pytest.skip("q_blocks not divisible by world_size")
        mask = _make_mock_block_mask(q_blocks, max_kv_per_row=q_blocks, density=0.5)
        for rank in range(world_size):
            original = _build_rank_plan(
                block_mask=mask, world_size=world_size, rank=rank
            )
            vectorized = _build_rank_plan_vectorized(
                block_mask=mask, world_size=world_size, rank=rank
            )
            assert _plans_equal(original, vectorized), (
                f"Mismatch for rank={rank} world_size={world_size} q_blocks={q_blocks}"
            )

    @pytest.mark.parametrize("world_size", [2, 4])
    @pytest.mark.parametrize("q_blocks", [8, 32])
    def test_with_full_blocks(self, world_size: int, q_blocks: int):
        if q_blocks % world_size != 0:
            pytest.skip("q_blocks not divisible by world_size")
        mask = _make_mock_block_mask(
            q_blocks, max_kv_per_row=q_blocks, density=0.5, include_full=True
        )
        for rank in range(world_size):
            original = _build_rank_plan(
                block_mask=mask, world_size=world_size, rank=rank
            )
            vectorized = _build_rank_plan_vectorized(
                block_mask=mask, world_size=world_size, rank=rank
            )
            assert _plans_equal(original, vectorized), (
                f"Mismatch for rank={rank} world_size={world_size} q_blocks={q_blocks}"
            )

    def test_empty_mask(self):
        mask = _make_mock_block_mask(8, max_kv_per_row=8, density=0.0)
        mask.kv_num_blocks = torch.zeros(1, 1, 8, dtype=torch.int32)
        for rank in range(2):
            original = _build_rank_plan(block_mask=mask, world_size=2, rank=rank)
            vectorized = _build_rank_plan_vectorized(
                block_mask=mask, world_size=2, rank=rank
            )
            assert _plans_equal(original, vectorized)

    def test_fully_dense(self):
        q_blocks = 16
        mask = _make_mock_block_mask(q_blocks, max_kv_per_row=q_blocks, density=1.0)
        mask.kv_num_blocks = torch.full((1, 1, q_blocks), q_blocks, dtype=torch.int32)
        for rank in range(4):
            original = _build_rank_plan(block_mask=mask, world_size=4, rank=rank)
            vectorized = _build_rank_plan_vectorized(
                block_mask=mask, world_size=4, rank=rank
            )
            assert _plans_equal(original, vectorized)


class TestBuildRankPlanPerformance:
    """Benchmark original vs vectorized at different scales.

    Skipped by default in CI. Run explicitly with::

        pytest -m benchmark tests/ring/test_ring_planning_bench.py -v -s
    """

    @pytest.fixture(autouse=True)
    def _require_cuda(self):
        if (
            not hasattr(torch, "cuda")
            or not hasattr(torch.cuda, "is_available")
            or not hasattr(torch.cuda, "device_count")
        ):
            pytest.skip("CUDA not available")
        if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
            pytest.skip(
                "CUDA not available or less than 4 GPUs (only run on nodes with at least 4 GPUs to prevent long running tests)"
            )

    @pytest.mark.parametrize(
        "q_blocks,world_size,density",
        [
            (64, 4, 0.4),
            (256, 4, 0.4),
            (512, 8, 0.3),
            (1024, 8, 0.15),
            (2048, 8, 0.1),
        ],
    )
    def test_speedup(self, q_blocks: int, world_size: int, density: float):
        mask = _make_mock_block_mask(
            q_blocks, max_kv_per_row=q_blocks, density=density, include_full=True
        )

        n_warmup = 2
        n_iters = 5

        for _ in range(n_warmup):
            _build_rank_plan(block_mask=mask, world_size=world_size, rank=0)
            _build_rank_plan_vectorized(block_mask=mask, world_size=world_size, rank=0)

        t0 = time.perf_counter()
        for _ in range(n_iters):
            for rank in range(world_size):
                _build_rank_plan(block_mask=mask, world_size=world_size, rank=rank)
        original_time = (time.perf_counter() - t0) / n_iters

        t0 = time.perf_counter()
        for _ in range(n_iters):
            for rank in range(world_size):
                _build_rank_plan_vectorized(
                    block_mask=mask, world_size=world_size, rank=rank
                )
        vectorized_time = (time.perf_counter() - t0) / n_iters

        speedup = original_time / max(vectorized_time, 1e-9)
        print(
            f"\nq_blocks={q_blocks} W={world_size}: "
            f"original={original_time * 1000:.2f}ms "
            f"vectorized={vectorized_time * 1000:.2f}ms "
            f"speedup={speedup:.1f}x"
        )


def _make_ring_planning_layout(seq_len: int) -> Any:
    """Build a mixed-attention layout for end-to-end ring planning benchmarks."""
    if seq_len % 512 != 0:
        raise ValueError(f"seq_len must be divisible by 512, got {seq_len=}.")
    num_blocks = seq_len // 128
    document_lens = [seq_len // 4] * 4
    split_lens = [128] * num_blocks
    base_modes = ["causal", "full", "noise", "full"]
    attn_modes = [base_modes[idx % len(base_modes)] for idx in range(num_blocks)]
    return build_packed_layout(
        document_lens,
        split_lens,
        attn_modes,
        device=torch.device("cpu"),
    )


class TestBuildRingPlanPerformance:
    """Benchmark reference vs direct block-sparse ring planning."""

    @pytest.fixture(autouse=True)
    def _require_cuda(self):
        if (
            not hasattr(torch, "cuda")
            or not hasattr(torch.cuda, "is_available")
            or not hasattr(torch.cuda, "device_count")
        ):
            pytest.skip("CUDA not available")
        if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
            pytest.skip(
                "CUDA not available or less than 4 GPUs (only run on nodes with at least 4 GPUs to prevent long running tests)"
            )

    @pytest.mark.parametrize(
        ("seq_len", "world_size"),
        [
            (1024, 2),
            (4096, 4),
        ],
    )
    @pytest.mark.parametrize("precompute_block_mask", [False, True])
    def test_reference_vs_direct(
        self,
        seq_len: int,
        world_size: int,
        precompute_block_mask: bool,
    ) -> None:
        layout = _make_ring_planning_layout(seq_len)
        block_mask = (
            create_layout_block_mask(layout, device=torch.device("cpu"))
            if precompute_block_mask
            else None
        )

        reference_plan = build_ring_plan(
            layout,
            world_size=world_size,
            device=torch.device("cpu"),
            block_mask=block_mask,
            use_direct_planner_block_sparse=False,
        )
        direct_plan = build_ring_plan(
            layout,
            world_size=world_size,
            device=torch.device("cpu"),
            block_mask=block_mask,
            use_direct_planner_block_sparse=True,
        )
        torch.testing.assert_close(
            direct_plan.token_permutation, reference_plan.token_permutation
        )
        torch.testing.assert_close(
            direct_plan.restore_permutation, reference_plan.restore_permutation
        )

        n_warmup = 2
        n_iters = 5
        for _ in range(n_warmup):
            build_ring_plan(
                layout,
                world_size=world_size,
                device=torch.device("cpu"),
                block_mask=block_mask,
                use_direct_planner_block_sparse=False,
            )
            build_ring_plan(
                layout,
                world_size=world_size,
                device=torch.device("cpu"),
                block_mask=block_mask,
                use_direct_planner_block_sparse=True,
            )

        t0 = time.perf_counter()
        for _ in range(n_iters):
            build_ring_plan(
                layout,
                world_size=world_size,
                device=torch.device("cpu"),
                block_mask=block_mask,
                use_direct_planner_block_sparse=False,
            )
        reference_time = (time.perf_counter() - t0) / n_iters

        t0 = time.perf_counter()
        for _ in range(n_iters):
            build_ring_plan(
                layout,
                world_size=world_size,
                device=torch.device("cpu"),
                block_mask=block_mask,
                use_direct_planner_block_sparse=True,
            )
        direct_time = (time.perf_counter() - t0) / n_iters

        speedup = reference_time / max(direct_time, 1e-9)
        print(
            f"\nseq_len={seq_len} W={world_size} "
            f"precompute={int(precompute_block_mask)}: "
            f"reference={reference_time * 1000:.2f}ms "
            f"direct={direct_time * 1000:.2f}ms "
            f"speedup={speedup:.1f}x"
        )
