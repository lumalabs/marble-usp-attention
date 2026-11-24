# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from usp_attention.ring.packed_layout import build_packed_layout
from usp_attention.ring.ring_planning import (
    RankRingPlan,
    RingPlan,
    RingStepPlan,
    _block_perm_to_token_perm,
    _count_overlap,
    _ptrr_scheduling,
    apply_token_permutation,
    build_ring_plan,
    create_layout_block_mask,
    get_rank_plan,
    inverse_permutation,
)


class TestInversePermutation:
    def test_roundtrip(self):
        perm = torch.tensor([3, 0, 2, 1])
        inv = inverse_permutation(perm)
        x = torch.arange(4)
        torch.testing.assert_close(x[perm][inv], x)

    def test_roundtrip_random(self):
        perm = torch.randperm(16)
        inv = inverse_permutation(perm)
        x = torch.arange(16)
        torch.testing.assert_close(x[perm][inv], x)

    def test_identity(self):
        perm = torch.arange(5)
        inv = inverse_permutation(perm)
        torch.testing.assert_close(inv, perm)


class TestApplyTokenPermutation:
    def test_reorder_dim1(self):
        x = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]])
        perm = torch.tensor([3, 1, 0, 2])
        result = apply_token_permutation(x, perm, seq_dim=1)
        expected = x[:, perm, :]
        torch.testing.assert_close(result, expected)

    def test_reorder_dim2(self):
        x = torch.arange(12).reshape(1, 3, 4).float()
        perm = torch.tensor([2, 0, 3, 1])
        result = apply_token_permutation(x, perm, seq_dim=2)
        expected = x[:, :, perm]
        torch.testing.assert_close(result, expected)


class TestBlockPermToTokenPerm:
    def test_small_blocks(self):
        block_perm = torch.tensor([2, 0, 1])
        result = _block_perm_to_token_perm(block_perm, block_size=2, seq_len=6)
        expected = torch.tensor([4, 5, 0, 1, 2, 3])
        torch.testing.assert_close(result, expected)

    def test_block_size_1(self):
        block_perm = torch.tensor([3, 1, 2, 0])
        result = _block_perm_to_token_perm(block_perm, block_size=1, seq_len=4)
        torch.testing.assert_close(result, block_perm)

    def test_not_divisible_raises(self):
        block_perm = torch.tensor([1, 0])
        with pytest.raises(NotImplementedError, match="seq_len must be divisible"):
            _block_perm_to_token_perm(block_perm, block_size=2, seq_len=5)

    def test_is_valid_permutation(self):
        block_perm = torch.tensor([3, 1, 0, 2])
        result = _block_perm_to_token_perm(block_perm, block_size=4, seq_len=16)
        assert result.numel() == 16
        assert set(result.tolist()) == set(range(16))


class TestPTRRScheduling:
    def test_all_indices_present(self):
        costs = torch.tensor([10.0, 1.0, 8.0, 2.0, 6.0, 3.0, 5.0, 4.0])
        grouped = _ptrr_scheduling(costs, group_size=2)
        assert grouped.shape == (2, 4)
        all_indices = grouped.reshape(-1).sort().values
        torch.testing.assert_close(all_indices, torch.arange(8))

    def test_groups_are_sorted(self):
        costs = torch.tensor([10.0, 1.0, 8.0, 2.0, 6.0, 3.0, 5.0, 4.0])
        grouped = _ptrr_scheduling(costs, group_size=2)
        for g in range(2):
            row = grouped[g]
            assert (row[1:] >= row[:-1]).all()

    def test_balanced_costs(self):
        costs = torch.tensor([10.0, 1.0, 8.0, 2.0, 6.0, 3.0])
        grouped = _ptrr_scheduling(costs, group_size=3)
        group_costs = [costs[grouped[g]].sum().item() for g in range(3)]
        spread = max(group_costs) - min(group_costs)
        naive_spread = costs.sum().item()
        assert spread < naive_spread

    def test_non_1d_raises(self):
        with pytest.raises(ValueError, match="Expected 1D"):
            _ptrr_scheduling(torch.zeros(2, 3), group_size=2)

    def test_not_divisible_raises(self):
        with pytest.raises(ValueError, match="num_tasks must be divisible"):
            _ptrr_scheduling(torch.zeros(5), group_size=2)


class TestCountOverlap:
    def test_full_overlap(self):
        row = torch.tensor([0, 1, 2, 3])
        assert _count_overlap(row, count=4, start=0, end=4) == 4

    def test_partial_overlap(self):
        row = torch.tensor([0, 1, 5, 6, 2, 3])
        assert _count_overlap(row, count=4, start=0, end=3) == 2

    def test_no_overlap(self):
        row = torch.tensor([10, 11, 12])
        assert _count_overlap(row, count=3, start=0, end=5) == 0

    def test_zero_count(self):
        row = torch.tensor([0, 1, 2])
        assert _count_overlap(row, count=0, start=0, end=3) == 0

    def test_count_less_than_length(self):
        row = torch.tensor([5, 1, 3, 0])
        assert _count_overlap(row, count=2, start=0, end=10) == 2


class TestGetRankPlan:
    def _make_plan(self, world_size: int) -> RingPlan:
        rank_plans = [
            RankRingPlan(
                rank=r,
                local_block_start=r * 2,
                local_block_end=(r + 1) * 2,
                steps=[
                    RingStepPlan(
                        step=0,
                        kv_owner_rank=r,
                        active_local_q_blocks=torch.tensor([0, 1]),
                        estimated_block_intersections=2,
                    )
                ],
            )
            for r in range(world_size)
        ]
        mock_block_mask = MagicMock()
        return RingPlan(
            world_size=world_size,
            block_size=128,
            token_permutation=torch.arange(world_size * 2),
            restore_permutation=torch.arange(world_size * 2),
            rank_plans=rank_plans,
            reordered_block_mask=mock_block_mask,
        )

    def test_valid_rank(self):
        plan = self._make_plan(4)
        rp = get_rank_plan(plan, rank=2)
        assert rp.rank == 2

    def test_negative_rank_raises(self):
        plan = self._make_plan(4)
        with pytest.raises(ValueError, match="rank out of range"):
            get_rank_plan(plan, rank=-1)

    def test_out_of_range_raises(self):
        plan = self._make_plan(4)
        with pytest.raises(ValueError, match="rank out of range"):
            get_rank_plan(plan, rank=4)

    def test_rank_mismatch_raises(self):
        plan = self._make_plan(2)
        plan.rank_plans[0] = RankRingPlan(
            rank=99,
            local_block_start=0,
            local_block_end=2,
            steps=[],
        )
        with pytest.raises(ValueError, match="Rank mismatch"):
            get_rank_plan(plan, rank=0)


def _assert_rank_ring_plan_equal(actual: RankRingPlan, expected: RankRingPlan) -> None:
    assert actual.rank == expected.rank
    assert actual.local_block_start == expected.local_block_start
    assert actual.local_block_end == expected.local_block_end
    assert len(actual.steps) == len(expected.steps)
    for actual_step, expected_step in zip(actual.steps, expected.steps):
        assert actual_step.step == expected_step.step
        assert actual_step.kv_owner_rank == expected_step.kv_owner_rank
        assert (
            actual_step.estimated_block_intersections
            == expected_step.estimated_block_intersections
        )
        torch.testing.assert_close(
            actual_step.active_local_q_blocks, expected_step.active_local_q_blocks
        )


def _assert_ring_plan_equal(actual: RingPlan, expected: RingPlan) -> None:
    assert actual.world_size == expected.world_size
    assert actual.block_size == expected.block_size
    torch.testing.assert_close(actual.token_permutation, expected.token_permutation)
    torch.testing.assert_close(actual.restore_permutation, expected.restore_permutation)
    assert len(actual.rank_plans) == len(expected.rank_plans)
    for actual_rank_plan, expected_rank_plan in zip(
        actual.rank_plans, expected.rank_plans
    ):
        _assert_rank_ring_plan_equal(actual_rank_plan, expected_rank_plan)


class TestBuildRingPlan:
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

    @pytest.mark.parametrize("precompute_block_mask", [False, True])
    @pytest.mark.parametrize(
        ("document_lens", "split_lens", "attn_modes"),
        [
            ([128, 128, 128, 128], [128, 128, 128, 128], ["causal"] * 4),
            ([512], [128, 128, 128, 128], ["causal", "full", "noise", "causal"]),
            (
                [256, 256, 256, 256],
                [128, 128, 128, 128, 128, 128, 128, 128],
                ["noise", "full", "causal", "full", "noise", "causal", "full", "noise"],
            ),
        ],
    )
    def test_direct_matches_reference(
        self,
        document_lens: list[int],
        split_lens: list[int],
        attn_modes: list[str],
        precompute_block_mask: bool,
    ) -> None:
        device = torch.device("cpu")
        layout = build_packed_layout(
            document_lens,
            split_lens,
            attn_modes,
            device=device,
        )
        block_mask = (
            create_layout_block_mask(layout, device=device)
            if precompute_block_mask
            else None
        )

        reference = build_ring_plan(
            layout,
            world_size=2,
            device=device,
            block_mask=block_mask,
            use_direct_planner_block_sparse=False,
        )
        direct = build_ring_plan(
            layout,
            world_size=2,
            device=device,
            block_mask=block_mask,
            use_direct_planner_block_sparse=True,
        )

        _assert_ring_plan_equal(direct, reference)
        assert reference.reordered_block_mask is not None
        assert direct.reordered_block_mask is None
