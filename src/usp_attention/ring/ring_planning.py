# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-FileCopyrightText: 2016- Facebook, Inc and the PyTorch contributors (_ptrr_scheduling)
# SPDX-FileCopyrightText: 2025 Bytedance Ltd. and/or its affiliates (packed-mask encoding adapted from BAGEL create_sparse_mask)
# SPDX-License-Identifier: MIT AND BSD-3-Clause AND Apache-2.0
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import (
    BlockMask,
    and_masks,
    create_block_mask,
    or_masks,
)

from usp_attention.ring.block_sparse import (
    BLOCK_EMPTY,
    BlockSparseResult,
    build_block_sparse_from_cross_qkv_aux,
)
from usp_attention.ring.packed_layout import PackedLayout


class CreateBlockMaskFn(Protocol):
    def __call__(
        self,
        mask_mod: Callable,
        B: int | None,
        H: int | None,
        Q_LEN: int,
        KV_LEN: int,
        device: str | torch.device | int | None = None,
        BLOCK_SIZE: int | tuple[int, int] = 128,
        _compile: bool = False,
    ) -> BlockMask: ...


CREATE_BLOCK_MASK: CreateBlockMaskFn | None = None


@dataclass(kw_only=True, slots=True)
class RingStepPlan:
    """Work plan for one ring step on one rank.

    Attributes:
        step: Ring iteration index.
        kv_owner_rank: Rank whose KV shard is consumed at this step.
        active_local_q_blocks: Local Q block indices that need compute this step.
        estimated_block_intersections: Estimated number of QxKV block intersections.
    """

    step: int
    kv_owner_rank: int
    active_local_q_blocks: Tensor
    estimated_block_intersections: int


@dataclass(kw_only=True, slots=True)
class RankRingPlan:
    """All ring steps for one rank."""

    rank: int
    local_block_start: int
    local_block_end: int
    steps: list[RingStepPlan]


@dataclass(kw_only=True, slots=True)
class RingPlan:
    """Global ring planning output.

    Attributes:
        world_size: Number of CP ranks.
        block_size: Q/KV block size used by block mask.
        token_permutation: Reorder input tokens before CP sharding.
        restore_permutation: Undo permutation after unshard.
        rank_plans: Per-rank step plans: every rank's in rank order by
            default, else those of the ``ranks`` given to
            :func:`build_ring_plan`; empty when skipped.
        reordered_block_mask: Optional reference-path block mask after token
            reordering.
    """

    world_size: int
    block_size: int
    token_permutation: Tensor
    restore_permutation: Tensor
    rank_plans: list[RankRingPlan]
    reordered_block_mask: BlockMask | None


@dataclass(kw_only=True, slots=True)
class PlannerSparseView:
    """Planner-only block-sparse metadata.

    Carries only the fields needed by PTRR cost estimation and per-rank step
    overlap planning, without requiring a full FlexAttention ``BlockMask``.
    """

    block_size: tuple[int, int]
    kv_num_blocks: Tensor
    kv_indices: Tensor
    full_kv_num_blocks: Tensor | None
    full_kv_indices: Tensor | None


def make_packed_layout_mask(layout: PackedLayout):
    """Create a FlexAttention mask_mod from packed layout metadata."""
    fn_seq = layout.full_and_noise_seq_id
    noise_seq = layout.noise_seq_id
    doc_ids = layout.document_id

    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    def full_and_noise_mask(b, h, q_idx, kv_idx):
        return (fn_seq[q_idx] == fn_seq[kv_idx]) & (fn_seq[q_idx] >= 0)

    def remove_noise_mask(b, h, q_idx, kv_idx):
        return ~((noise_seq[kv_idx] >= 0) & (noise_seq[q_idx] != noise_seq[kv_idx]))

    def sample_mask(b, h, q_idx, kv_idx):
        return doc_ids[q_idx] == doc_ids[kv_idx]

    return and_masks(
        or_masks(causal_mask, full_and_noise_mask),
        remove_noise_mask,
        sample_mask,
    )


# Every compiled ``create_block_mask`` shares one dynamo cache (one code object),
# and each new length or mask closure adds an entry. Raise the limits wherever it
# is called so long runs never hit dynamo's default of 8 recompiles.
CREATE_BLOCK_MASK_DYNAMO_CONFIG = {
    "accumulated_recompile_limit": 10000,
    "recompile_limit": 10000,
    "cache_size_limit": 10000,
    "accumulated_cache_size_limit": 10000,
}


def create_layout_block_mask(
    layout: PackedLayout,
    *,
    device: str | torch.device,
    block_size: int = 128,
) -> BlockMask:
    """Materialize a reference ``BlockMask`` for cost estimation and scheduling."""
    global CREATE_BLOCK_MASK
    sparse_mask = make_packed_layout_mask(layout)
    with torch._dynamo.config.patch(**CREATE_BLOCK_MASK_DYNAMO_CONFIG):
        if CREATE_BLOCK_MASK is None:
            CREATE_BLOCK_MASK = torch.compile(create_block_mask, dynamic=False)
        return CREATE_BLOCK_MASK(
            sparse_mask,
            B=None,
            H=None,
            Q_LEN=layout.seq_len,
            KV_LEN=layout.seq_len,
            device=device,
            BLOCK_SIZE=block_size,
        )


def _coerce_block_size(block_size: int | tuple[int, int]) -> tuple[int, int]:
    """Normalize block size values to a 2-tuple of ints."""
    if isinstance(block_size, int):
        return (block_size, block_size)
    return (int(block_size[0]), int(block_size[1]))


def _planner_sparse_view_from_source(
    source: PlannerSparseView | BlockMask | BlockSparseResult,
) -> PlannerSparseView:
    """Normalize a block-sparse source into planner-only metadata."""
    if isinstance(source, PlannerSparseView):
        return source

    block_size_attr = getattr(source, "BLOCK_SIZE", getattr(source, "block_size", 128))
    return PlannerSparseView(
        block_size=_coerce_block_size(block_size_attr),
        kv_num_blocks=source.kv_num_blocks,
        kv_indices=source.kv_indices,
        full_kv_num_blocks=source.full_kv_num_blocks,
        full_kv_indices=source.full_kv_indices,
    )


def _build_layout_planner_sparse_view(
    layout: PackedLayout,
    *,
    block_size: int = 128,
) -> PlannerSparseView:
    """Build planner block sparsity directly from a packed self-attention layout."""
    position_ids = torch.arange(
        layout.seq_len, dtype=torch.int64, device=layout.document_id.device
    )
    result = build_block_sparse_from_cross_qkv_aux(
        [
            layout.full_and_noise_seq_id.contiguous(),
            layout.full_and_noise_seq_id.contiguous(),
            layout.noise_seq_id.contiguous(),
            layout.noise_seq_id.contiguous(),
            layout.document_id.contiguous(),
            layout.document_id.contiguous(),
            position_ids,
            position_ids,
        ],
        block_size=block_size,
    )
    return _planner_sparse_view_from_source(result)


def _ptrr_scheduling(process_time: Tensor, group_size: int) -> Tensor:
    """Processing-Time based Round-Robin scheduling over 1D costs.

    Adapted from PyTorch's ``_PTRRLoadBalancer.ptrr_scheduling``
    (torch/distributed/tensor/experimental/_context_parallel/_load_balancer.py,
    BSD 3-Clause; see THIRD_PARTY_NOTICES.md).

    Raises:
        ValueError: If ``process_time`` is not 1-D, or its length is not divisible
            by ``group_size``.
    """
    if process_time.ndim != 1:
        raise ValueError(f"Expected 1D process_time, got {process_time.shape=}.")
    num_tasks = int(process_time.numel())
    if num_tasks % group_size != 0:
        raise ValueError(
            f"num_tasks must be divisible by group_size, got {num_tasks=} and "
            f"{group_size=}."
        )
    _, sorted_desc = torch.sort(process_time, descending=True, stable=True)
    sorted_desc_reversed = torch.flip(sorted_desc.view(-1, group_size), dims=[1]).view(
        -1
    )
    tasks_in_groups = torch.where(
        torch.arange(num_tasks, device=process_time.device) // group_size % 2 == 0,
        sorted_desc,
        sorted_desc_reversed,
    )
    tasks_in_groups = tasks_in_groups.view(-1, group_size).transpose(0, 1)
    tasks_in_groups, _ = torch.sort(tasks_in_groups, dim=1)
    return tasks_in_groups


def _non_sparse_kv_num_blocks(
    block_mask: PlannerSparseView | BlockMask | BlockSparseResult,
) -> Tensor:
    view = _planner_sparse_view_from_source(block_mask)
    full = view.full_kv_num_blocks
    if full is None:
        return view.kv_num_blocks
    return view.kv_num_blocks + full


def _ptrr_block_permutation(
    block_mask: PlannerSparseView | BlockMask | BlockSparseResult,
    world_size: int,
) -> Tensor:
    """Return block permutation using PTRR cost balancing.

    Raises:
        NotImplementedError: If the block mask has batch or head dimensions != 1.
    """
    non_sparse = _non_sparse_kv_num_blocks(block_mask)
    if non_sparse.shape[0] != 1 or non_sparse.shape[1] != 1:
        raise NotImplementedError(
            "This prototype supports only B=1, H=1 block masks for planning."
        )
    block_cost = non_sparse[0, 0].to(torch.float32)
    grouped = _ptrr_scheduling(block_cost, world_size)
    return grouped.reshape(-1).to(torch.long)


def _block_perm_to_token_perm(
    block_perm: Tensor, *, block_size: int, seq_len: int
) -> Tensor:
    """Expand a block permutation into a token permutation.

    Raises:
        NotImplementedError: If ``seq_len`` is not divisible by ``block_size``.
        ValueError: If the resulting permutation length doesn't match ``seq_len``.
    """
    if seq_len % block_size != 0:
        raise NotImplementedError(
            f"seq_len must be divisible by block_size, got {seq_len=} and {block_size=}."
        )
    offsets = torch.arange(block_size, dtype=torch.long, device=block_perm.device)
    token_perm = (block_perm.unsqueeze(1) * block_size + offsets.unsqueeze(0)).reshape(
        -1
    )
    if int(token_perm.numel()) != seq_len:
        raise ValueError(
            f"Token permutation length mismatch: {token_perm.numel()=} vs {seq_len=}."
        )
    return token_perm


def inverse_permutation(permutation: Tensor) -> Tensor:
    """Compute inverse indices such that x[permutation][inverse] == x."""
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(
        permutation.numel(), device=permutation.device, dtype=permutation.dtype
    )
    return inverse


def apply_token_permutation(
    x: Tensor, token_permutation: Tensor, *, seq_dim: int = 1
) -> Tensor:
    """Reorder a sequence tensor by token permutation."""
    return torch.index_select(x, dim=seq_dim, index=token_permutation.to(x.device))


def _count_overlap(row_indices: Tensor, count: int, start: int, end: int) -> int:
    if count <= 0:
        return 0
    vals = row_indices[:count]
    overlap = (vals >= start) & (vals < end)
    return int(overlap.sum().item())


def _build_rank_plan(
    *,
    block_mask: PlannerSparseView | BlockMask | BlockSparseResult,
    world_size: int,
    rank: int,
) -> RankRingPlan:
    view = _planner_sparse_view_from_source(block_mask)
    kv_num_blocks = view.kv_num_blocks[0, 0].cpu()
    kv_indices = view.kv_indices[0, 0].cpu()
    full_kv_num_blocks = (
        view.full_kv_num_blocks[0, 0].cpu()
        if view.full_kv_num_blocks is not None
        else None
    )
    full_kv_indices = (
        view.full_kv_indices[0, 0].cpu() if view.full_kv_indices is not None else None
    )
    q_blocks = int(kv_num_blocks.numel())
    if q_blocks % world_size != 0:
        raise ValueError(
            f"q_blocks must be divisible by world_size, got {q_blocks=} and {world_size=}."
        )
    blocks_per_rank = q_blocks // world_size
    local_start = rank * blocks_per_rank
    local_end = (rank + 1) * blocks_per_rank

    steps: list[RingStepPlan] = []
    for step in range(world_size):
        kv_owner = (rank - step) % world_size
        kv_start = kv_owner * blocks_per_rank
        kv_end = (kv_owner + 1) * blocks_per_rank
        active_blocks: list[int] = []
        estimated_intersections = 0
        for global_q in range(local_start, local_end):
            n_sparse = int(kv_num_blocks[global_q].item())
            local_intersections = _count_overlap(
                kv_indices[global_q], n_sparse, kv_start, kv_end
            )
            if full_kv_num_blocks is not None and full_kv_indices is not None:
                n_full = int(full_kv_num_blocks[global_q].item())
                local_intersections += _count_overlap(
                    full_kv_indices[global_q], n_full, kv_start, kv_end
                )
            if local_intersections > 0:
                active_blocks.append(global_q - local_start)
                estimated_intersections += local_intersections
        steps.append(
            RingStepPlan(
                step=step,
                kv_owner_rank=kv_owner,
                active_local_q_blocks=torch.tensor(active_blocks, dtype=torch.long),
                estimated_block_intersections=estimated_intersections,
            )
        )
    return RankRingPlan(
        rank=rank,
        local_block_start=local_start,
        local_block_end=local_end,
        steps=steps,
    )


def _build_rank_plan_vectorized(
    *,
    block_mask: PlannerSparseView | BlockMask | BlockSparseResult,
    world_size: int,
    rank: int,
) -> RankRingPlan:
    """Vectorized version of :func:`_build_rank_plan`.

    Replaces the per-row Python loop with batched tensor operations:

    1. Build a validity mask over ``kv_indices`` using ``kv_num_blocks``
       to zero out padding entries beyond each row's actual count.
    2. For each step, compute the KV block range ``[kv_start, kv_end)``
       and do a batched range-membership check across all local Q rows
       at once.
    3. Sum per-row to get intersection counts, then derive
       ``active_local_q_blocks`` and ``estimated_block_intersections``.

    Raises:
        ValueError: If ``q_blocks`` is not divisible by ``world_size``.
    """
    view = _planner_sparse_view_from_source(block_mask)
    kv_num_blocks = view.kv_num_blocks[0, 0].cpu()
    kv_indices = view.kv_indices[0, 0].cpu()
    q_blocks = int(kv_num_blocks.numel())

    if q_blocks % world_size != 0:
        raise ValueError(
            f"q_blocks must be divisible by world_size, got {q_blocks=} and {world_size=}."
        )
    blocks_per_rank = q_blocks // world_size
    local_start = rank * blocks_per_rank
    local_end = (rank + 1) * blocks_per_rank

    local_sparse_counts = kv_num_blocks[local_start:local_end]
    local_sparse_max = (
        int(local_sparse_counts.max().item()) if local_sparse_counts.numel() > 0 else 0
    )
    local_sparse_indices = kv_indices[local_start:local_end, :local_sparse_max]

    has_full = view.full_kv_num_blocks is not None and view.full_kv_indices is not None
    if has_full:
        full_kv_num_blocks = view.full_kv_num_blocks[0, 0].cpu()
        full_kv_indices_all = view.full_kv_indices[0, 0].cpu()
        local_full_counts = full_kv_num_blocks[local_start:local_end]
        local_full_max = (
            int(local_full_counts.max().item()) if local_full_counts.numel() > 0 else 0
        )
        local_full_indices = full_kv_indices_all[local_start:local_end, :local_full_max]

        # Build separate validity masks then concatenate, because valid
        # sparse entries are columns [0, sparse_max) and valid full entries
        # are columns [sparse_max, sparse_max + full_max).
        if local_sparse_max > 0:
            sparse_cols = torch.arange(local_sparse_max)
            sparse_valid = sparse_cols.unsqueeze(0) < local_sparse_counts.unsqueeze(1)
        else:
            sparse_valid = torch.zeros(blocks_per_rank, 0, dtype=torch.bool)
        if local_full_max > 0:
            full_cols = torch.arange(local_full_max)
            full_valid = full_cols.unsqueeze(0) < local_full_counts.unsqueeze(1)
        else:
            full_valid = torch.zeros(blocks_per_rank, 0, dtype=torch.bool)

        combined_indices = torch.cat([local_sparse_indices, local_full_indices], dim=1)
        valid_mask = torch.cat([sparse_valid, full_valid], dim=1)
    else:
        combined_indices = local_sparse_indices
        if local_sparse_max > 0:
            col_range = torch.arange(local_sparse_max, device=combined_indices.device)
            valid_mask = col_range.unsqueeze(0) < local_sparse_counts.unsqueeze(1)
        else:
            valid_mask = torch.zeros(
                blocks_per_rank, 0, dtype=torch.bool, device=combined_indices.device
            )

    steps: list[RingStepPlan] = []
    for step in range(world_size):
        kv_owner = (rank - step) % world_size
        kv_start = kv_owner * blocks_per_rank
        kv_end = (kv_owner + 1) * blocks_per_rank

        in_range = (
            valid_mask & (combined_indices >= kv_start) & (combined_indices < kv_end)
        )
        per_row = in_range.sum(dim=1)

        active_mask = per_row > 0
        active_local_q_blocks = active_mask.nonzero(as_tuple=False).squeeze(1)
        estimated_intersections = int(per_row.sum().item())

        steps.append(
            RingStepPlan(
                step=step,
                kv_owner_rank=kv_owner,
                active_local_q_blocks=active_local_q_blocks.to(torch.long),
                estimated_block_intersections=estimated_intersections,
            )
        )

    return RankRingPlan(
        rank=rank,
        local_block_start=local_start,
        local_block_end=local_end,
        steps=steps,
    )


def _build_local_rank_plan(
    layout: PackedLayout, *, block_size: int, world_size: int, rank: int
) -> RankRingPlan:
    """One rank's step plan, classifying only that rank's query blocks.

    Same result as :func:`_build_rank_plan_vectorized` on the layout's full
    planner view (:func:`_build_layout_planner_sparse_view`): a block pair's
    class depends only on its two blocks, and the KV side still spans the whole
    sequence. Costs ``1 / world_size`` of the full ``n_blocks**2`` grid.

    Raises:
        ValueError: If the block count is not divisible by ``world_size``.
    """
    q_blocks = layout.seq_len // block_size
    if q_blocks % world_size != 0:
        raise ValueError(
            f"q_blocks must be divisible by world_size, got {q_blocks=} and {world_size=}."
        )
    blocks_per_rank = q_blocks // world_size
    local_start = rank * blocks_per_rank
    local_end = (rank + 1) * blocks_per_rank
    q = slice(local_start * block_size, local_end * block_size)
    fn = layout.full_and_noise_seq_id.contiguous()
    noise = layout.noise_seq_id.contiguous()
    doc = layout.document_id.contiguous()
    pos = torch.arange(layout.seq_len, dtype=torch.int64, device=doc.device)
    block_type = build_block_sparse_from_cross_qkv_aux(
        [fn[q], fn, noise[q], noise, doc[q], doc, pos[q], pos],
        block_size=block_size,
        types_only=True,
    ).block_type
    visible = (block_type != BLOCK_EMPTY).cpu()
    steps: list[RingStepPlan] = []
    for step in range(world_size):
        kv_owner = (rank - step) % world_size
        kv_start = kv_owner * blocks_per_rank
        per_row = visible[:, kv_start : kv_start + blocks_per_rank].sum(dim=1)
        steps.append(
            RingStepPlan(
                step=step,
                kv_owner_rank=kv_owner,
                active_local_q_blocks=(per_row > 0)
                .nonzero(as_tuple=False)
                .squeeze(1)
                .to(torch.long),
                estimated_block_intersections=int(per_row.sum().item()),
            )
        )
    return RankRingPlan(
        rank=rank,
        local_block_start=local_start,
        local_block_end=local_end,
        steps=steps,
    )


# Rank plans (per-step block overlaps) are diagnostics only (trace labels), and
# need the reordered sequence's block grid (all of it, or each requested rank's
# rows); skip them above this many 128-token blocks (2M tokens).
_RANK_PLAN_MAX_BLOCKS = 16384
# Query blocks classified per chunk when computing PTRR costs, which bounds the
# planner's memory at chunk x n_blocks instead of n_blocks^2.
_COST_CHUNK_BLOCKS = 1024


def _ptrr_block_costs(
    layout: PackedLayout, *, block_size: int, position_ids: Tensor | None = None
) -> Tensor:
    """Non-empty KV blocks per query block (the PTRR cost), without the full grid.

    Same values as counting the rows of the layout's full block classification,
    computed one chunk of query blocks at a time. ``position_ids`` gives each
    token's causal position when the layout is stored in another order.
    """
    n = layout.seq_len
    if n % block_size != 0:
        raise NotImplementedError(
            f"seq_len must be divisible by {block_size}, got {n=}."
        )
    device = layout.document_id.device
    pos = (
        torch.arange(n, dtype=torch.int64, device=device)
        if position_ids is None
        else position_ids.to(device=device, dtype=torch.int64)
    )
    fn = layout.full_and_noise_seq_id.contiguous()
    noise = layout.noise_seq_id.contiguous()
    doc = layout.document_id.contiguous()
    # Documents are contiguous runs, and tiles across documents are empty:
    # each chunk of query blocks is classified against its documents' span
    # only (same counts, far less work for many or short documents).
    run = torch.cat([doc.new_zeros(1), (doc[1:] != doc[:-1]).cumsum(0)])
    _, run_len = torch.unique_consecutive(doc, return_counts=True)
    run_end = run_len.cumsum(0)
    run_start = run_end - run_len
    costs = []
    step = _COST_CHUNK_BLOCKS * block_size
    for q0 in range(0, n, step):
        q1 = min(q0 + step, n)
        q = slice(q0, q1)
        k0 = int(run_start[run[q0]]) // block_size * block_size
        k1 = -(-int(run_end[run[q1 - 1]]) // block_size) * block_size
        k = slice(k0, min(k1, n))
        result = build_block_sparse_from_cross_qkv_aux(
            [fn[q], fn[k], noise[q], noise[k], doc[q], doc[k], pos[q], pos[k]],
            block_size=block_size,
            types_only=True,
        )
        costs.append((result.block_type != 0).sum(dim=1))
    return torch.cat(costs).to(torch.float32)


def build_ring_plan(
    layout: PackedLayout,
    *,
    world_size: int,
    device: str | torch.device = "cuda",
    block_mask: BlockMask | None = None,
    use_direct_planner_block_sparse: bool = False,
    position_ids: Tensor | None = None,
    ranks: Sequence[int] | None = None,
) -> RingPlan:
    """Build a ring-friendly load-balance plan for mixed attention.

    This produces:

    1) A token permutation to apply before CP sharding.
    2) A per-rank, per-step block work plan for ring KV rotation.

    If ``block_mask`` is provided (the original-order BlockMask), it is
    reused for cost estimation instead of being rebuilt from the layout.
    If ``use_direct_planner_block_sparse`` is true, direct block-sparsity
    classification is used for planning wherever the reference ``BlockMask``
    is not required.

    The step plans only label profiler ranges. ``ranks`` builds them for those
    ranks alone (a rank's plan does not depend on which others are built; with
    the direct planner only its own query blocks are classified), and an empty
    sequence skips them; ``None`` builds every rank's.

    Raises:
        ValueError: If ``world_size`` < 1.
    """
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size=}.")

    if block_mask is not None:
        planner_sparse: PlannerSparseView | BlockMask | BlockSparseResult = block_mask
        block_size = int(_coerce_block_size(block_mask.BLOCK_SIZE)[0])
        block_perm = _ptrr_block_permutation(planner_sparse, world_size)
    elif use_direct_planner_block_sparse:
        block_size = 128
        block_cost = _ptrr_block_costs(
            layout, block_size=block_size, position_ids=position_ids
        )
        block_perm = _ptrr_scheduling(block_cost, world_size).reshape(-1).to(torch.long)
    else:
        block_mask = create_layout_block_mask(layout, device=device)
        planner_sparse = block_mask
        block_size = int(_coerce_block_size(block_mask.BLOCK_SIZE)[0])
        block_perm = _ptrr_block_permutation(planner_sparse, world_size)

    token_perm = _block_perm_to_token_perm(
        block_perm,
        block_size=block_size,
        seq_len=layout.seq_len,
    )
    restore_perm = inverse_permutation(token_perm)

    reordered_layout = layout.reordered(token_perm)
    if (ranks is not None and not ranks) or (
        use_direct_planner_block_sparse
        and (
            position_ids is not None
            or layout.seq_len // block_size > _RANK_PLAN_MAX_BLOCKS
        )
    ):
        # None requested, or skipped: rank plans assume index order (custom
        # positions) and very long sequences skip these diagnostics.
        return RingPlan(
            world_size=world_size,
            block_size=block_size,
            token_permutation=token_perm,
            restore_permutation=restore_perm,
            rank_plans=[],
            reordered_block_mask=None,
        )
    if use_direct_planner_block_sparse and ranks is not None:
        return RingPlan(
            world_size=world_size,
            block_size=block_size,
            token_permutation=token_perm,
            restore_permutation=restore_perm,
            rank_plans=[
                _build_local_rank_plan(
                    reordered_layout,
                    block_size=block_size,
                    world_size=world_size,
                    rank=rank,
                )
                for rank in ranks
            ],
            reordered_block_mask=None,
        )
    if use_direct_planner_block_sparse:
        reordered_planner_sparse = _build_layout_planner_sparse_view(
            reordered_layout, block_size=block_size
        )
        reordered_block_mask = None
    else:
        reordered_block_mask = create_layout_block_mask(
            reordered_layout, device=device, block_size=block_size
        )
        reordered_planner_sparse = reordered_block_mask
    rank_plans = [
        _build_rank_plan_vectorized(
            block_mask=reordered_planner_sparse, world_size=world_size, rank=rank
        )
        for rank in (range(world_size) if ranks is None else ranks)
    ]

    return RingPlan(
        world_size=world_size,
        block_size=block_size,
        token_permutation=token_perm,
        restore_permutation=restore_perm,
        rank_plans=rank_plans,
        reordered_block_mask=reordered_block_mask,
    )


def get_rank_plan(plan: RingPlan, *, rank: int) -> RankRingPlan:
    """Return a single rank plan with bounds checks.

    Raises:
        ValueError: If ``rank`` is out of range or the plan ordering is inconsistent.
    """
    if rank < 0 or rank >= len(plan.rank_plans):
        raise ValueError(
            f"rank out of range for plan: {rank=}, {len(plan.rank_plans)=}."
        )
    rank_plan = plan.rank_plans[rank]
    if rank_plan.rank != rank:
        raise ValueError(
            f"Rank mismatch in plan ordering: requested {rank=}, found {rank_plan.rank=}."
        )
    return rank_plan


# ---------------------------------------------------------------------------
# Cross Q/KV mask utilities for ring attention step masks.
#
# Backend-agnostic: the FA4 ring steps (flash4_ring_attn.py) and the KV gather
# (halo.py) build their step masks from these.
# ---------------------------------------------------------------------------


def build_cross_qkv_step_aux_tensors(
    *,
    global_aux_tensors: list[Tensor],
    q_start: int,
    q_len: int,
    kv_start: int,
    kv_len: int,
    global_position_ids: Tensor | None = None,
) -> list[Tensor]:
    """Build per-step aux tensors for cross Q/KV sparse mask.

    Slices global ``[fn_seq, noise_seq, doc_seq]`` into Q and KV windows
    and returns 8 tensors: ``[q_fn, kv_fn, q_noise, kv_noise, q_doc, kv_doc,
    q_pos, kv_pos]``.

    Raises:
        ValueError: If global_aux_tensors does not contain exactly 3 tensors, if
            slice parameters are invalid, or if slices exceed global aux length.
    """
    if len(global_aux_tensors) != 3:
        raise ValueError(
            f"Expected 3 global aux tensors [fn, noise, doc], got "
            f"{len(global_aux_tensors)=}."
        )
    fn_seq, noise_seq, doc_seq = global_aux_tensors
    if q_start < 0 or kv_start < 0 or q_len <= 0 or kv_len <= 0:
        raise ValueError(
            f"Invalid slice parameters: {q_start=}, {q_len=}, {kv_start=}, {kv_len=}."
        )
    if q_start + q_len > fn_seq.numel() or kv_start + kv_len > fn_seq.numel():
        raise ValueError(
            "Requested q/kv slices exceed global aux length: "
            f"{q_start + q_len=}, {kv_start + kv_len=}, {fn_seq.numel()=}."
        )
    if global_position_ids is not None:
        if global_position_ids.ndim != 1:
            raise ValueError(
                f"global_position_ids must be 1D, got {global_position_ids.shape=}."
            )
        if global_position_ids.numel() != fn_seq.numel():
            raise ValueError(
                "global_position_ids length mismatch with aux tensors: "
                f"{global_position_ids.numel()=} vs {fn_seq.numel()}."
            )

    device = fn_seq.device
    if global_position_ids is None:
        q_pos = torch.arange(q_start, q_start + q_len, dtype=torch.int64, device=device)
        kv_pos = torch.arange(
            kv_start, kv_start + kv_len, dtype=torch.int64, device=device
        )
    else:
        q_pos = (
            global_position_ids[q_start : q_start + q_len]
            .to(device=device)
            .contiguous()
        )
        kv_pos = (
            global_position_ids[kv_start : kv_start + kv_len]
            .to(device=device)
            .contiguous()
        )
    return [
        fn_seq[q_start : q_start + q_len].contiguous(),
        fn_seq[kv_start : kv_start + kv_len].contiguous(),
        noise_seq[q_start : q_start + q_len].contiguous(),
        noise_seq[kv_start : kv_start + kv_len].contiguous(),
        doc_seq[q_start : q_start + q_len].contiguous(),
        doc_seq[kv_start : kv_start + kv_len].contiguous(),
        q_pos,
        kv_pos,
    ]


def _python_cross_qkv_mask_from_aux(
    aux_tensors: list[Tensor], sliding_window: int | None = None
):
    """Build a Python mask callable matching cross Q/KV sparse semantics.

    Raises:
        ValueError: If aux_tensors does not contain exactly 8 tensors.
    """
    if len(aux_tensors) != 8:
        raise ValueError(
            f"Expected 8 aux tensors for cross q/kv mask, got {len(aux_tensors)=}."
        )
    q_fn, kv_fn, q_noise, kv_noise, q_doc, kv_doc, q_pos, kv_pos = aux_tensors

    def _mask(b, h, q_idx, kv_idx):
        del b, h
        causal = q_pos[q_idx] >= kv_pos[kv_idx]
        if sliding_window:
            causal = causal & (q_pos[q_idx] - kv_pos[kv_idx] < sliding_window)
        full_noise = (q_fn[q_idx] == kv_fn[kv_idx]) & (q_fn[q_idx] >= 0)
        remove_noise = ~((kv_noise[kv_idx] >= 0) & (q_noise[q_idx] != kv_noise[kv_idx]))
        same_doc = q_doc[q_idx] == kv_doc[kv_idx]
        return (causal | full_noise) & remove_noise & same_doc

    return _mask
