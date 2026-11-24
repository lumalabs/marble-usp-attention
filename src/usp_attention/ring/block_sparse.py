# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Direct block-sparsity classification from cross-QKV aux tensors.

Determines block-level full/partial/empty status without materializing a
dense ``(Q_LEN x KV_LEN)`` boolean mask.  For a 4096-token shard with
block_size=128, this evaluates a 32x32 block grid (1024 pairs) instead
of 16M element pairs -- roughly a 16,000x reduction in work.

The classification is conservative: borderline blocks are marked
"partial" so the attention kernel applies the element-level mask_mod.
Full blocks skip mask evaluation entirely. Empty blocks are skipped.

This module is backend-agnostic and usable by both Flash4 and Flex
attention backends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from usp_attention.fa4.flash4_block_sparse import (
    _dense_to_ordered,
)

if TYPE_CHECKING:
    from torch import Tensor
else:
    Tensor = torch.Tensor


BLOCK_EMPTY = 0
BLOCK_PARTIAL = 1
BLOCK_FULL = 2


@dataclass(kw_only=True, slots=True)
class BlockSparseResult:
    """Block-level sparsity classification.

    Attributes:
        n_q_blocks: Number of Q blocks.
        n_kv_blocks: Number of KV blocks.
        block_size: Tuple of (Q_BLOCK_SIZE, KV_BLOCK_SIZE).
        block_type: Dense (n_q_blocks, n_kv_blocks) tensor with values
            BLOCK_EMPTY (0), BLOCK_PARTIAL (1), or BLOCK_FULL (2); ``None``
            for compact results built without it.
        kv_num_blocks: BCSR row counts for partial+full blocks (forward).
        kv_indices: BCSR column indices for partial+full blocks (forward).
        full_kv_num_blocks: BCSR row counts for full-only blocks (forward).
        full_kv_indices: BCSR column indices for full-only blocks (forward).
        q_num_blocks: Transposed BCSR row counts (backward).
        q_indices: Transposed BCSR column indices (backward).
        full_q_num_blocks: Transposed full BCSR row counts (backward).
        full_q_indices: Transposed full BCSR column indices (backward).
    """

    n_q_blocks: int
    n_kv_blocks: int
    block_size: tuple[int, int]
    block_type: Tensor | None
    kv_num_blocks: Tensor
    kv_indices: Tensor
    full_kv_num_blocks: Tensor
    full_kv_indices: Tensor
    q_num_blocks: Tensor
    q_indices: Tensor
    full_q_num_blocks: Tensor
    full_q_indices: Tensor


def _pad_to_blocks(x: Tensor, block_size: int, fill_value: int = 0) -> Tensor:
    """Pad a 1D tensor so its length is a multiple of block_size."""
    remainder = x.numel() % block_size
    if remainder == 0:
        return x
    pad_len = block_size - remainder
    return torch.cat([x, x.new_full((pad_len,), fill_value)])


def _block_has_ids(ids_b: torch.Tensor, unique_ids: torch.Tensor) -> torch.Tensor:
    """``(blocks, len(unique_ids))`` 0 / 1 rows: which non-negative IDs each block holds."""
    has = torch.zeros(ids_b.shape[0], unique_ids.numel(), device=ids_b.device)
    valid = ids_b >= 0
    rows = torch.arange(ids_b.shape[0], device=ids_b.device)
    rows = rows.unsqueeze(1).expand_as(ids_b)[valid]
    has[rows, torch.searchsorted(unique_ids, ids_b[valid])] = 1.0
    return has


def build_block_sparse_from_cross_qkv_aux(
    aux_tensors: list[Tensor],
    block_size: int | tuple[int, int] = 128,
    *,
    types_only: bool = False,
    compact: bool = False,
    with_types: bool = True,
) -> BlockSparseResult:
    """Classify block sparsity directly from cross-QKV aux tensors.

    Takes the 8-element ``aux_tensors`` list ``[q_fn, kv_fn, q_noise,
    kv_noise, q_doc, kv_doc, q_pos, kv_pos]`` and returns block-sparse
    structure without materializing a dense Q*KV mask.

    The classification is conservative: blocks that cannot be proven
    full or empty are marked partial.

    Args:
        aux_tensors: 8-element list of 1D tensors.
        block_size: Block granularity or (query, KV) tile sizes (default 128).

        types_only: Return only ``block_type`` (index fields ``None``), e.g.
            for PTRR's tile costs.
        compact: Classify only block pairs whose document ranges meet (all
            others are empty) and return compact index lists, as wide as the
            longest row; each row's first ``count`` entries equal the dense
            lists', the only ones FA4 reads. Falls back to the dense grid when
            a block holds more than 32 distinct noise or full-span IDs or most
            pairs share a document (long documents).
        with_types: With ``compact``, also return the dense ``block_type``.

    Returns:
        BlockSparseResult with BCSR-format block indices for forward
        and backward.

    Raises:
        ValueError: If aux_tensors does not contain exactly 8 tensors.
    """
    if len(aux_tensors) != 8:
        raise ValueError(
            f"Expected 8 aux tensors for cross q/kv mask, got {len(aux_tensors)=}."
        )
    q_fn, kv_fn, q_noise, kv_noise, q_doc, kv_doc, q_pos, kv_pos = aux_tensors
    device = q_fn.device

    q_block_size, kv_block_size = (
        (block_size, block_size) if isinstance(block_size, int) else block_size
    )
    if q_block_size <= 0 or kv_block_size <= 0:
        raise ValueError("Block sizes must be positive")

    q_fn = _pad_to_blocks(q_fn, q_block_size, fill_value=-1)
    kv_fn = _pad_to_blocks(kv_fn, kv_block_size, fill_value=-1)
    q_noise = _pad_to_blocks(q_noise, q_block_size, fill_value=-1)
    kv_noise = _pad_to_blocks(kv_noise, kv_block_size, fill_value=-1)
    q_doc = _pad_to_blocks(q_doc, q_block_size)
    kv_doc = _pad_to_blocks(kv_doc, kv_block_size)
    q_pos = _pad_to_blocks(q_pos, q_block_size)
    kv_pos = _pad_to_blocks(kv_pos, kv_block_size)

    n_q_blocks = q_fn.numel() // q_block_size
    n_kv_blocks = kv_fn.numel() // kv_block_size

    # -- Per-block summary stats via reshape + reduce --
    q_fn_b = q_fn.reshape(n_q_blocks, q_block_size)
    kv_fn_b = kv_fn.reshape(n_kv_blocks, kv_block_size)
    q_noise_b = q_noise.reshape(n_q_blocks, q_block_size)
    kv_noise_b = kv_noise.reshape(n_kv_blocks, kv_block_size)
    q_doc_b = q_doc.reshape(n_q_blocks, q_block_size)
    kv_doc_b = kv_doc.reshape(n_kv_blocks, kv_block_size)
    q_pos_b = q_pos.reshape(n_q_blocks, q_block_size)
    kv_pos_b = kv_pos.reshape(n_kv_blocks, kv_block_size)

    q_doc_min = q_doc_b.min(dim=1).values
    q_doc_max = q_doc_b.max(dim=1).values
    kv_doc_min = kv_doc_b.min(dim=1).values
    kv_doc_max = kv_doc_b.max(dim=1).values

    q_fn_min = q_fn_b.min(dim=1).values
    q_fn_max = q_fn_b.max(dim=1).values
    kv_fn_min = kv_fn_b.min(dim=1).values
    kv_fn_max = kv_fn_b.max(dim=1).values

    q_noise_min = q_noise_b.min(dim=1).values
    q_noise_max = q_noise_b.max(dim=1).values
    kv_noise_min = kv_noise_b.min(dim=1).values
    kv_noise_max = kv_noise_b.max(dim=1).values

    q_pos_min = q_pos_b.min(dim=1).values
    q_pos_max = q_pos_b.max(dim=1).values
    kv_pos_min = kv_pos_b.min(dim=1).values
    kv_pos_max = kv_pos_b.max(dim=1).values

    if compact or types_only:
        result = _build_block_sparse_pairs(
            {
                "q_doc_min": q_doc_min,
                "q_doc_max": q_doc_max,
                "kv_doc_min": kv_doc_min,
                "kv_doc_max": kv_doc_max,
                "q_fn_min": q_fn_min,
                "q_fn_max": q_fn_max,
                "kv_fn_min": kv_fn_min,
                "kv_fn_max": kv_fn_max,
                "q_noise_min": q_noise_min,
                "q_noise_max": q_noise_max,
                "kv_noise_min": kv_noise_min,
                "kv_noise_max": kv_noise_max,
                "q_pos_min": q_pos_min,
                "q_pos_max": q_pos_max,
                "kv_pos_min": kv_pos_min,
                "kv_pos_max": kv_pos_max,
                "q_noise_b": q_noise_b,
                "kv_noise_b": kv_noise_b,
                "q_fn_b": q_fn_b,
                "kv_fn_b": kv_fn_b,
            },
            n_q_blocks,
            n_kv_blocks,
            block_size,
            types_only=types_only,
            with_types=with_types or not compact,
        )
        if result is not None:
            return result

    # -- Vectorized block-pair classification --
    # Broadcast to (n_q_blocks, n_kv_blocks) grid.
    # Q stats have shape (n_q, 1), KV stats have shape (1, n_kv).
    q_doc_min_g = q_doc_min.unsqueeze(1)
    q_doc_max_g = q_doc_max.unsqueeze(1)
    kv_doc_min_g = kv_doc_min.unsqueeze(0)
    kv_doc_max_g = kv_doc_max.unsqueeze(0)

    q_fn_min_g = q_fn_min.unsqueeze(1)
    q_fn_max_g = q_fn_max.unsqueeze(1)
    kv_fn_min_g = kv_fn_min.unsqueeze(0)
    kv_fn_max_g = kv_fn_max.unsqueeze(0)

    q_noise_min_g = q_noise_min.unsqueeze(1)
    q_noise_max_g = q_noise_max.unsqueeze(1)
    kv_noise_min_g = kv_noise_min.unsqueeze(0)
    kv_noise_max_g = kv_noise_max.unsqueeze(0)

    q_pos_min_g = q_pos_min.unsqueeze(1)
    q_pos_max_g = q_pos_max.unsqueeze(1)
    kv_pos_min_g = kv_pos_min.unsqueeze(0)
    kv_pos_max_g = kv_pos_max.unsqueeze(0)

    # Both blocks are single-document and share the same doc ID.
    q_single_doc = q_doc_min_g == q_doc_max_g
    kv_single_doc = kv_doc_min_g == kv_doc_max_g
    same_doc = q_single_doc & kv_single_doc & (q_doc_min_g == kv_doc_min_g)

    # Document ranges are completely disjoint.
    diff_doc = (q_doc_max_g < kv_doc_min_g) | (kv_doc_max_g < q_doc_min_g)

    # Full-attention segment: both blocks have the same fn_seq >= 0.
    q_fn_uniform = q_fn_min_g == q_fn_max_g
    kv_fn_uniform = kv_fn_min_g == kv_fn_max_g
    all_full_attn = (
        same_doc
        & q_fn_uniform
        & kv_fn_uniform
        & (q_fn_min_g >= 0)
        & (kv_fn_min_g >= 0)
        & (q_fn_min_g == kv_fn_min_g)
    )

    # ``remove_noise`` in the reference mask is asymmetric:
    #
    #     (kv_noise < 0) | (q_noise == kv_noise)
    #
    # For the block-level "every pair satisfies remove_noise" guarantee
    # required to classify a block as FULL, we need either:
    #   (a) every KV token has noise < 0 (``kv_no_noise``), or
    #   (b) every Q token has the same positive noise ID N and every
    #       positive KV noise ID is also N. Mixed KV blocks like
    #       {-1, N} are fine because kv_noise < 0 already passes.
    #
    # A set-level "KV positive noise IDs are a subset of Q's" check is
    # necessary but NOT sufficient on its own: e.g. Q={0,1}, KV={0}
    # has kv_subset_of_q=True, yet the pair (q_i with noise=1, kv_j
    # with noise=0) fails remove_noise. Subset must be combined with
    # a Q-uniform-positive requirement for elementwise correctness.
    kv_no_noise = kv_noise_max_g < 0

    all_noise = torch.cat([q_noise.reshape(-1), kv_noise.reshape(-1)])
    unique_noise_ids = all_noise[all_noise >= 0].unique()
    if unique_noise_ids.numel() == 0:
        kv_subset_of_q = torch.ones(
            n_q_blocks, n_kv_blocks, dtype=torch.bool, device=device
        )
        remove_noise_none = torch.zeros(
            n_q_blocks, n_kv_blocks, dtype=torch.bool, device=device
        )
    else:
        # Which noise IDs each block holds, as (blocks, ids) indicator rows,
        # then shared-ID counts by one matmul: memory stays O(blocks * ids +
        # n_q_blocks * n_kv_blocks) even with thousands of noise spans
        # (long interleaved generation), never (n_q, n_kv, ids).
        q_has_nid = _block_has_ids(q_noise_b, unique_noise_ids)  # (n_q_blocks, n_ids)
        kv_has_nid = _block_has_ids(
            kv_noise_b, unique_noise_ids
        )  # (n_kv_blocks, n_ids)
        # Counts are at most a block's token count, exact in fp16 / fp32.
        mm_dtype = torch.float16 if device.type == "cuda" else torch.float32
        shared = q_has_nid.to(mm_dtype) @ kv_has_nid.to(mm_dtype).T
        kv_count = kv_has_nid.sum(dim=1).to(mm_dtype)
        # KV's positive noise IDs must be a subset of Q's: all KV IDs present in Q.
        kv_subset_of_q = shared == kv_count.unsqueeze(0)

        # remove_noise_none: every KV token is noise-tagged AND no KV noise ID
        # appears in Q.  (kv_noise >= 0 everywhere) & (no shared noise IDs).
        kv_all_noise = kv_noise_min_g >= 0
        noise_sets_disjoint = shared == 0
        remove_noise_none = kv_all_noise & noise_sets_disjoint

    # Elementwise-correct "remove_noise holds for every (q, kv) pair in the
    # block": either KV has no noise tokens, or Q is uniform on a single
    # positive noise ID and every positive KV noise ID equals it (enforced
    # via kv_subset_of_q when combined with Q uniformity).
    q_noise_uniform = q_noise_min_g == q_noise_max_g
    q_single_positive_noise = q_noise_uniform & (q_noise_min_g >= 0)
    remove_noise_all_elementwise = kv_no_noise | (
        q_single_positive_noise & kv_subset_of_q
    )

    full_attn_safe = all_full_attn & remove_noise_all_elementwise

    # Below diagonal: causal passes for ALL element pairs when
    # q_pos_min > kv_pos_max. The mask is (causal | full_noise) &
    # remove_noise & same_doc. Since causal is true everywhere, the
    # result is just remove_noise & same_doc. The block is full iff
    # remove_noise is true for EVERY element pair -- same elementwise
    # condition as above.
    causal_below = same_doc & (q_pos_min_g > kv_pos_max_g)
    below_diagonal_full = causal_below & remove_noise_all_elementwise

    # If full-attention IDs cannot possibly match anywhere in the block,
    # then an above-diagonal block is provably empty because both causal
    # and full_noise fail for every element pair.
    #
    # Range-based disjointness is fast but misses mixed boundary blocks
    # where fn_seq ranges overlap without any actual ID match (e.g.
    # q_fn={0}, kv_fn={-1, 3} -- range [0,0] vs [-1,3] overlaps but
    # the positive-ID sets {0} and {3} are disjoint).  We refine with
    # a per-block membership bitmap: for each unique positive fn_seq ID,
    # mark which Q/KV blocks contain it, then check actual set overlap
    # via matrix multiply.
    fn_ranges_disjoint = (q_fn_max_g < kv_fn_min_g) | (kv_fn_max_g < q_fn_min_g)
    no_full_noise_range = (q_fn_max_g < 0) | (kv_fn_max_g < 0) | fn_ranges_disjoint

    all_fn = torch.cat([q_fn.reshape(-1), kv_fn.reshape(-1)])
    unique_fn_ids = all_fn[all_fn >= 0].unique()
    if unique_fn_ids.numel() == 0:
        no_full_noise_possible = no_full_noise_range
    else:
        # Per-block indicator rows (O(blocks * ids) memory, never
        # (blocks, block_size, ids)); any shared positive fn_seq ID -> matmul > 0.
        q_has_id = _block_has_ids(q_fn_b, unique_fn_ids)
        kv_has_id = _block_has_ids(kv_fn_b, unique_fn_ids)
        mm_dtype = torch.float16 if device.type == "cuda" else torch.float32
        fn_overlap = (q_has_id.to(mm_dtype) @ kv_has_id.to(mm_dtype).T) > 0
        no_full_noise_possible = no_full_noise_range | ~fn_overlap

    # Above diagonal: q_pos_max < kv_pos_min means causal fails for ALL
    # elements. The block is empty if full_noise is impossible everywhere,
    # or if remove_noise is false everywhere.
    causal_above = q_pos_max_g < kv_pos_min_g
    above_diagonal_empty = causal_above & (no_full_noise_possible | remove_noise_none)

    # -- Assemble block_type grid --
    block_type = torch.full(
        (n_q_blocks, n_kv_blocks), BLOCK_PARTIAL, dtype=torch.int32, device=device
    )
    block_type[diff_doc] = BLOCK_EMPTY
    block_type[remove_noise_none] = BLOCK_EMPTY
    block_type[above_diagonal_empty] = BLOCK_EMPTY
    block_type[full_attn_safe] = BLOCK_FULL
    block_type[below_diagonal_full] = BLOCK_FULL

    if types_only:
        return BlockSparseResult(
            n_q_blocks=n_q_blocks,
            n_kv_blocks=n_kv_blocks,
            block_size=(q_block_size, kv_block_size),
            block_type=block_type,
            kv_num_blocks=None,
            kv_indices=None,
            full_kv_num_blocks=None,
            full_kv_indices=None,
            q_num_blocks=None,
            q_indices=None,
            full_q_num_blocks=None,
            full_q_indices=None,
        )

    if compact:
        # Dense-ish step: compact lists straight from the grid (row-major
        # nonzero is already sorted by row, then column).
        lists = []
        for value in (BLOCK_PARTIAL, BLOCK_FULL):
            hit = block_type == value
            fwd = hit.nonzero()
            bwd = hit.T.nonzero()
            lists.append(
                (
                    _ordered_compact(fwd[:, 0], fwd[:, 1], n_q_blocks),
                    _ordered_compact(bwd[:, 0], bwd[:, 1], n_kv_blocks),
                )
            )
        (kv_num, kv_ind), (q_num, q_ind) = lists[0]
        (full_kv_num, full_kv_ind), (full_q_num, full_q_ind) = lists[1]
        return BlockSparseResult(
            n_q_blocks=n_q_blocks,
            n_kv_blocks=n_kv_blocks,
            block_size=(block_size, block_size),
            block_type=block_type if with_types else None,
            kv_num_blocks=kv_num,
            kv_indices=kv_ind,
            full_kv_num_blocks=full_kv_num,
            full_kv_indices=full_kv_ind,
            q_num_blocks=q_num,
            q_indices=q_ind,
            full_q_num_blocks=full_q_num,
            full_q_indices=full_q_ind,
        )

    # -- Convert to BCSR --
    # BlockMask convention: kv_num_blocks/kv_indices contains partial-only
    # blocks (need mask_mod evaluation), full_kv_num_blocks/full_kv_indices
    # contains full-only blocks (skip mask_mod). These are disjoint sets.
    partial_mask = (
        (block_type == BLOCK_PARTIAL).to(torch.int8).unsqueeze(0).unsqueeze(0)
    )
    full_mask = (block_type == BLOCK_FULL).to(torch.int8).unsqueeze(0).unsqueeze(0)

    kv_num_blocks, kv_indices = _dense_to_ordered(partial_mask)
    full_kv_num_blocks, full_kv_indices = _dense_to_ordered(full_mask)

    # Q-indexed (backward) lists straight from the transposed masks.
    q_num_blocks, q_indices = _dense_to_ordered(partial_mask.transpose(-2, -1))
    full_q_num_blocks, full_q_indices = _dense_to_ordered(full_mask.transpose(-2, -1))

    return BlockSparseResult(
        n_q_blocks=n_q_blocks,
        n_kv_blocks=n_kv_blocks,
        block_size=(q_block_size, kv_block_size),
        block_type=block_type,
        kv_num_blocks=kv_num_blocks,
        kv_indices=kv_indices,
        full_kv_num_blocks=full_kv_num_blocks,
        full_kv_indices=full_kv_indices,
        q_num_blocks=q_num_blocks,
        q_indices=q_indices,
        full_q_num_blocks=full_q_num_blocks,
        full_q_indices=full_q_indices,
    )


# A block holds at most this many distinct noise / full-span IDs for the
# pair-wise path; layouts with more fall back to the dense grid.
_MAX_ID_SLOTS = 32
# Above this share of candidate pairs, the dense grid classifies faster.
_MAX_PAIR_DENSITY = 0.3
# Pairs classified per chunk (bounds the (pairs, slots, slots) comparisons).
_PAIR_CHUNK = 1 << 22


def _id_slots(ids_b: Tensor) -> Tensor | None:
    """Each block's distinct non-negative IDs, ascending, padded with -1.

    Returns ``(blocks, K)`` with ``K`` the most distinct IDs in any block (at
    least 1), or ``None`` above :data:`_MAX_ID_SLOTS`.
    """
    s = ids_b.sort(dim=1).values
    first = torch.ones_like(s, dtype=torch.bool)
    first[:, 1:] = s[:, 1:] != s[:, :-1]
    valid = first & (s >= 0)
    k = int(valid.sum(dim=1).max()) if s.shape[0] else 0
    if k > _MAX_ID_SLOTS:
        return None
    out = torch.full((s.shape[0], max(k, 1)), -1, dtype=s.dtype, device=s.device)
    rank = valid.cumsum(dim=1) - 1
    rows = torch.arange(s.shape[0], device=s.device).unsqueeze(1).expand_as(s)
    out[rows[valid], rank[valid]] = s[valid]
    return out


def _shared_ids(q_slots: Tensor, kv_slots: Tensor) -> Tensor:
    """Per pair: how many of the KV block's IDs the Q block also holds."""
    eq = (q_slots.unsqueeze(2) == kv_slots.unsqueeze(1)) & (kv_slots >= 0).unsqueeze(1)
    return eq.any(dim=1).sum(dim=1)


def _ordered_compact(rows: Tensor, cols: Tensor, n_rows: int) -> tuple[Tensor, Tensor]:
    """Counts and row-wise ascending column lists of sorted ``(rows, cols)`` pairs.

    The index tensor is ``(1, 1, n_rows, width)`` with ``width`` the longest
    row (at least 1); entries past a row's count are 0. FA4 reads only the
    first ``count`` entries of each row, which equal those of
    :func:`~usp_attention.fa4.flash4_block_sparse._dense_to_ordered`.
    """
    counts = torch.bincount(rows, minlength=n_rows).to(torch.int32)
    width = max(int(counts.max()) if n_rows else 0, 1)
    starts = torch.cumsum(counts, 0, dtype=torch.int64) - counts.to(torch.int64)
    pos = torch.arange(rows.numel(), device=rows.device) - starts[rows]
    idx = torch.zeros(n_rows * width, dtype=torch.int32, device=rows.device)
    idx[rows.to(torch.int64) * width + pos] = cols.to(torch.int32)
    idx = idx.view(n_rows, width)
    return counts.view(1, 1, -1).contiguous(), idx.view(
        1, 1, n_rows, width
    ).contiguous()


def _build_block_sparse_pairs(
    blocks: dict[str, Tensor],
    n_q_blocks: int,
    n_kv_blocks: int,
    block_size: int,
    *,
    types_only: bool,
    with_types: bool,
) -> BlockSparseResult | None:
    """:func:`build_block_sparse_from_cross_qkv_aux` over candidate pairs only.

    Pairs whose document ranges are disjoint are empty; the others are found
    by a sorted interval search and classified with exactly the dense path's
    rules, so block types (and every row's first ``count`` indices) match.
    Returns ``None`` when a block holds too many distinct IDs, or when more
    than :data:`_MAX_PAIR_DENSITY` of all pairs are candidates.
    """
    b = blocks
    device = b["q_doc_min"].device
    noise_ids = (b["q_noise_b"] >= 0).any() | (b["kv_noise_b"] >= 0).any()
    fn_ids = (b["q_fn_b"] >= 0).any() | (b["kv_fn_b"] >= 0).any()
    slots = {}
    for name, use in (("noise", noise_ids), ("fn", fn_ids)):
        if bool(use):
            q_s, kv_s = _id_slots(b[f"q_{name}_b"]), _id_slots(b[f"kv_{name}_b"])
            if q_s is None or kv_s is None:
                return None
            slots[name] = (q_s, kv_s)

    # Candidate pairs: KV blocks whose document range meets the Q block's.
    order = torch.argsort(b["kv_doc_min"], stable=True)
    dmin_sorted = b["kv_doc_min"][order]
    span = int((b["kv_doc_max"] - b["kv_doc_min"]).max()) if n_kv_blocks else 0
    lo = torch.searchsorted(dmin_sorted, b["q_doc_min"] - span, side="left")
    hi = torch.searchsorted(dmin_sorted, b["q_doc_max"], side="right")
    n_cand = (hi - lo).clamp_min(0)
    # Dense-ish steps (long documents) are faster on the dense grid.
    if int(n_cand.sum()) > _MAX_PAIR_DENSITY * n_q_blocks * n_kv_blocks:
        return None
    q_idx = torch.repeat_interleave(torch.arange(n_q_blocks, device=device), n_cand)
    offs = torch.arange(q_idx.numel(), device=device) - torch.repeat_interleave(
        torch.cumsum(n_cand, 0) - n_cand, n_cand
    )
    kv_idx = order[lo[q_idx] + offs]
    keep = b["kv_doc_max"][kv_idx] >= b["q_doc_min"][q_idx]
    q_idx, kv_idx = q_idx[keep], kv_idx[keep]

    types = torch.empty(q_idx.numel(), dtype=torch.int32, device=device)
    for start in range(0, q_idx.numel(), _PAIR_CHUNK):
        qi, ki = q_idx[start : start + _PAIR_CHUNK], kv_idx[start : start + _PAIR_CHUNK]

        def q(name: str) -> Tensor:
            return b[f"q_{name}"][qi]

        def kv(name: str) -> Tensor:
            return b[f"kv_{name}"][ki]

        same_doc = (
            (q("doc_min") == q("doc_max"))
            & (kv("doc_min") == kv("doc_max"))
            & (q("doc_min") == kv("doc_min"))
        )
        all_full_attn = (
            same_doc
            & (q("fn_min") == q("fn_max"))
            & (kv("fn_min") == kv("fn_max"))
            & (q("fn_min") >= 0)
            & (kv("fn_min") >= 0)
            & (q("fn_min") == kv("fn_min"))
        )
        kv_no_noise = kv("noise_max") < 0
        if "noise" in slots:
            q_s, kv_s = slots["noise"]
            shared = _shared_ids(q_s[qi], kv_s[ki])
            kv_count = (kv_s[ki] >= 0).sum(dim=1)
            kv_subset_of_q = shared == kv_count
            remove_noise_none = (kv("noise_min") >= 0) & (shared == 0)
        else:
            kv_subset_of_q = torch.ones_like(same_doc)
            remove_noise_none = torch.zeros_like(same_doc)
        q_single_positive_noise = (q("noise_min") == q("noise_max")) & (
            q("noise_min") >= 0
        )
        remove_noise_all = kv_no_noise | (q_single_positive_noise & kv_subset_of_q)
        full_attn_safe = all_full_attn & remove_noise_all
        below_full = same_doc & (q("pos_min") > kv("pos_max")) & remove_noise_all
        fn_ranges_disjoint = (q("fn_max") < kv("fn_min")) | (kv("fn_max") < q("fn_min"))
        no_full_noise = (q("fn_max") < 0) | (kv("fn_max") < 0) | fn_ranges_disjoint
        if "fn" in slots:
            q_s, kv_s = slots["fn"]
            no_full_noise = no_full_noise | (_shared_ids(q_s[qi], kv_s[ki]) == 0)
        above_empty = (q("pos_max") < kv("pos_min")) & (
            no_full_noise | remove_noise_none
        )
        t = torch.full_like(qi, BLOCK_PARTIAL, dtype=torch.int32)
        t[remove_noise_none | above_empty] = BLOCK_EMPTY
        t[full_attn_safe | below_full] = BLOCK_FULL
        types[start : start + qi.numel()] = t

    block_type = None
    if types_only or with_types:
        block_type = torch.full(
            (n_q_blocks, n_kv_blocks), BLOCK_EMPTY, dtype=torch.int32, device=device
        )
        block_type[q_idx, kv_idx] = types
    if types_only:
        return BlockSparseResult(
            n_q_blocks=n_q_blocks,
            n_kv_blocks=n_kv_blocks,
            block_size=(block_size, block_size),
            block_type=block_type,
            kv_num_blocks=None,
            kv_indices=None,
            full_kv_num_blocks=None,
            full_kv_indices=None,
            q_num_blocks=None,
            q_indices=None,
            full_q_num_blocks=None,
            full_q_indices=None,
        )

    lists = {}
    for kind, value in (("partial", BLOCK_PARTIAL), ("full", BLOCK_FULL)):
        sel = types == value
        qs, ks = q_idx[sel], kv_idx[sel]
        by_q = torch.argsort(qs * n_kv_blocks + ks)
        by_kv = torch.argsort(ks * n_q_blocks + qs)
        lists[kind] = (
            _ordered_compact(qs[by_q], ks[by_q], n_q_blocks),
            _ordered_compact(ks[by_kv], qs[by_kv], n_kv_blocks),
        )
    (kv_num, kv_ind), (q_num, q_ind) = lists["partial"]
    (full_kv_num, full_kv_ind), (full_q_num, full_q_ind) = lists["full"]
    return BlockSparseResult(
        n_q_blocks=n_q_blocks,
        n_kv_blocks=n_kv_blocks,
        block_size=(block_size, block_size),
        block_type=block_type,
        kv_num_blocks=kv_num,
        kv_indices=kv_ind,
        full_kv_num_blocks=full_kv_num,
        full_kv_indices=full_kv_ind,
        q_num_blocks=q_num,
        q_indices=q_ind,
        full_q_num_blocks=full_q_num,
        full_q_indices=full_q_ind,
    )
